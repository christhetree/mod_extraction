import logging
import os
from contextlib import nullcontext
from typing import Dict, Optional

import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T
from torch import nn
from torch.optim import Optimizer

from lfo_tcn.losses import get_loss_func_by_name
from lfo_tcn.models import HiddenStateModel
from lfo_tcn.util import linear_interpolate_last_dim

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class BaseLightingModule(pl.LightningModule):
    default_loss_dict = {"l1": 1.0, "mse": 0.0}

    def __init__(self,
                 loss_dict: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        if loss_dict is None:
            loss_dict = self.default_loss_dict
        self.loss_dict = loss_dict
        self.loss_funcs = nn.ParameterList([get_loss_func_by_name(name) for name, _ in loss_dict.items()])

    def calc_and_log_losses(self, y_hat: T, y: T, prefix: str, should_log: bool = True) -> T:
        loss_values = [loss_func(y_hat, y) for loss_func in self.loss_funcs]
        loss = None
        for (name, weighting), loss_value in zip(self.loss_dict.items(), loss_values):
            if should_log:
                # TODO(cm): log batch size
                self.log(
                    f"{prefix}/{name}",
                    loss_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
            if weighting > 0:
                if loss is None:
                    loss = weighting * loss_value
                else:
                    loss += weighting * loss_value
        if should_log:
            self.log(
                f"{prefix}/loss",
                loss,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        return loss


class LFOExtraction(BaseLightingModule):
    def __init__(self,
                 model: nn.Module,
                 sr: float,
                 sub_batch_size: Optional[int] = None,
                 loss_dict: Optional[Dict[str, float]] = None) -> None:
        super().__init__(loss_dict)
        self.save_hyperparameters(ignore=["model"])
        log.info(f"\n{self.hparams}")
        self.model = model
        self.sr = sr
        self.sub_batch_size = sub_batch_size

    def forward(self, x: T) -> T:
        return self.model(x)

    def common_step(self,
                    batch: (Optional[T], T, T, Dict[str, T]),
                    is_training: bool) -> (T, Dict[str, T], Dict[str, T]):
        prefix = "train" if is_training else "val"
        dry, wet, mod_sig, fx_params = batch
        mod_sig_hat = self.model(wet).squeeze(1)
        mod_sig = linear_interpolate_last_dim(mod_sig, mod_sig_hat.size(-1), align_corners=True)
        assert mod_sig.shape == mod_sig_hat.shape

        loss = self.calc_and_log_losses(mod_sig_hat, mod_sig, prefix)

        data_dict = {
            "wet": wet.detach().float().cpu(),
            "mod_sig": mod_sig.detach().float().cpu(),
            "mod_sig_hat": mod_sig_hat.detach().float().cpu(),
        }
        if dry is not None:
            data_dict["dry"] = dry.detach().float().cpu(),

        fx_params = {k: v.detach().float().cpu() if isinstance(v, T) else v for k, v in fx_params.items()}

        # TODO(cm)
        # for idx, (d, w, m_h) in enumerate(zip(data_dict["dry"],
        #                                       data_dict["wet"],
        #                                       data_dict["mod_sig_hat"])):
        #     # if "mod_sig" in data_dict:
        #     #     m = data_dict["mod_sig"][idx]
        #     #     plt.plot(m)
        #     plt.plot(m_h)
        #     plt.title(f"mod_sig_{idx}")
        #     plt.show()
        #     # plot_spectrogram(d, title=f"dry_{idx}", save_name=f"dry_{idx}", sr=self.sr)
        #     plot_spectrogram(w, title=f"wet_{idx}", save_name=f"wet_{idx}", sr=self.sr)

        return loss, data_dict, fx_params

    def sub_batch_size_common_step(self,
                                   batch: (Optional[T], T, T, Dict[str, T]),
                                   is_training: bool) -> (T, Dict[str, T], Dict[str, T]):
        dry, wet, mod_sig, fx_params = batch
        inferred_bs = mod_sig.size(0)
        assert inferred_bs >= self.sub_batch_size
        assert inferred_bs % self.sub_batch_size == 0
        losses = []
        out_data_dict = None
        out_fx_params = None
        for start_idx in range(0, inferred_bs, self.sub_batch_size):
            end_idx = start_idx + self.sub_batch_size
            sub_dry = None
            if dry is not None:
                sub_dry = dry[start_idx:end_idx, ...]
            sub_wet = wet[start_idx:end_idx, ...]
            sub_mod_sig = mod_sig[start_idx:end_idx, ...]
            sub_fx_params = {k: v[start_idx:end_idx, ...] for k, v in fx_params.items()}
            sub_batch = (sub_dry, sub_wet, sub_mod_sig, sub_fx_params)
            loss, out_data_dict, out_fx_params = self.common_step(sub_batch, is_training=is_training)
            losses.append(loss)
        assert losses
        assert out_data_dict is not None
        assert out_fx_params is not None
        loss = tr.stack(losses, dim=0).mean(dim=0)
        return loss, out_data_dict, out_fx_params

    def training_step(self, batch: (T, T, T, Dict[str, T]), batch_idx: int) -> T:
        if self.sub_batch_size is None:
            loss, _, _ = self.common_step(batch, is_training=True)
        else:
            loss, _, _ = self.sub_batch_size_common_step(batch, is_training=True)
        return loss

    def validation_step(self, batch: (T, T, T, Dict[str, T]), batch_idx: int) -> (T, Dict[str, T], Dict[str, T]):
        if self.sub_batch_size is None:
            loss, data_dict, fx_params = self.common_step(batch, is_training=False)
        else:
            loss, data_dict, fx_params = self.sub_batch_size_common_step(batch, is_training=False)
        return loss, data_dict, fx_params


class LFOEffectModeling(BaseLightingModule):
    default_loss_dict = {"l1": 1.0, "esr": 0.0, "dc": 0.0}

    def __init__(self,
                 effect_model: nn.Module,
                 lfo_model: Optional[nn.Module] = None,
                 lfo_model_weights_path: Optional[str] = None,
                 freeze_lfo_model: bool = False,
                 sr: float = 44100,
                 loss_dict: Optional[Dict[str, float]] = None) -> None:
        super().__init__(loss_dict)
        self.freeze_lfo_model = freeze_lfo_model
        self.sr = sr
        self.use_gt_mod_sig = lfo_model is None
        self.save_hyperparameters(ignore=["effect_model", "lfo_model", "param_model"])
        log.info(f"\n{self.hparams}")
        self.effect_model = effect_model

        if lfo_model is not None:
            if lfo_model_weights_path is not None:
                log.info("Loading LFO model weights")
                assert os.path.isfile(lfo_model_weights_path)
                lfo_model.load_state_dict(tr.load(lfo_model_weights_path))
            if freeze_lfo_model:
                log.info("Freezing LFO model")
                lfo_model.eval()
                for param in lfo_model.parameters():
                    param.requires_grad = False
        else:
            log.info("Using ground truth mod_sig")

        self.lfo_model = lfo_model

    def extract_mod_sig(self, wet: T, mod_sig: Optional[T] = None) -> (T, Optional[T]):
        with tr.no_grad() if self.lfo_model is None or self.freeze_lfo_model else nullcontext():
            if self.lfo_model is None:
                assert mod_sig is not None
                assert mod_sig.ndim == 2
                mod_sig_hat = mod_sig
            else:
                mod_sig_hat = self.lfo_model(wet).squeeze(1)

            if mod_sig is not None and mod_sig.size(-1) != mod_sig_hat.size(-1):
                mod_sig = linear_interpolate_last_dim(mod_sig, mod_sig_hat.size(-1), align_corners=True)
                assert mod_sig.shape == mod_sig_hat.shape
            return mod_sig_hat, mod_sig

    def common_step(self,
                    batch: (T, T, Optional[T], Optional[Dict[str, T]]),
                    is_training: bool) -> (T, Dict[str, T], Optional[Dict[str, T]]):
        prefix = "train" if is_training else "val"
        dry, wet, mod_sig, fx_params = batch
        assert dry.size(-1) == wet.size(-1)

        mod_sig_hat, mod_sig = self.extract_mod_sig(wet, mod_sig)
        mod_sig_hat_sr = linear_interpolate_last_dim(mod_sig_hat, dry.size(-1), align_corners=True)
        mod_sig_hat_sr = mod_sig_hat_sr.unsqueeze(1)

        if isinstance(self.effect_model, HiddenStateModel):
            self.effect_model.detach_hidden()
        wet_hat = self.effect_model(dry, mod_sig_hat_sr)
        assert dry.shape == wet.shape == wet_hat.shape

        loss = self.calc_and_log_losses(wet_hat, wet, prefix)

        data_dict = {
            "dry": dry.detach().float().cpu(),
            "wet": wet.detach().float().cpu(),
            "wet_hat": wet_hat.detach().float().cpu(),
            "mod_sig_hat": mod_sig_hat.detach().float().cpu(),
        }
        if mod_sig is not None:
            data_dict["mod_sig"] = mod_sig.detach().float().cpu()

        if fx_params is not None:
            fx_params = {k: v.detach().float().cpu() for k, v in fx_params.items()}
        return loss, data_dict, fx_params

    def training_step(self, batch: (T, T, T, Dict[str, T]), batch_idx: int) -> T:
        loss, _, _ = self.common_step(batch, is_training=True)
        return loss

    def validation_step(self,
                        batch: (T, T, T, Dict[str, T]),
                        batch_idx: int) -> (T, Dict[str, T], Optional[Dict[str, T]]):
        return self.common_step(batch, is_training=False)


class TBPTTLFOEffectModeling(LFOEffectModeling):
    def __init__(self,
                 warmup_n_samples: int,
                 step_n_samples: int,
                 effect_model: HiddenStateModel,
                 lfo_model: Optional[nn.Module] = None,
                 lfo_model_weights_path: Optional[str] = None,
                 freeze_lfo_model: bool = True,
                 param_model: Optional[nn.Module] = None,
                 sr: float = 44100,
                 loss_dict: Optional[Dict[str, float]] = None) -> None:
        assert warmup_n_samples > 0
        self.warmup_n_samples = warmup_n_samples
        self.step_n_samples = step_n_samples
        super().__init__(effect_model,
                         lfo_model,
                         lfo_model_weights_path,
                         freeze_lfo_model=freeze_lfo_model,
                         sr=sr,
                         loss_dict=loss_dict)
        self.param_model = param_model
        self.automatic_optimization = False

    def common_step(self,
                    batch: (T, T, Optional[T], Optional[Dict[str, T]]),
                    is_training: bool) -> (T, Dict[str, T], Optional[Dict[str, T]]):
        opt: Optimizer = self.optimizers()
        if is_training:
            assert not isinstance(opt, list), "Only supports 1 optimizer for now"

        prefix = "train" if is_training else "val"
        dry, wet, mod_sig, fx_params = batch
        assert dry.size(-1) == wet.size(-1)
        assert dry.size(-1) >= self.warmup_n_samples + self.step_n_samples

        mod_sig_hat, mod_sig = self.extract_mod_sig(wet, mod_sig)
        mod_sig_hat_sr = linear_interpolate_last_dim(mod_sig_hat, dry.size(-1), align_corners=True)
        mod_sig_hat_sr = mod_sig_hat_sr.unsqueeze(1)

        # fb_param = fx_params["feedback"].float().view(-1, 1, 1)
        # fb_param = fb_param.repeat(1, 1, mod_sig_hat_sr.size(-1))
        # mod_sig_hat_sr = tr.cat([mod_sig_hat_sr, fb_param], dim=1)

        self.effect_model.clear_hidden()
        warmup_latent_sr = mod_sig_hat_sr[:, :, :self.warmup_n_samples]

        param_latent = None
        if self.param_model is not None:
            param_latent = self.param_model(wet).unsqueeze(-1)
            warmup_param_latent_sr = param_latent.repeat(1, 1, self.warmup_n_samples)
            warmup_latent_sr = tr.cat([warmup_latent_sr, warmup_param_latent_sr], dim=1)

        warmup_dry = dry[:, :, :self.warmup_n_samples]
        warmup_wet_hat = self.effect_model(warmup_dry, warmup_latent_sr)
        if is_training:
            self.effect_model.detach_hidden()
            opt.zero_grad()

        wet_hat_chunks = [warmup_wet_hat]
        for start_idx in range(self.warmup_n_samples, dry.size(-1), self.step_n_samples):
            end_idx = start_idx + self.step_n_samples
            if end_idx > dry.size(-1):
                break

            if is_training:
                mod_sig_hat, mod_sig = self.extract_mod_sig(wet, mod_sig)
                mod_sig_hat_sr = linear_interpolate_last_dim(mod_sig_hat, dry.size(-1), align_corners=True)
                mod_sig_hat_sr = mod_sig_hat_sr.unsqueeze(1)

            step_latent_sr = mod_sig_hat_sr[:, :, start_idx:end_idx]
            step_dry = dry[:, :, start_idx:end_idx]
            step_wet = wet[:, :, start_idx:end_idx]

            if self.param_model is not None:
                if is_training:
                    param_latent = self.param_model(wet).unsqueeze(-1)
                step_param_latent_sr = param_latent.repeat(1, 1, self.step_n_samples)
                step_latent_sr = tr.cat([step_latent_sr, step_param_latent_sr], dim=1)

            step_wet_hat = self.effect_model(step_dry, step_latent_sr)
            wet_hat_chunks.append(step_wet_hat)
            if is_training:
                step_loss = self.calc_and_log_losses(step_wet_hat, step_wet, prefix, should_log=False)
                self.manual_backward(step_loss)
                opt.step()
                self.effect_model.detach_hidden()
                opt.zero_grad()

        wet_hat = tr.cat(wet_hat_chunks, dim=-1)
        wet_hat_n_samples = wet_hat.size(-1)

        # Remove warmup section to avoid click
        dry = dry[:, :, self.warmup_n_samples:wet_hat_n_samples]
        wet = wet[:, :, self.warmup_n_samples:wet_hat_n_samples]
        wet_hat = wet_hat[:, :, self.warmup_n_samples:wet_hat_n_samples]
        assert dry.shape == wet.shape == wet_hat.shape

        batch_loss = self.calc_and_log_losses(wet_hat, wet, prefix, should_log=True)
        data_dict = {
            "dry": dry.detach().float().cpu(),
            "wet": wet.detach().float().cpu(),
            "wet_hat": wet_hat.detach().float().cpu(),
            "mod_sig_hat": mod_sig_hat.detach().float().cpu(),
        }
        if mod_sig is not None:
            data_dict["mod_sig"] = mod_sig.detach().float().cpu()

        if fx_params is not None:
            fx_params = {k: v.detach().float().cpu() for k, v in fx_params.items()}

        # for idx, (d, w, w_h, m_h) in enumerate(zip(data_dict["dry"],
        #                                            data_dict["wet"],
        #                                            data_dict["wet_hat"],
        #                                            data_dict["mod_sig_hat"])):
        #     if "mod_sig" in data_dict:
        #         m = data_dict["mod_sig"][idx]
        #         plt.plot(m)
        #     plt.plot(m_h)
        #     plt.title(f"mod_sig_{idx}")
        #     plt.show()
        #     plot_spectrogram(d, title=f"dry_{idx}", save_name=f"dry_{idx}", sr=self.sr)
        #     plot_spectrogram(w, title=f"wet_{idx}", save_name=f"wet_{idx}", sr=self.sr)
        #     plot_spectrogram(w_h, title=f"wet_hat_{idx}", save_name=f"wet_hat_{idx}", sr=self.sr)

        return batch_loss, data_dict, fx_params
