import logging
import os
from contextlib import nullcontext
from typing import Dict, Optional

import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T
from torch import nn
from torch.optim import Optimizer

from lfo_tcn.modulations import stretch_corners, find_valid_mod_sig_indices, make_rand_mod_signal
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
                 sr: float = 44100,
                 use_dry: bool = False,
                 model_smooth_n_frames: int = 4,
                 should_stretch: bool = False,
                 max_n_corners: int = 20,
                 stretch_smooth_n_frames: int = 16,
                 sub_batch_size: Optional[int] = None,
                 loss_dict: Optional[Dict[str, float]] = None) -> None:
        super().__init__(loss_dict)
        self.save_hyperparameters(ignore=["model"])
        log.info(f"\n{self.hparams}")
        self.model = model
        self.sr = sr
        self.use_dry = use_dry
        self.model_smooth_n_frames = model_smooth_n_frames
        self.should_stretch = should_stretch
        self.max_n_corners = max_n_corners
        self.stretch_smooth_n_frames = stretch_smooth_n_frames
        self.sub_batch_size = sub_batch_size

    def forward(self, x: T) -> T:
        return self.model(x)

    def center_crop_mod_sig(self, mod_sig: T, size: int) -> T:
        if size == mod_sig.size(-1):
            return mod_sig
        assert size < mod_sig.size(-1)
        padding = mod_sig.size(-1) - size
        pad_l = padding // 2
        pad_r = padding - pad_l
        mod_sig = mod_sig[:, pad_l:-pad_r]
        return mod_sig

    def common_step(self,
                    batch: (Optional[T], T, Optional[T], Optional[Dict[str, T]]),
                    is_training: bool) -> (T, Dict[str, T], Optional[Dict[str, T]]):
        prefix = "train" if is_training else "val"
        dry, wet, mod_sig, fx_params = batch

        if self.use_dry:
            assert dry is not None
            mod_sig_hat, latent = self.model(tr.cat([dry, wet], dim=1))
        else:
            mod_sig_hat, latent = self.model(wet)
        mod_sig_hat = mod_sig_hat.squeeze(1)

        # dry = wet
        # phase_all = fx_params["phase"]
        # freq_all = fx_params["rate_hz"]
        # shapes_all = fx_params["shape"]
        # mod_sig_hat = make_rand_mod_signal(
        #     batch_size=wet.size(0),
        #     n_samples=345,
        #     sr=172.5,
        #     freq_min=0.5,
        #     freq_max=3.0,
        #     shapes_all=shapes_all,
        #     freq_all=freq_all,
        #     phase_all=phase_all,
        #     freq_error=0.25,
        #     phase_error=0.5,
        # )

        if mod_sig is None:
            mod_sig = tr.zeros_like(mod_sig_hat)
        else:
            mod_sig = linear_interpolate_last_dim(mod_sig, mod_sig_hat.size(-1), align_corners=True)

        assert mod_sig.shape == mod_sig_hat.shape
        if self.model_smooth_n_frames > 1:
            mod_sig_hat = mod_sig_hat.unfold(dimension=-1, size=self.model_smooth_n_frames, step=1)
            mod_sig_hat = tr.mean(mod_sig_hat, dim=-1, keepdim=False)
            mod_sig = self.center_crop_mod_sig(mod_sig, mod_sig_hat.size(-1))

        if self.should_stretch:
            mod_sig_hat = stretch_corners(mod_sig_hat,
                                          max_n_corners=self.max_n_corners,
                                          smooth_n_frames=self.stretch_smooth_n_frames)
            if self.stretch_smooth_n_frames > 1:
                mod_sig = self.center_crop_mod_sig(mod_sig, mod_sig_hat.size(-1))
        assert mod_sig.shape == mod_sig_hat.shape

        # valid_indices = find_valid_mod_sig_indices(mod_sig_hat)
        # if not valid_indices:
        #     log.info("No valid LFO signals found")
        #     return None, None, None
        # log.info(f"Found {len(valid_indices)} valid LFO signals")
        # dry = dry[valid_indices, ...]
        # wet = wet[valid_indices, ...]
        # mod_sig_hat = mod_sig_hat[valid_indices, ...]
        # if mod_sig is not None:
        #     mod_sig = mod_sig[valid_indices, ...]

        loss = self.calc_and_log_losses(mod_sig_hat, mod_sig, prefix)

        data_dict = {
            "wet": wet.detach().float().cpu(),
            "mod_sig": mod_sig.detach().float().cpu(),
            "mod_sig_hat": mod_sig_hat.detach().float().cpu(),
            # "latent": latent.detach().float().cpu(),
        }
        if dry is not None:
            data_dict["dry"] = dry.detach().float().cpu()

        if fx_params is not None:
            fx_params = {k: v.detach().float().cpu() if isinstance(v, T) else v for k, v in fx_params.items()}

        # TODO(cm)
        if data_dict["mod_sig_hat"].size(0) < 10:
            from lfo_tcn.plotting import plot_spectrogram
            for idx, (d, w, m_h) in enumerate(zip(data_dict["dry"],
                                                  data_dict["wet"],
                                                  data_dict["mod_sig_hat"])):
                from matplotlib import pyplot as plt
                if "mod_sig" in data_dict:
                    m = data_dict["mod_sig"][idx]
                    plt.plot(m)
                plt.plot(m_h)
                plt.title(f"mod_sig_{idx}")
                plt.show()
                # plot_spectrogram(d, title=f"dry_{idx}", save_name=f"dry_{idx}", sr=self.sr)
                plot_spectrogram(w, title=f"wet_{idx}", save_name=f"wet_{idx}", sr=self.sr)
            exit()

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


# TODO(cm): refactor
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
                mod_sig_hat, latent = self.lfo_model(wet)
                mod_sig_hat = mod_sig_hat.squeeze(1)

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
        result = self.common_step(batch, is_training=True)
        if result is None:
            return None
        loss = result[0]
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
                 use_dry: bool = False,
                 model_smooth_n_frames: int = 8,
                 should_stretch: bool = False,
                 max_n_corners: int = 16,
                 stretch_smooth_n_frames: int = 8,
                 discard_invalid_lfos: bool = False,
                 loss_dict: Optional[Dict[str, float]] = None) -> None:
        assert warmup_n_samples > 0
        super().__init__(effect_model,
                         lfo_model,
                         lfo_model_weights_path,
                         freeze_lfo_model=freeze_lfo_model,
                         sr=sr,
                         loss_dict=loss_dict)
        self.warmup_n_samples = warmup_n_samples
        self.step_n_samples = step_n_samples
        self.param_model = param_model
        self.use_dry = use_dry
        self.model_smooth_n_frames = model_smooth_n_frames
        self.should_stretch = should_stretch
        self.max_n_corners = max_n_corners
        self.stretch_smooth_n_frames = stretch_smooth_n_frames
        self.discard_invalid_lfos = discard_invalid_lfos
        self.automatic_optimization = False

    # TODO(cm): refactor
    def center_crop_mod_sig(self, mod_sig: T, size: int) -> T:
        if size == mod_sig.size(-1):
            return mod_sig
        assert size < mod_sig.size(-1)
        padding = mod_sig.size(-1) - size
        pad_l = padding // 2
        pad_r = padding - pad_l
        mod_sig = mod_sig[..., pad_l:-pad_r]
        return mod_sig

    def smooth_stretch_crop_mod_sig(self, mod_sig_hat: T, mod_sig: Optional[T] = None) -> (T, Optional[T], int):
        orig_n_frames = mod_sig_hat.size(-1)
        if self.model_smooth_n_frames > 1:
            mod_sig_hat = mod_sig_hat.unfold(dimension=-1, size=self.model_smooth_n_frames, step=1)
            mod_sig_hat = tr.mean(mod_sig_hat, dim=-1, keepdim=False)
            if mod_sig is not None:
                mod_sig = self.center_crop_mod_sig(mod_sig, mod_sig_hat.size(-1))

        if self.should_stretch:
            mod_sig_hat = stretch_corners(mod_sig_hat,
                                          max_n_corners=self.max_n_corners,
                                          smooth_n_frames=self.stretch_smooth_n_frames)
            if self.stretch_smooth_n_frames > 1 and mod_sig is not None:
                mod_sig = self.center_crop_mod_sig(mod_sig, mod_sig_hat.size(-1))
        new_n_frames = mod_sig_hat.size(-1)
        removed_n_frames = orig_n_frames - new_n_frames
        return mod_sig_hat, mod_sig, removed_n_frames

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

        lfo_model_input = wet
        if self.use_dry:
            lfo_model_input = tr.cat([dry, wet], dim=1)

        mod_sig_hat, mod_sig = self.extract_mod_sig(lfo_model_input, mod_sig)
        mod_sig_hat, mod_sig, removed_n_frames = self.smooth_stretch_crop_mod_sig(mod_sig_hat, mod_sig)
        n_frames = mod_sig_hat.size(-1)
        n_samples = int((n_frames / (n_frames + removed_n_frames)) * dry.size(-1))
        dry = self.center_crop_mod_sig(dry, n_samples)
        wet = self.center_crop_mod_sig(wet, n_samples)

        # freq_all = tr.ones((wet.size(0),)) * 0.75
        # freq_all = tr.ones((wet.size(0),)) * 2.0
        # mod_sig_hat = make_rand_mod_signal(
        #     batch_size=wet.size(0),
        #     n_samples=345,
        #     sr=172.5,
        #     freq_min=0.5,
        #     freq_max=2.0,
        #     shapes=["tri"],
        #     freq_all=freq_all,
        #     freq_error=0.0,
        #     # freq_error=0.25,
        # )
        # mod_sig_hat = mod_sig_hat.to(self.device)

        # mod_sig_hat = tr.zeros((wet.size(0), 345)).to(self.device)

        if self.discard_invalid_lfos:
            valid_indices = find_valid_mod_sig_indices(mod_sig_hat)
            if not valid_indices:
                log.info("No valid LFO signals found")
                return None
            # log.info(f"Found {len(valid_indices)} valid LFO signals")
            dry = dry[valid_indices, ...]
            wet = wet[valid_indices, ...]
            mod_sig_hat = mod_sig_hat[valid_indices, ...]
            if mod_sig is not None:
                mod_sig = mod_sig[valid_indices, ...]

        mod_sig_hat_sr = linear_interpolate_last_dim(mod_sig_hat, dry.size(-1), align_corners=True)
        mod_sig_hat_sr = mod_sig_hat_sr.unsqueeze(1)

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

            if is_training and not self.freeze_lfo_model:
                mod_sig_hat, mod_sig = self.extract_mod_sig(lfo_model_input, mod_sig)
                mod_sig_hat, mod_sig, _ = self.smooth_stretch_crop_mod_sig(mod_sig_hat, mod_sig)
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
            fx_params = {k: v.detach().float().cpu() if isinstance(v, T) else v for k, v in fx_params.items()}

        if wet.size(0) < 10:
            from matplotlib import pyplot as plt
            from lfo_tcn.plotting import plot_spectrogram
            for idx, (d, w, w_h, m_h) in enumerate(zip(data_dict["dry"],
                                                       data_dict["wet"],
                                                       data_dict["wet_hat"],
                                                       data_dict["mod_sig_hat"])):
                if "mod_sig" in data_dict:
                    m = data_dict["mod_sig"][idx]
                    plt.plot(m)
                plt.plot(m_h)
                plt.title(f"mod_sig_{idx}")
                plt.show()
                plot_spectrogram(d, title=f"dry_{idx}", save_name=f"dry_{idx}", sr=self.sr)
                plot_spectrogram(w, title=f"wet_{idx}", save_name=f"wet_{idx}", sr=self.sr)
                plot_spectrogram(w_h, title=f"wet_hat_{idx}", save_name=f"wet_hat_{idx}", sr=self.sr)
            exit()

        return batch_loss, data_dict, fx_params
