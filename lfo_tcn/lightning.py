import logging
import os
from typing import Dict, Optional

import pytorch_lightning as pl
import torch as tr
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from lfo_tcn.util import get_loss_func_by_name

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class LFOExtraction(pl.LightningModule):
    default_loss_dict = {"l1": 1.0, "mse": 0.0}

    def __init__(self,
                 model: nn.Module,
                 sr: float,
                 loss_dict: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.sr = sr
        if loss_dict is None:
            loss_dict = self.default_loss_dict
        self.loss_dict = loss_dict
        self.loss_funcs = [get_loss_func_by_name(name) for name, _ in loss_dict.items()]

    def forward(self, x: T) -> T:
        return self.model(x)

    def common_step(self, batch: (T, T, T, Dict[str, T]), is_training: bool) -> (T, Dict[str, T], Dict[str, T]):
        prefix = "train" if is_training else "val"
        dry, wet, mod_sig, fx_params = batch
        mod_sig_hat = self.model(wet).squeeze(1)
        mod_sig = F.interpolate(mod_sig.unsqueeze(1),
                                mod_sig_hat.size(-1),
                                mode="linear",
                                align_corners=True).squeeze(1)
        assert mod_sig.shape == mod_sig_hat.shape

        # TODO(cm): refactor into separate method
        loss_values = [loss_func(mod_sig_hat, mod_sig) for loss_func in self.loss_funcs]
        loss = tr.tensor(0.0)
        for (name, weighting), loss_value in zip(self.loss_dict.items(), loss_values):
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
                loss += weighting * loss_value
        self.log(
            f"{prefix}/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        data_dict = {
            "dry": dry.detach().float().cpu(),
            "wet": wet.detach().float().cpu(),
            "mod_sig": mod_sig.detach().float().cpu(),
            "mod_sig_hat": mod_sig_hat.detach().float().cpu(),
        }
        fx_params = {k: v.detach().float().cpu() for k, v in fx_params.items()}
        return loss, data_dict, fx_params

    def training_step(self, batch: (T, T, T, Dict[str, T]), batch_idx: int) -> T:
        loss, _, _ = self.common_step(batch, is_training=True)
        return loss

    def validation_step(self, batch: (T, T, T, Dict[str, T]), batch_idx: int) -> (T, Dict[str, T], Dict[str, T]):
        return self.common_step(batch, is_training=False)


class LFOEffectModeling(pl.LightningModule):
    default_loss_dict = {"esr": 1.0, "dc": 1.0}

    def __init__(self,
                 effect_model: nn.Module,
                 lfo_model: Optional[nn.Module] = None,
                 lfo_model_weights_path: Optional[str] = None,
                 freeze_lfo_model: bool = False,
                 loss_dict: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        self.use_gt_mod_sig = lfo_model is None
        self.save_hyperparameters(ignore=["effect_model", "lfo_model"])
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
        self.lfo_model = lfo_model
        if loss_dict is None:
            loss_dict = self.default_loss_dict
        self.loss_dict = loss_dict
        self.loss_funcs = [get_loss_func_by_name(name) for name, _ in loss_dict.items()]

    def forward(self, dry: T, mod_sig: T = None) -> T:
        return self.model(dry, mod_sig)

    def common_step(self,
                    batch: (T, T, Optional[T], Optional[Dict[str, T]]),
                    is_training: bool) -> (T, Dict[str, T], Optional[Dict[str, T]]):
        prefix = "train" if is_training else "val"
        dry, wet, mod_sig, fx_params = batch
        if mod_sig is None:
            assert self.lfo_model is not None

        if self.lfo_model is None:
            mod_sig_hat = mod_sig.unsqueeze(1)
        else:
            mod_sig_hat = self.lfo_model(wet)

        mod_sig_hat_sr = F.interpolate(mod_sig_hat,
                                       dry.size(-1),
                                       mode="linear",
                                       align_corners=True)
        wet_hat = self.forward(dry, mod_sig_hat_sr)
        assert dry.shape == wet.shape == wet_hat.shape

        if mod_sig is not None and mod_sig_hat.size(-1) != mod_sig_hat.size(-1):
            mod_sig = F.interpolate(mod_sig.unsqueeze(1),
                                    mod_sig_hat.size(-1),
                                    mode="linear",
                                    align_corners=True).squeeze(1)
        mod_sig_hat = mod_sig_hat.squeeze(1)
        assert mod_sig.shape == mod_sig_hat.shape

        # TODO(cm): refactor into separate method
        loss_values = [loss_func(wet_hat, wet) for loss_func in self.loss_funcs]
        loss = tr.tensor(0.0)
        for (name, weighting), loss_value in zip(self.loss_dict.items(), loss_values):
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
                loss += weighting * loss_value
        self.log(
            f"{prefix}/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        data_dict = {
            "dry": dry.detach().float().cpu(),
            "wet": wet.detach().float().cpu(),
            "wet_hat": wet.detach().float().cpu(),
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
