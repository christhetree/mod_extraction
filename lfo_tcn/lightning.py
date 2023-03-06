import logging
import os
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class LFOExtraction(pl.LightningModule):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, x: T) -> T:
        return self.model(x)

    def common_step(self, batch: Tuple[T, ...], is_training: bool) -> (T, Dict[str, T]):
        prefix = "train" if is_training else "val"
        dry, wet, mod_sig = batch
        mod_sig_hat = self.model(wet).squeeze(1)
        mod_sig = F.interpolate(mod_sig.unsqueeze(1),
                                mod_sig_hat.size(-1),
                                mode="linear",
                                align_corners=True).squeeze(1)
        assert mod_sig.shape == mod_sig_hat.shape

        mse = self.mse(mod_sig_hat, mod_sig)
        l1 = self.l1(mod_sig_hat, mod_sig)
        loss = l1

        self.log(
            f"{prefix}/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/mse",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/l1",
            l1,
            on_step=False,
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
        return loss, data_dict

    def training_step(self, batch: Tuple[T, ...], batch_idx: int) -> T:
        loss, _ = self.common_step(batch, is_training=True)
        return loss

    def validation_step(self, batch, batch_idx) -> T:
        loss, _ = self.common_step(batch, is_training=False)
        return loss
