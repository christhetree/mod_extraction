import logging
import os
from typing import Any, Dict

from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, Callback, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor as T

from lfo_tcn.plotting import plot_spectrogram, plot_mod_sig, fig2img

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ConsoleLRMonitor(LearningRateMonitor):
    # TODO(cm): enable every n steps
    def on_train_epoch_start(self,
                             trainer: Trainer,
                             *args: Any,
                             **kwargs: Any) -> None:
        super().on_train_epoch_start(trainer, *args, **kwargs)
        if self.logging_interval != "step":
            interval = "epoch" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)
            latest_stat_str = {k: f"{v:.8f}" for k, v in latest_stat.items()}
            if latest_stat:
                log.info(f"Current LR: {latest_stat_str}")


class LogModSigCallback(Callback):
    def __init__(self, n_examples: int = 5) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.images = []

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: (T, Dict[str, T], Dict[str, T]),
                                batch: (T, T, T, Dict[str, T]),
                                batch_idx: int,
                                dataloader_idx: int) -> None:
        if outputs is None:
            return
        _, data_dict, fx_params = outputs
        wet = data_dict["wet"]
        mod_sig_hat = data_dict["mod_sig_hat"]
        mod_sig = data_dict["mod_sig"]
        if batch_idx == 0:
            self.images = []
            for idx in range(self.n_examples):
                if idx < mod_sig_hat.size(0):
                    fig, ax = plt.subplots(nrows=2, figsize=(6, 10), sharex="all", squeeze=True)
                    params = {k: v[idx] for k, v in fx_params.items()}
                    title = ", ".join([f"{k}: {v:.2f}" for k, v in params.items()
                                       if k not in {"mix", "rate_hz"}])
                    w = wet[idx]
                    plot_spectrogram(w, ax[0], title, sr=pl_module.sr)
                    m_hat = mod_sig_hat[idx]
                    m = mod_sig[idx]
                    plot_mod_sig(ax[1], m_hat, m)
                    fig.tight_layout()
                    img = fig2img(fig)
                    self.images.append(img)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.images:
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image(key="mod_sig_plots",
                                     images=self.images,
                                     step=trainer.global_step)
