import logging
import os
from typing import Dict, Any, List

import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch import Tensor as T
from torch.utils.data import DataLoader

from lfo_tcn.datasets import PedalboardPhaserDataset, RandomAudioChunkAndModSigDataset, RandomAudioChunkDataset
from lfo_tcn.fx import FlangerModule
from lfo_tcn.util import display_spectrogram

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class PedalboardPhaserDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 train_dir: str,
                 val_dir: str,
                 train_num_examples_per_epoch: int,
                 val_num_examples_per_epoch: int,
                 n_samples: int,
                 fx_config: Dict[str, Any],
                 sr: float = 44100,
                 ext: str = "wav",
                 silence_threshold_energy: float = 1e-4,
                 n_retries: int = 20,
                 num_workers: int = 0,
                 use_debug_mode: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters()
        log.info(f"\n{self.hparams}")
        self.batch_size = batch_size
        assert os.path.isdir(train_dir)
        self.train_dir = train_dir
        assert os.path.isdir(val_dir)
        self.val_dir = val_dir
        self.train_num_examples_per_epoch = train_num_examples_per_epoch
        self.val_num_examples_per_epoch = val_num_examples_per_epoch
        self.n_samples = n_samples
        self.fx_config = fx_config
        self.sr = sr
        self.ext = ext
        self.silence_threshold_energy = silence_threshold_energy
        self.n_retries = n_retries
        self.num_workers = num_workers
        self.use_debug_mode = use_debug_mode
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = PedalboardPhaserDataset(
                self.train_dir,
                self.n_samples,
                self.fx_config,
                self.sr,
                self.ext,
                self.train_num_examples_per_epoch,
                self.silence_threshold_energy,
                self.n_retries,
                self.use_debug_mode,
            )
        if stage == "validate" or "fit":
            self.val_dataset = PedalboardPhaserDataset(
                self.val_dir,
                self.n_samples,
                self.fx_config,
                self.sr,
                self.ext,
                self.val_num_examples_per_epoch,
                self.silence_threshold_energy,
                self.n_retries,
                self.use_debug_mode,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )


class FlangerCPUDataModule(PedalboardPhaserDataModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flanger = None

    def setup(self, stage: str) -> None:
        self.flanger = FlangerModule(batch_size=self.batch_size,
                                     n_ch=1,
                                     n_samples=self.n_samples,
                                     max_delay_ms=self.fx_config["flanger"]["max_delay_ms"],
                                     sr=self.sr)
        if stage == "fit":
            self.train_dataset = RandomAudioChunkAndModSigDataset(
                self.train_dir,
                self.n_samples,
                self.fx_config,
                self.sr,
                self.ext,
                self.train_num_examples_per_epoch,
                self.silence_threshold_energy,
                self.n_retries,
                self.use_debug_mode,
            )
        if stage == "validate" or "fit":
            self.val_dataset = RandomAudioChunkAndModSigDataset(
                self.val_dir,
                self.n_samples,
                self.fx_config,
                self.sr,
                self.ext,
                self.val_num_examples_per_epoch,
                self.silence_threshold_energy,
                self.n_retries,
                self.use_debug_mode,
            )

    def on_before_batch_transfer(self, batch: List[T], dataloader_idx: int) -> List[T]:
        dry, mod_sig = batch
        feedback = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["feedback"]["min"],
            self.fx_config["flanger"]["feedback"]["max"]
        )
        width = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["width"]["min"],
            self.fx_config["flanger"]["width"]["max"]
        )
        depth = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["depth"]["min"],
            self.fx_config["flanger"]["depth"]["max"]
        )
        mix = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["mix"]["min"],
            self.fx_config["flanger"]["mix"]["max"]
        )
        wet = self.flanger(dry, mod_sig, feedback, width, depth, mix)

        if self.use_debug_mode:
            for idx, (d, w, m) in enumerate(zip(dry, wet, mod_sig)):
                plt.plot(m.squeeze(0))
                plt.title("mod_sig")
                plt.show()
                display_spectrogram(d, save_audio=True, name="flanger_dry", idx=idx)
                display_spectrogram(w, save_audio=True, name="flanger_wet", idx=idx)

        return [dry, wet, mod_sig]