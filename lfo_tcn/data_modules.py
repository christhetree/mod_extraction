import logging
import os
from typing import Dict, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lfo_tcn.datasets import PedalboardPhaserGeneratorDataset

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
            self.train_dataset = PedalboardPhaserGeneratorDataset(
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
            self.val_dataset = PedalboardPhaserGeneratorDataset(
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


if __name__ == "__main__":
    dm = PedalboardPhaserDataModule(
        32,
        "/Users/puntland/local_christhetree/aim/erase-fx/data/guitarset/train",
        "/Users/puntland/local_christhetree/aim/erase-fx/data/guitarset/val",
        20,
        10,
        88100,
        {},
    )
    print(dm.hparams)
