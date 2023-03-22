import logging
import os
from typing import Dict, Any, Optional, List

import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch import Tensor as T
from torch.utils.data import DataLoader

from lfo_tcn.datasets import PedalboardPhaserDataset, RandomAudioChunkAndModSigDataset, RandomAudioChunkDataset, \
    RandomAudioChunkDryWetDataset, InterwovenDataset, PreprocessedDataset, RandomPreprocessedDataset
from lfo_tcn.fx import MonoFlangerChorusModule
from lfo_tcn.plotting import plot_spectrogram

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class InterwovenDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 train_dataset_args: List[Dict[str, Any]],
                 val_dataset_args: List[Dict[str, Any]],
                 shared_train_args: Optional[Dict[str, Any]] = None,
                 shared_val_args: Optional[Dict[str, Any]] = None,
                 shared_args: Optional[Dict[str, Any]] = None,
                 num_workers: int = 0) -> None:
        super().__init__()
        self.save_hyperparameters()
        log.info(f"\n{self.hparams}")
        self.batch_size = batch_size
        self.train_dataset_args = train_dataset_args
        self.val_dataset_args = val_dataset_args
        if shared_train_args is None:
            self.shared_train_args = {}
        else:
            self.shared_train_args = shared_train_args
        if shared_val_args is None:
            self.shared_val_args = {}
        else:
            self.shared_val_args = shared_val_args
        self.num_workers = num_workers
        if shared_args is not None:
            for k, v in shared_args.items():
                if k not in self.shared_train_args:
                    self.shared_train_args[k] = v
                else:
                    log.info(f"Found existing key in shared_train_args: {k}")
                if k not in self.shared_val_args:
                    self.shared_val_args[k] = v
                else:
                    log.info(f"Found existing key in shared_val_args: {k}")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = InterwovenDataset(
                self.train_dataset_args,
                self.shared_train_args,
            )
            assert len(self.train_dataset.datasets) <= self.batch_size
        if stage == "validate" or "fit":
            self.val_dataset = InterwovenDataset(
                self.val_dataset_args,
                self.shared_val_args,
            )
            assert len(self.val_dataset.datasets) <= self.batch_size

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


class RandomAudioChunkDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 train_dir: str,
                 val_dir: str,
                 train_num_examples_per_epoch: int,
                 val_num_examples_per_epoch: int,
                 n_samples: int,
                 sr: float,
                 ext: str = "wav",
                 silence_fraction_allowed: float = 0.1,
                 silence_threshold_energy: float = 1e-6,
                 n_retries: int = 10,
                 num_workers: int = 0,
                 use_debug_mode: bool = False,
                 check_dataset: bool = True,
                 end_buffer_n_samples: int = 0) -> None:
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
        self.sr = sr
        self.ext = ext
        self.silence_fraction_allowed = silence_fraction_allowed
        self.silence_threshold_energy = silence_threshold_energy
        self.n_retries = n_retries
        self.num_workers = num_workers
        self.use_debug_mode = use_debug_mode
        self.check_dataset = check_dataset
        self.end_buffer_n_samples = end_buffer_n_samples
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = RandomAudioChunkDataset(
                self.train_dir,
                n_samples=self.n_samples,
                sr=self.sr,
                ext=self.ext,
                num_examples_per_epoch=self.train_num_examples_per_epoch,
                silence_fraction_allowed=self.silence_fraction_allowed,
                silence_threshold_energy=self.silence_threshold_energy,
                n_retries=self.n_retries,
                use_debug_mode=self.use_debug_mode,
                check_dataset=self.check_dataset,
                end_buffer_n_samples=self.end_buffer_n_samples,
            )
        if stage == "validate" or "fit":
            self.val_dataset = RandomAudioChunkDataset(
                self.val_dir,
                n_samples=self.n_samples,
                sr=self.sr,
                ext=self.ext,
                num_examples_per_epoch=self.val_num_examples_per_epoch,
                silence_fraction_allowed=self.silence_fraction_allowed,
                silence_threshold_energy=self.silence_threshold_energy,
                n_retries=self.n_retries,
                use_debug_mode=self.use_debug_mode,
                check_dataset=self.check_dataset,
                end_buffer_n_samples=self.end_buffer_n_samples,
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


class RandomAudioChunkDryWetDataModule(RandomAudioChunkDataModule):
    def __init__(self,
                 batch_size: int,
                 dry_train_dir: str,
                 dry_val_dir: str,
                 wet_train_dir: str,
                 wet_val_dir: str,
                 train_num_examples_per_epoch: int,
                 val_num_examples_per_epoch: int,
                 n_samples: int,
                 sr: float,
                 ext: str = "wav",
                 silence_fraction_allowed: float = 0.1,
                 silence_threshold_energy: float = 1e-6,
                 n_retries: int = 10,
                 num_workers: int = 0,
                 use_debug_mode: bool = False,
                 check_dataset: bool = True,
                 end_buffer_n_samples: int = 0) -> None:
        super().__init__(batch_size,
                         dry_train_dir,
                         dry_val_dir,
                         train_num_examples_per_epoch,
                         val_num_examples_per_epoch,
                         n_samples,
                         sr,
                         ext,
                         silence_fraction_allowed,
                         silence_threshold_energy,
                         n_retries,
                         num_workers,
                         use_debug_mode,
                         check_dataset,
                         end_buffer_n_samples)
        self.save_hyperparameters()
        self.dry_train_dir = dry_train_dir
        self.dry_val_dir = dry_val_dir
        self.wet_train_dir = wet_train_dir
        self.wet_val_dir = wet_val_dir

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = RandomAudioChunkDryWetDataset(
                self.dry_train_dir,
                self.wet_train_dir,
                n_samples=self.n_samples,
                sr=self.sr,
                ext=self.ext,
                num_examples_per_epoch=self.train_num_examples_per_epoch,
                silence_fraction_allowed=self.silence_fraction_allowed,
                silence_threshold_energy=self.silence_threshold_energy,
                n_retries=self.n_retries,
                use_debug_mode=self.use_debug_mode,
                check_dataset=self.check_dataset,
                end_buffer_n_samples=self.end_buffer_n_samples,
            )
        if stage == "validate" or "fit":
            self.val_dataset = RandomAudioChunkDryWetDataset(
                self.dry_val_dir,
                self.wet_val_dir,
                n_samples=self.n_samples,
                sr=self.sr,
                ext=self.ext,
                num_examples_per_epoch=self.val_num_examples_per_epoch,
                silence_fraction_allowed=self.silence_fraction_allowed,
                silence_threshold_energy=self.silence_threshold_energy,
                n_retries=self.n_retries,
                use_debug_mode=self.use_debug_mode,
                check_dataset=self.check_dataset,
                end_buffer_n_samples=self.end_buffer_n_samples,
            )

    def on_before_batch_transfer(self,
                                 batch: (T, T),
                                 dataloader_idx: int) -> (T, T, Optional[T], Optional[Dict[str, T]]):
        dry, wet = batch
        return dry, wet, None, None


class PedalboardPhaserDataModule(RandomAudioChunkDataModule):
    def __init__(self,
                 fx_config: Dict[str, Any],
                 batch_size: int,
                 train_dir: str,
                 val_dir: str,
                 train_num_examples_per_epoch: int,
                 val_num_examples_per_epoch: int,
                 n_samples: int,
                 sr: float,
                 ext: str = "wav",
                 silence_fraction_allowed: float = 0.1,
                 silence_threshold_energy: float = 1e-6,
                 n_retries: int = 10,
                 num_workers: int = 0,
                 use_debug_mode: bool = False,
                 check_dataset: bool = True,
                 end_buffer_n_samples: int = 0) -> None:
        super().__init__(batch_size,
                         train_dir,
                         val_dir,
                         train_num_examples_per_epoch,
                         val_num_examples_per_epoch,
                         n_samples,
                         sr,
                         ext,
                         silence_fraction_allowed,
                         silence_threshold_energy,
                         n_retries,
                         num_workers,
                         use_debug_mode,
                         check_dataset,
                         end_buffer_n_samples)
        self.save_hyperparameters()
        self.fx_config = fx_config

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = PedalboardPhaserDataset(
                self.fx_config,
                self.train_dir,
                n_samples=self.n_samples,
                sr=self.sr,
                ext=self.ext,
                num_examples_per_epoch=self.train_num_examples_per_epoch,
                silence_fraction_allowed=self.silence_fraction_allowed,
                silence_threshold_energy=self.silence_threshold_energy,
                n_retries=self.n_retries,
                use_debug_mode=self.use_debug_mode,
                check_dataset=self.check_dataset,
                end_buffer_n_samples=self.end_buffer_n_samples,
            )
        if stage == "validate" or "fit":
            self.val_dataset = PedalboardPhaserDataset(
                self.fx_config,
                self.val_dir,
                n_samples=self.n_samples,
                sr=self.sr,
                ext=self.ext,
                num_examples_per_epoch=self.val_num_examples_per_epoch,
                silence_fraction_allowed=self.silence_fraction_allowed,
                silence_threshold_energy=self.silence_threshold_energy,
                n_retries=self.n_retries,
                use_debug_mode=self.use_debug_mode,
                check_dataset=self.check_dataset,
                end_buffer_n_samples=self.end_buffer_n_samples,
            )


class FlangerCPUDataModule(PedalboardPhaserDataModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flanger = None

    def setup(self, stage: str) -> None:
        self.flanger = MonoFlangerChorusModule(batch_size=self.batch_size,
                                               n_ch=1,
                                               n_samples=self.n_samples,
                                               sr=self.sr,
                                               min_delay_ms=self.fx_config["flanger"]["min_delay_ms"],
                                               max_lfo_delay_ms=self.fx_config["flanger"]["max_lfo_delay_ms"])
        if stage == "fit":
            self.train_dataset = RandomAudioChunkAndModSigDataset(
                self.fx_config,
                self.train_dir,
                n_samples=self.n_samples,
                sr=self.sr,
                ext=self.ext,
                num_examples_per_epoch=self.train_num_examples_per_epoch,
                silence_fraction_allowed=self.silence_fraction_allowed,
                silence_threshold_energy=self.silence_threshold_energy,
                n_retries=self.n_retries,
                use_debug_mode=self.use_debug_mode,
                check_dataset=self.check_dataset,
                end_buffer_n_samples=self.end_buffer_n_samples,
            )
        if stage == "validate" or "fit":
            self.val_dataset = RandomAudioChunkAndModSigDataset(
                self.fx_config,
                self.val_dir,
                n_samples=self.n_samples,
                sr=self.sr,
                ext=self.ext,
                num_examples_per_epoch=self.val_num_examples_per_epoch,
                silence_fraction_allowed=self.silence_fraction_allowed,
                silence_threshold_energy=self.silence_threshold_energy,
                n_retries=self.n_retries,
                use_debug_mode=self.use_debug_mode,
                check_dataset=self.check_dataset,
                end_buffer_n_samples=self.end_buffer_n_samples,
            )

    def on_before_batch_transfer(self, batch: (T, T), dataloader_idx: int) -> (T, T, T, Dict[str, T]):
        dry, mod_sig, fx_params = batch
        feedback = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["feedback"]["min"],
            self.fx_config["flanger"]["feedback"]["max"],
            n=self.batch_size,
        )
        width = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["width"]["min"],
            self.fx_config["flanger"]["width"]["max"],
            n=self.batch_size,
        )
        depth = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["depth"]["min"],
            self.fx_config["flanger"]["depth"]["max"],
            n=self.batch_size,
        )
        mix = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["mix"]["min"],
            self.fx_config["flanger"]["mix"]["max"],
            n=self.batch_size,
        )
        fx_params["depth"] = depth
        fx_params["feedback"] = feedback
        fx_params["max_lfo_delay_ms"] = self.flanger.max_lfo_delay_ms
        fx_params["min_delay_ms"] = self.flanger.min_delay_ms
        fx_params["mix"] = mix
        fx_params["width"] = width

        wet = self.flanger(dry, mod_sig, feedback, width, depth, mix)

        if self.use_debug_mode:
            for idx, (d, w, m) in enumerate(zip(dry, wet, mod_sig)):
                plt.plot(m.squeeze(0))
                plt.title(f"flanger_mod_sig_{idx}")
                plt.show()
                plot_spectrogram(d, title=f"flanger_dry_{idx}", save_name=f"flanger_dry_{idx}", sr=self.sr)
                plot_spectrogram(w, title=f"flanger_wet_{idx}", save_name=f"flanger_wet_{idx}", sr=self.sr)

        return dry, wet, mod_sig, fx_params


class PreprocessedDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 train_dir: str,
                 val_dir: str,
                 n_samples: int,
                 sr: float,
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
        self.n_samples = n_samples
        self.sr = sr
        self.num_workers = num_workers
        self.use_debug_mode = use_debug_mode

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = PreprocessedDataset(self.train_dir,
                                                     self.n_samples,
                                                     self.sr,
                                                     use_debug_mode=self.use_debug_mode)
        if stage == "validate" or "fit":
            self.val_dataset = PreprocessedDataset(self.val_dir,
                                                   self.n_samples,
                                                   self.sr,
                                                   use_debug_mode=self.use_debug_mode)

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


class RandomPreprocessedDataModule(PreprocessedDataModule):
    def __init__(self,
                 train_num_examples_per_epoch: int,
                 val_num_examples_per_epoch: int,
                 batch_size: int,
                 train_dir: str,
                 val_dir: str,
                 n_samples: int,
                 sr: float,
                 num_workers: int = 0,
                 use_debug_mode: bool = False) -> None:
        super().__init__(batch_size, train_dir, val_dir, n_samples, sr, num_workers, use_debug_mode)
        self.train_num_examples_per_epoch = train_num_examples_per_epoch
        self.val_num_examples_per_epoch = val_num_examples_per_epoch

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = RandomPreprocessedDataset(self.train_num_examples_per_epoch,
                                                           self.train_dir,
                                                           self.n_samples,
                                                           self.sr,
                                                           use_debug_mode=self.use_debug_mode)
        if stage == "validate" or "fit":
            self.val_dataset = RandomPreprocessedDataset(self.val_num_examples_per_epoch,
                                                         self.val_dir,
                                                         self.n_samples,
                                                         self.sr,
                                                         use_debug_mode=self.use_debug_mode)
