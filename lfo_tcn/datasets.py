import logging
import os
from collections import defaultdict
from typing import Dict, Optional, List, Any, Tuple, Type, Union

import torch
import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from pedalboard import Pedalboard, Phaser
from scipy.stats import loguniform
from torch import Tensor as T
from torch.utils.data import Dataset
from tqdm import tqdm

from lfo_tcn import fx
from lfo_tcn.fx import make_mod_signal
from lfo_tcn.plotting import plot_spectrogram
from lfo_tcn.util import linear_interpolate_last_dim

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def get_dataset_class(name: str) -> Type[Dataset]:
    if name == "random_audio_chunk":
        return RandomAudioChunkDataset
    elif name == "random_audio_chunk_dry_wet":
        return RandomAudioChunkDryWetDataset
    elif name == "random_audio_chunk_and_mod_sig":
        return RandomAudioChunkAndModSigDataset
    elif name == "pedalboard_phaser":
        return PedalboardPhaserDataset
    elif name == "tremolo":
        return TremoloDataset
    elif name == "preproc":
        return PreprocessedDataset
    elif name == "random_preproc":
        return RandomPreprocessedDataset
    else:
        raise ValueError(f"Unknown dataset name: {name}")


class InterwovenDataset(Dataset):
    def __init__(
            self,
            dataset_args: List[Dict[str, Any]],
            common_args: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.dataset_args = dataset_args
        self.common_args = common_args

        dataset_names = []
        dataset_weightings = []
        datasets = []
        for ds_args in dataset_args:
            assert "dataset_name" in ds_args
            ds_name = ds_args.pop("dataset_name")
            dataset_names.append(ds_name)
            n_copies = ds_args.pop("n_copies", 1)
            dataset_weightings.append(n_copies)
            for k, v in common_args.items():
                if k not in ds_args:
                    ds_args[k] = v
            for _ in range(n_copies):
                ds_class = get_dataset_class(ds_name)
                ds = ds_class(**ds_args)  # TODO(cm): check random seed
                datasets.append(ds)
        self.dataset_names = dataset_names
        self.dataset_weightings = dataset_weightings
        self.datasets = datasets
        log.info(f"dataset_names = {dataset_names}")
        log.info(f"dataset_weightings = {dataset_weightings}")

        self.size = len(datasets[0])
        assert all(len(d) == self.size for d in datasets)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Any:
        ds_idx = idx % len(self.datasets)
        ds = self.datasets[ds_idx]
        item = ds[idx]
        return item


class RandomAudioChunkDataset(Dataset):
    def __init__(
            self,
            input_dir: str,
            n_samples: int,
            sr: float,
            ext: str = "wav",
            num_examples_per_epoch: int = 10000,
            silence_fraction_allowed: float = 0.2,
            silence_threshold_energy: float = 1e-6,  # Around -60 dBFS
            n_retries: int = 10,
            use_debug_mode: bool = False,
            check_dataset: bool = True,
            min_suitable_files_fraction: int = 0.5,
            end_buffer_n_samples: int = 0,
    ) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.n_samples = n_samples
        self.sr = sr
        self.ext = ext
        self.num_examples_per_epoch = num_examples_per_epoch
        self.silence_fraction_allowed = silence_fraction_allowed
        self.silence_threshold_energy = silence_threshold_energy
        self.n_retries = n_retries
        self.use_debug_mode = use_debug_mode
        self.check_dataset = check_dataset
        self.min_suitable_files_fraction = min_suitable_files_fraction
        self.end_buffer_n_samples = end_buffer_n_samples
        self.max_n_consecutive_silent_samples = int(silence_fraction_allowed * n_samples)

        input_paths = self.get_file_paths(input_dir, ext)

        total_n_samples = 0
        filtered_input_paths = []
        for input_path in input_paths:
            file_info = torchaudio.info(input_path)
            if file_info.num_frames < n_samples:
                log.debug(f"Too short, removing: {input_path}")
                continue
            if file_info.sample_rate != sr:
                log.info(f"Bad sample rate of {file_info.sample_rate}, removing: {input_path}")
                continue
            total_n_samples += file_info.num_frames
            filtered_input_paths.append(input_path)
        log.info(f"Filtered down to {len(filtered_input_paths)} input files")
        log.info(f"Found {total_n_samples / sr:.0f} seconds ({total_n_samples / sr / 60.0:.2f} minutes) of audio")
        assert len(filtered_input_paths) > 0

        self.input_paths = filtered_input_paths
        if check_dataset:
            assert self.check_dataset_for_suitable_files(n_samples,
                                                         min_suitable_files_fraction,
                                                         end_buffer_n_samples), \
                "Could not find a suitable non-silent audio chunk in the dataset"

    def check_dataset_for_suitable_files(self,
                                         n_samples: int,
                                         min_suitable_files_fraction: float,
                                         end_buffer_n_samples: int = 0) -> bool:
        min_n_suitable_files = int(min_suitable_files_fraction * len(self.input_paths))
        min_n_suitable_files = max(1, min_n_suitable_files)
        n_suitable_files = 0
        for file_path in tqdm(self.input_paths):
            for _ in range(self.n_retries):
                audio_chunk = self.find_audio_chunk_in_file(file_path, n_samples, end_buffer_n_samples)
                if audio_chunk is not None:
                    n_suitable_files += 1
                    break
        log.info(f"Found {n_suitable_files} suitable files out of {len(self.input_paths)} files "
                 f"({n_suitable_files / len(self.input_paths) * 100:.2f}%)")
        return n_suitable_files >= min_n_suitable_files

    def check_for_silence(self, audio_chunk: T) -> bool:
        window_size = self.max_n_consecutive_silent_samples
        hop_len = window_size // 4
        energy = audio_chunk ** 2
        unfolded = energy.unfold(dimension=-1, size=window_size, step=hop_len)
        mean_energies = tr.mean(unfolded, dim=-1, keepdim=False)
        n_silent = (mean_energies < self.silence_threshold_energy).sum().item()
        return n_silent > 0

    def find_audio_chunk_in_file(self,
                                 file_path: str,
                                 n_samples: int,
                                 end_buffer_n_samples: int = 0) -> Optional[Tuple[T, int]]:
        file_n_samples = torchaudio.info(file_path).num_frames
        if n_samples > file_n_samples - end_buffer_n_samples:
            return None
        start_idx = self.randint(0, file_n_samples - n_samples - end_buffer_n_samples + 1)
        audio_chunk, sr = torchaudio.load(
            file_path,
            frame_offset=start_idx,
            num_frames=n_samples,
        )
        if self.check_for_silence(audio_chunk):
            log.debug("Skipping audio chunk because of silence")
            return None
        return audio_chunk, start_idx

    def search_dataset_for_audio_chunk(self, n_samples: int, end_buffer_n_samples: int = 0) -> (T, str, int, int):
        file_path_pool = list(self.input_paths)
        file_path = self.choice(file_path_pool)
        file_path_pool.remove(file_path)
        audio_chunk = None
        n_attempts = 0

        while audio_chunk is None:
            audio_chunk = self.find_audio_chunk_in_file(file_path, n_samples, end_buffer_n_samples)
            if audio_chunk is None:
                n_attempts += 1
            if n_attempts >= self.n_retries:
                assert file_path_pool, "This should never happen if `check_dataset_for_suitable_files` was run"
                file_path = self.choice(file_path_pool)
                file_path_pool.remove(file_path)
                n_attempts = 0

        audio_chunk, start_idx = audio_chunk
        ch_idx = 0
        if audio_chunk.size(0) > 1:
            ch_idx = self.randint(0, audio_chunk.size(0))
            audio_chunk = audio_chunk[ch_idx, :].view(1, -1)

        return audio_chunk, file_path, ch_idx, start_idx

    def __len__(self) -> int:
        return self.num_examples_per_epoch

    def __getitem__(self, _) -> T:
        audio_chunk, _, _, _ = self.search_dataset_for_audio_chunk(self.n_samples, self.end_buffer_n_samples)
        return audio_chunk

    @staticmethod
    def get_file_paths(input_dir: str, ext: str) -> List[str]:
        assert os.path.isdir(input_dir)
        input_paths = []
        for root_dir, _, file_names in os.walk(input_dir):
            for file_name in file_names:
                if file_name.endswith(ext) and not file_name.startswith("."):
                    input_paths.append(os.path.join(root_dir, file_name))
        input_paths = sorted(input_paths)
        log.info(f"Found {len(input_paths)} files in {input_dir}")
        assert len(input_paths) > 0
        return input_paths

    @staticmethod
    def choice(items: List[Any]) -> Any:
        assert len(items) > 0
        idx = RandomAudioChunkDataset.randint(0, len(items))
        return items[idx]

    @staticmethod
    def randint(low: int, high: int, n: int = 1) -> Union[int, T]:
        x = tr.randint(low=low, high=high, size=(n,))
        if n == 1:
            return x.item()
        return x

    @staticmethod
    def sample_uniform(low: float, high: float, n: int = 1) -> Union[float, T]:
        x = (tr.rand(n) * (high - low)) + low
        if n == 1:
            return x.item()
        return x

    @staticmethod
    def sample_log_uniform(low: float, high: float, n: int = 1) -> Union[float, T]:
        # TODO(cm): replace with torch
        if low == high:
            if n == 1:
                return low
            else:
                return tr.full(size=(n,), fill_value=low)
        x = loguniform.rvs(low, high, size=n)
        if n == 1:
            return float(x)
        return torch.from_numpy(x)


class RandomAudioChunkDryWetDataset(RandomAudioChunkDataset):
    def __init__(
            self,
            dry_dir: str,
            wet_dir: str,
            n_samples: int,
            sr: float,
            ext: str = "wav",
            num_examples_per_epoch: int = 10000,
            silence_fraction_allowed: float = 0.1,
            silence_threshold_energy: float = 1e-6,
            n_retries: int = 10,
            use_debug_mode: bool = False,
            check_dataset: bool = True,
            min_suitable_files_fraction: int = 0.5,
            end_buffer_n_samples: int = 0,
    ) -> None:
        super().__init__(dry_dir,
                         n_samples,
                         sr,
                         ext,
                         num_examples_per_epoch,
                         silence_fraction_allowed,
                         silence_threshold_energy,
                         n_retries,
                         use_debug_mode,
                         check_dataset,
                         min_suitable_files_fraction,
                         end_buffer_n_samples)
        self.dry_dir = dry_dir
        self.wet_dir = wet_dir
        self.end_buffer_n_samples = end_buffer_n_samples
        all_wet_paths = self.get_file_paths(wet_dir, ext)
        dry_paths = self.input_paths
        all_wet_names_to_wet_path = {os.path.basename(p): p for p in all_wet_paths}
        wet_paths = []
        name_to_wet_path = {}
        for dry_p in dry_paths:
            name = os.path.basename(dry_p)
            assert name in all_wet_names_to_wet_path, f"Missing wet file: {name}"
            wet_p = all_wet_names_to_wet_path[name]
            dry_info = torchaudio.info(dry_p)
            wet_info = torchaudio.info(wet_p)
            assert dry_info.sample_rate == wet_info.sample_rate, f"Different sample rates: {dry_p}, {wet_p}"
            assert abs(dry_info.num_frames - wet_info.num_frames) <= end_buffer_n_samples, \
                f"Different lengths: {dry_p}: {dry_info.num_frames}, {wet_p}: {wet_info.num_frames}"
            assert dry_info.num_channels == wet_info.num_channels, f"Different channels: {dry_p}, {wet_p}"
            wet_paths.append(wet_p)
            name_to_wet_path[name] = wet_p
        wet_paths = sorted(wet_paths)
        self.dry_paths = dry_paths
        self.wet_paths = wet_paths
        self.name_to_wet_path = name_to_wet_path

    def __getitem__(self, _) -> (T, T):
        dry_chunk, dry_path, ch_idx, start_idx = self.search_dataset_for_audio_chunk(self.n_samples,
                                                                                     self.end_buffer_n_samples)
        dry_name = os.path.basename(dry_path)
        wet_path = self.name_to_wet_path[dry_name]
        wet_chunk, _ = torchaudio.load(
            wet_path,
            frame_offset=start_idx,
            num_frames=self.n_samples,
        )
        if wet_chunk.size(0) > 1:
            wet_chunk = wet_chunk[ch_idx, :].view(1, -1)
        assert dry_chunk.shape == wet_chunk.shape
        return dry_chunk, wet_chunk


class RandomAudioChunkAndModSigDataset(RandomAudioChunkDataset):
    def __init__(
            self,
            fx_config: Dict[str, Any],
            input_dir: str,
            n_samples: int,
            sr: float,
            ext: str = "wav",
            num_examples_per_epoch: int = 10000,
            silence_fraction_allowed: float = 0.1,
            silence_threshold_energy: float = 1e-6,
            n_retries: int = 10,
            use_debug_mode: bool = False,
            check_dataset: bool = True,
            min_suitable_files_fraction: int = 0.5,
            end_buffer_n_samples: int = 0,
    ) -> None:
        super().__init__(input_dir,
                         n_samples,
                         sr,
                         ext,
                         num_examples_per_epoch,
                         silence_fraction_allowed,
                         silence_threshold_energy,
                         n_retries,
                         use_debug_mode,
                         check_dataset,
                         min_suitable_files_fraction,
                         end_buffer_n_samples)
        self.fx_config = fx_config

    def __getitem__(self, _) -> (T, T, Dict[str, T]):
        audio_chunk = super().__getitem__(_)
        rate_hz = self.sample_log_uniform(self.fx_config["mod_sig"]["rate_hz"]["min"],
                                          self.fx_config["mod_sig"]["rate_hz"]["max"])
        phase = self.sample_uniform(self.fx_config["mod_sig"]["phase"]["min"],
                                    self.fx_config["mod_sig"]["phase"]["max"])
        shape = self.choice(self.fx_config["mod_sig"]["shapes"])
        exp = self.fx_config["mod_sig"]["exp"]
        mod_sig = make_mod_signal(self.n_samples, self.sr, rate_hz, phase, shape, exp)
        fx_params = {
            "rate_hz": rate_hz,
            "phase": phase,
            "shape": shape,
            "exp": exp,
        }
        return audio_chunk, mod_sig, fx_params


class PedalboardPhaserDataset(RandomAudioChunkAndModSigDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert "pedalboard_phaser" in self.fx_config
        self.max_file_n_samples = 0
        for file_path in self.input_paths:
            file_n_samples = torchaudio.info(file_path).num_frames
            if file_n_samples > self.max_file_n_samples:
                self.max_file_n_samples = file_n_samples
        log.info(f"max_file_n_samples = {self.max_file_n_samples} ({self.max_file_n_samples / self.sr:.2f} seconds)")

        dataset_min_rate_period = (self.max_file_n_samples - self.n_samples) / self.sr
        dataset_min_rate_hz = 1 / dataset_min_rate_period
        phaser_min_rate_hz = self.fx_config["pedalboard_phaser"]["rate_hz"]["min"]
        log.info(f"dataset_min_rate_hz = {dataset_min_rate_hz:.4f}")
        log.info(f" phaser_min_rate_hz = {phaser_min_rate_hz:.4f}")
        assert dataset_min_rate_hz <= phaser_min_rate_hz

        min_rate_n_samples = int((self.sr / phaser_min_rate_hz) + 0.5)
        max_proc_n_samples = self.n_samples + min_rate_n_samples
        log.debug(f"max_proc_n_samples = {max_proc_n_samples}")

        if self.check_dataset:
            assert self.check_dataset_for_suitable_files(max_proc_n_samples, 0.1), \
                "Could not find a suitable non-silent audio chunk in the dataset to support the lowest phaser rate_hz"
            log.info(f">10% of the dataset can handle the max_proc_n_samples required for the lowest phaser rate_hz")

    def __getitem__(self, idx: int) -> (T, T, T, Dict[str, float]):
        rate_hz = self.sample_log_uniform(
            self.fx_config["pedalboard_phaser"]["rate_hz"]["min"],
            self.fx_config["pedalboard_phaser"]["rate_hz"]["max"],
        )
        rate_n_samples = int((self.sr / rate_hz) + 0.5)
        proc_n_samples = self.n_samples + rate_n_samples

        audio_chunk, _, _, _ = self.search_dataset_for_audio_chunk(proc_n_samples, self.end_buffer_n_samples)

        proc_audio, fx_params = self.apply_pedalboard_phaser(audio_chunk,
                                                             self.sr,
                                                             rate_hz,
                                                             self.fx_config["pedalboard_phaser"])
        proc_mod_sig = make_mod_signal(proc_n_samples, self.sr, rate_hz, tr.pi / 2, "cos")

        # TODO(cm): calc phase and add to fx_params
        start_idx = self.randint(0, proc_n_samples - self.n_samples + 1)
        dry = audio_chunk[:, start_idx:start_idx + self.n_samples]
        wet = proc_audio[:, start_idx:start_idx + self.n_samples]
        mod_sig = proc_mod_sig[start_idx:start_idx + self.n_samples]

        if self.use_debug_mode:
            plt.plot(mod_sig.squeeze(0))
            plt.title(f"phaser_mod_sig_{idx}")
            plt.show()
            plot_spectrogram(dry, title=f"phaser_dry_{idx}", save_name=f"phaser_dry_{idx}", sr=self.sr)
            plot_spectrogram(wet, title=f"phaser_wet_{idx}", save_name=f"phaser_wet_{idx}", sr=self.sr)

        fx_params = defaultdict(float, fx_params)  # TODO(cm)
        mod_sig = linear_interpolate_last_dim(mod_sig, 882, align_corners=True)  # TODO
        return dry, wet, mod_sig, fx_params

    @staticmethod
    def apply_pedalboard_phaser(x: T,
                                sr: float,
                                rate_hz: float,
                                ranges: Dict[str, Dict[str, float]]) -> (T, Dict[str, float]):
        board = Pedalboard()
        depth = RandomAudioChunkDataset.sample_uniform(ranges["depth"]["min"], ranges["depth"]["max"])
        centre_frequency_hz = RandomAudioChunkDataset.sample_log_uniform(ranges["centre_frequency_hz"]["min"],
                                                                         ranges["centre_frequency_hz"]["max"])
        feedback = RandomAudioChunkDataset.sample_uniform(ranges["feedback"]["min"], ranges["feedback"]["max"])
        mix = RandomAudioChunkDataset.sample_uniform(ranges["mix"]["min"], ranges["mix"]["max"])
        board.append(Phaser(rate_hz=rate_hz,
                            depth=depth,
                            centre_frequency_hz=centre_frequency_hz,
                            feedback=feedback,
                            mix=mix))
        y = tr.from_numpy(board(x.numpy(), sr))
        y = tr.clip(y, -1.0, 1.0)  # TODO(cm): should clip flag
        fx_params = {
            "depth": depth,
            # "centre_frequency_hz": centre_frequency_hz,  # TODO
            "feedback": feedback,
            "mix": mix,
            "rate_hz": rate_hz,
            "shape": "cos",
        }
        return y, fx_params


class TremoloDataset(RandomAudioChunkAndModSigDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert "tremolo" in self.fx_config

    def __getitem__(self, idx: int) -> (T, T, T, Dict[str, float]):
        dry, mod_sig, fx_params = super().__getitem__(idx)
        mix = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["tremolo"]["mix"]["min"],
            self.fx_config["tremolo"]["mix"]["max"],
        )
        fx_params["mix"] = mix
        wet = fx.apply_tremolo(dry.unsqueeze(0), mod_sig.unsqueeze(0), mix)
        wet = wet.squeeze(0)

        # TODO(cm): Extract into method
        if self.use_debug_mode:
            plt.plot(mod_sig)
            plt.title(f"tremolo_mod_sig_{idx}")
            plt.show()
            plot_spectrogram(dry, title=f"tremolo_dry_{idx}", save_name=f"tremolo_dry_{idx}", sr=self.sr)
            plot_spectrogram(wet, title=f"tremolo_wet_{idx}", save_name=f"tremolo_wet_{idx}", sr=self.sr)

        fx_params = defaultdict(float, fx_params)  # TODO(cm)
        return dry, wet, mod_sig, fx_params


class PreprocessedDataset(Dataset):
    def __init__(self,
                 input_dir: str,
                 n_samples: int,
                 sr: float,
                 use_debug_mode: bool = False) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.n_samples = n_samples
        self.sr = sr
        self.use_debug_mode = use_debug_mode
        self.pt_paths = RandomAudioChunkDataset.get_file_paths(input_dir, ".pt")
        self.dry_paths = [f"{p[:-3]}_dry.wav" for p in self.pt_paths]
        self.wet_paths = [f"{p[:-3]}_wet.wav" for p in self.pt_paths]

    def __len__(self) -> int:
        return len(self.pt_paths)

    def __getitem__(self, idx: int) -> (T, T, T, Dict[str, Any]):
        pt_path = self.pt_paths[idx]
        dry_path = self.dry_paths[idx]
        wet_path = self.wet_paths[idx]
        data = tr.load(pt_path)
        mod_sig = data["mod_sig"]
        fx_params = data["fx_params"]
        dry, sr = torchaudio.load(dry_path)
        assert sr == self.sr
        assert dry.size(-1) == self.n_samples
        wet, sr = torchaudio.load(wet_path)
        assert sr == self.sr
        assert wet.size(-1) == self.n_samples

        if self.use_debug_mode:
            plt.plot(mod_sig)
            plt.title(f"mod_sig_{idx}")
            plt.show()
            plot_spectrogram(dry, title=f"dry_{idx}", save_name=f"dry_{idx}", sr=sr)
            plot_spectrogram(wet, title=f"wet_{idx}", save_name=f"wet_{idx}", sr=sr)

        return dry, wet, mod_sig, fx_params


class RandomPreprocessedDataset(PreprocessedDataset):
    def __init__(self,
                 num_examples_per_epoch: int,
                 input_dir: str,
                 n_samples: int,
                 sr: float,
                 use_debug_mode: bool = False) -> None:
        super().__init__(input_dir, n_samples, sr, use_debug_mode)
        self.num_examples_per_epoch = num_examples_per_epoch

    def __len__(self) -> int:
        return self.num_examples_per_epoch

    def __getitem__(self, idx: int) -> (T, T, T, Dict[str, Any]):
        rand_idx = RandomAudioChunkDataset.randint(0, len(self.pt_paths))
        return super().__getitem__(rand_idx)
