import logging
import os
from typing import Dict, Optional, List, Any

import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from pedalboard import Pedalboard, Phaser
from scipy.stats import loguniform
from torch import Tensor as T
from torch.utils.data import Dataset
from tqdm import tqdm

from lfo_tcn.fx import make_mod_signal
from lfo_tcn.plotting import plot_spectrogram

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class RandomAudioChunkDataset(Dataset):
    def __init__(
            self,
            input_dir: str,
            n_samples: int,
            sr: float = 44100,
            ext: str = "wav",
            num_examples_per_epoch: int = 10000,
            silence_fraction_allowed: float = 0.2,
            silence_threshold_energy: float = 1e-6,  # Around -60 dBFS
            n_retries: int = 20,
            check_dataset: bool = True,
            min_suitable_files_fraction: int = 0.5,
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
        self.check_dataset = check_dataset
        self.min_suitable_files_fraction = min_suitable_files_fraction
        self.max_n_consecutive_silent_samples = int(silence_fraction_allowed * n_samples)

        assert os.path.isdir(input_dir)
        input_paths = []
        for root_dir, _, file_names in os.walk(input_dir):
            for file_name in file_names:
                if file_name.endswith(ext) and not file_name.startswith("."):
                    input_paths.append(os.path.join(root_dir, file_name))
        input_paths = sorted(input_paths)
        log.info(f"Found {len(input_paths)} input files")
        assert len(input_paths) > 0

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
            assert self.check_dataset_for_suitable_files(self.n_samples, self.min_suitable_files_fraction), \
                "Could not find a suitable non-silent audio chunk in the dataset"

    def check_dataset_for_suitable_files(self, n_samples: int, min_suitable_files_fraction: float) -> bool:
        min_n_suitable_files = int(min_suitable_files_fraction * len(self.input_paths))
        min_n_suitable_files = max(1, min_n_suitable_files)
        n_suitable_files = 0
        for file_path in tqdm(self.input_paths):
            for _ in range(self.n_retries):
                audio_chunk = self.find_audio_chunk_in_file(file_path, n_samples)
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

    def find_audio_chunk_in_file(self, file_path: str, n_samples: int) -> Optional[T]:
        file_n_samples = torchaudio.info(file_path).num_frames
        if n_samples > file_n_samples:
            return None
        start_idx = self.randint(0, file_n_samples - n_samples + 1)
        audio_chunk, sr = torchaudio.load(
            file_path,
            frame_offset=start_idx,
            num_frames=n_samples,
        )
        if self.check_for_silence(audio_chunk):
            log.debug("Skipping audio chunk because of silence")
            return None
        return audio_chunk

    def search_dataset_for_audio_chunk(self, n_samples: int) -> T:
        file_path_pool = list(self.input_paths)
        file_path = self.choice(file_path_pool)
        file_path_pool.remove(file_path)
        audio_chunk = None
        n_attempts = 0

        while audio_chunk is None:
            audio_chunk = self.find_audio_chunk_in_file(file_path, n_samples)
            if audio_chunk is None:
                n_attempts += 1
            if n_attempts >= self.n_retries:
                assert file_path_pool, "This should never happen if `check_dataset_for_suitable_files` was run"
                file_path = self.choice(file_path_pool)
                file_path_pool.remove(file_path)
                n_attempts = 0

        if audio_chunk.size(0) > 1:
            ch_idx = self.randint(0, audio_chunk.size(0))
            audio_chunk = audio_chunk[ch_idx, :].view(1, -1)

        return audio_chunk

    def __len__(self):
        return self.num_examples_per_epoch

    def __getitem__(self, _) -> T:
        audio_chunk = self.search_dataset_for_audio_chunk(self.n_samples)
        return audio_chunk

    @staticmethod
    def choice(items: List[Any]) -> Any:
        assert len(items) > 0
        idx = RandomAudioChunkDataset.randint(0, len(items))
        return items[idx]

    @staticmethod
    def randint(low: int, high: int) -> int:
        return tr.randint(low=low, high=high, size=(1,)).item()

    @staticmethod
    def sample_uniform(low: float, high: float) -> float:
        return (tr.rand(1).item() * (high - low)) + low

    @staticmethod
    def sample_log_uniform(low: float, high: float) -> float:
        # TODO(cm): replace with torch
        return float(loguniform.rvs(low, high))


class RandomAudioChunkAndModSigDataset(RandomAudioChunkDataset):
    def __init__(
            self,
            input_dir: str,
            n_samples: int,
            fx_config: Dict[str, Any],
            sr: float = 44100,
            ext: str = "wav",
            num_examples_per_epoch: int = 10000,
            silence_fraction_allowed: float = 0.2,
            silence_threshold_energy: float = 1e-6,
            n_retries: int = 20,
            use_debug_mode: bool = False,
            check_dataset: bool = True,
            min_suitable_files_fraction: int = 0.5,
    ) -> None:
        super().__init__(input_dir,
                         n_samples,
                         sr,
                         ext,
                         num_examples_per_epoch,
                         silence_fraction_allowed,
                         silence_threshold_energy,
                         n_retries,
                         check_dataset,
                         min_suitable_files_fraction)
        self.fx_config = fx_config
        self.use_debug_mode = use_debug_mode

    def __getitem__(self, _) -> (T, T):
        audio_chunk = super().__getitem__(_)
        rate_hz = self.sample_log_uniform(self.fx_config["mod_sig"]["rate_hz"]["min"],
                                          self.fx_config["mod_sig"]["rate_hz"]["max"])
        phase = self.sample_uniform(self.fx_config["mod_sig"]["phase"]["min"],
                                    self.fx_config["mod_sig"]["phase"]["max"])
        shape = self.choice(self.fx_config["mod_sig"]["shapes"])
        mod_sig = make_mod_signal(self.n_samples, self.sr, rate_hz, phase, shape)
        return audio_chunk, mod_sig


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

        audio_chunk = self.search_dataset_for_audio_chunk(proc_n_samples)

        proc_audio, fx_params = self.apply_pedalboard_phaser(audio_chunk,
                                                             self.sr,
                                                             rate_hz,
                                                             self.fx_config["pedalboard_phaser"])
        proc_mod_sig = make_mod_signal(proc_n_samples, self.sr, rate_hz, tr.pi / 2, "cos")

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
        fx_params = {
            "depth": depth,
            "centre_frequency_hz": centre_frequency_hz,
            "feedback": feedback,
            "mix": mix,
            "rate_hz": rate_hz,
        }
        return y, fx_params
