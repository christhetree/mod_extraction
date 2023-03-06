import glob
import logging
import os
from typing import Dict, Optional, List, Any

import torch as tr
import torchaudio
import yaml
from matplotlib import pyplot as plt
from pedalboard import Pedalboard, Phaser
from scipy.stats import loguniform
from torch import Tensor as T
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram

from lfo_tcn.fx import make_mod_signal

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
            silence_threshold_energy: float = 1e-4,
            n_retries: int = 20,
    ) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.n_samples = n_samples
        self.sr = sr
        self.ext = ext
        self.num_examples_per_epoch = num_examples_per_epoch
        self.silence_threshold_energy = silence_threshold_energy
        self.n_retries = n_retries

        assert os.path.isdir(input_dir)
        input_paths = glob.glob(os.path.join(input_dir, f"*.{ext}"))
        input_paths = sorted(input_paths)
        log.info(f"Found {len(input_paths)} input files")
        assert len(input_paths) > 0
        filtered_input_paths = []
        for input_path in input_paths:
            file_info = torchaudio.info(input_path)
            if file_info.num_frames < n_samples:
                log.info(f"Too short, removing: {input_path}")
                continue
            if file_info.sample_rate != sr:
                log.info(f"Bad sample rate, removing: {input_path}")
                continue
            filtered_input_paths.append(input_path)
        log.info(f"Filtered down to {len(filtered_input_paths)} input files")
        self.input_paths = filtered_input_paths

    def check_for_silence(self, audio_chunk: T) -> bool:
        return (audio_chunk ** 2).mean() < self.silence_threshold_energy

    def find_audio_chunk(self, file_path: str, n_samples: int) -> Optional[T]:
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
            return None
        return audio_chunk

    def __len__(self):
        return self.num_examples_per_epoch

    def __getitem__(self, _) -> T:
        file_path = self.choice(self.input_paths)
        audio_chunk = None
        n_attempts = 0
        while audio_chunk is None:
            audio_chunk = self.find_audio_chunk(file_path, self.n_samples)
            if audio_chunk is not None:
                n_attempts += 1
            if n_attempts > self.n_retries:
                file_path = self.choice(self.input_paths)
                n_attempts = 0

        if audio_chunk.size(0) > 1:
            ch_idx = self.randint(0, audio_chunk.size(0))
            audio_chunk = audio_chunk[ch_idx, :].view(1, -1)

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
            silence_threshold_energy: float = 1e-4,
            n_retries: int = 20,
            use_debug_mode: bool = False,
    ) -> None:
        super().__init__(input_dir, n_samples, sr, ext, num_examples_per_epoch, silence_threshold_energy, n_retries)
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


class PedalboardPhaserGeneratorDataset(RandomAudioChunkAndModSigDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_file_n_samples = 0
        for file_path in self.input_paths:
            file_n_samples = torchaudio.info(file_path).num_frames
            if file_n_samples > self.max_file_n_samples:
                self.max_file_n_samples = file_n_samples
        log.info(f"max_file_n_samples = {self.max_file_n_samples}")
        dataset_min_rate_period = (self.max_file_n_samples - self.n_samples) / self.sr
        dataset_min_rate_hz = 1 / dataset_min_rate_period
        log.info(f"dataset_min_rate_hz = {dataset_min_rate_hz:.2f}")
        assert dataset_min_rate_hz <= self.fx_config["pedalboard_phaser"]["rate_hz"]["min"]
        self.spectrogram = Spectrogram(n_fft=512, power=1, center=True, normalized=False)

    def __getitem__(self, _) -> (T, T, T):
        rate_hz = self.sample_log_uniform(
            self.fx_config["pedalboard_phaser"]["rate_hz"]["min"],
            self.fx_config["pedalboard_phaser"]["rate_hz"]["max"],
        )
        rate_n_samples = int((self.sr / rate_hz) + 0.5)
        audio_chunk = None
        file_idx = self.randint(0, len(self.input_paths))
        file_path = self.input_paths[file_idx]
        proc_n_samples = self.n_samples + rate_n_samples
        n_attempts = 0
        while audio_chunk is None:
            audio_chunk = self.find_audio_chunk(file_path, proc_n_samples)
            if audio_chunk is None:
                n_attempts += 1
            if n_attempts > self.n_retries:
                file_idx = self.randint(0, len(self.input_paths))
                file_path = self.input_paths[file_idx]
                n_attempts = 0

        if audio_chunk.size(0) > 1:
            ch_idx = self.randint(0, audio_chunk.size(0))
            audio_chunk = audio_chunk[ch_idx, :].view(1, -1)

        proc_audio, _ = self.apply_pedalboard_phaser(audio_chunk,
                                                     self.sr,
                                                     rate_hz,
                                                     self.fx_config["pedalboard_phaser"])
        proc_mod_sig = make_mod_signal(proc_n_samples, self.sr, rate_hz, tr.pi / 2, "cos")

        start_idx = self.randint(0, proc_n_samples - self.n_samples + 1)
        dry = audio_chunk[:, start_idx:start_idx + self.n_samples]
        wet = proc_audio[:, start_idx:start_idx + self.n_samples]
        mod_sig = proc_mod_sig[start_idx:start_idx + self.n_samples]

        if self.use_debug_mode:
            plt.plot(mod_sig)
            plt.show()
            torchaudio.save("../out/dry.wav", dry, self.sr)
            torchaudio.save("../out/wet.wav", wet, self.sr)
            y_spec = tr.log(self.spectrogram(wet.squeeze()))
            plt.imshow(y_spec, aspect="auto", interpolation="none")
            plt.show()

        return dry, wet, mod_sig

    @staticmethod
    def apply_pedalboard_phaser(x: T,
                                sr: float,
                                rate_hz: float,
                                ranges: Dict[str, Dict[str, float]]) -> (T, T):
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
        p = tr.tensor([rate_hz,
                       depth,
                       centre_frequency_hz,
                       feedback,
                       mix])
        return y, p
