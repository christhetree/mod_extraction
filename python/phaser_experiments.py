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
from torch import Tensor as T, nn
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def make_mod_signal(n_samples: int,
                    sr: float,
                    freq: float,
                    phase: float = 0.0,
                    shape: str = "cos") -> T:
    assert n_samples > 0
    assert 0.0 < freq < sr / 2.0
    assert -2 * tr.pi <= phase <= 2 * tr.pi
    assert shape in {"cos", "tri", "saw", "rsaw", "sqr"}
    argument = tr.cumsum(2 * tr.pi * tr.full((n_samples,), freq) / sr, dim=0) + phase

    if shape == "cos":
        return (tr.cos(argument + tr.pi) + 1.0) / 2.0
    if shape == "sqr":
        cos = tr.cos(argument + tr.pi)
        sqr = tr.sign(cos)
        return (sqr + 1.0) / 2.0
    saw = tr.remainder(argument, 2 * tr.pi) / (2 * tr.pi)
    if shape == "saw":
        return saw
    if shape == "rsaw":
        return 1.0 - saw
    tri = 2 * saw
    tri = tr.where(tri > 1.0, 2.0 - tri, tri)
    return tri


def apply_tremolo(x: T, mod_sig: T, mix: float = 1.0) -> T:
    assert x.size(-1) == mod_sig.size(-1)
    assert 0.0 <= mix <= 1.0

    return ((1.0 - mix) * x) + (mix * mod_sig * x)


def apply_delay(x: T, mod_sig: T, mix: float = 1.0, use_feedback: bool = False) -> T:
    assert x.size(-1) == mod_sig.size(-1)
    out = tr.clone(x)
    for x_idx in range(x.size(-1)):
        delay_idx = int(x_idx - mod_sig[..., x_idx])
        if 0 <= delay_idx < x.size(-1):
            if use_feedback:
                out[..., x_idx] = ((1.0 - mix) * out[..., x_idx]) + (mix * out[..., delay_idx])
            else:
                out[..., x_idx] = ((1.0 - mix) * x[..., x_idx]) + (mix * x[..., delay_idx])
    return out


def apply_flanger(x: T,
                  mod_sig: T,
                  delay_buf: T,
                  out_buf: T,
                  max_delay_samples: int,
                  feedback: float = 0.0,
                  width: float = 1.0,
                  depth: float = 1.0,
                  mix: float = 1.0) -> T:
                  # max_delay_ms: float = 10.0,
                  # sr: float = 44100) -> T:
    assert x.ndim == 3
    assert x.size(0) == mod_sig.size(0)
    assert x.size(-1) == mod_sig.size(-1)
    assert 0.0 <= width <= 1.0
    assert 0.0 <= feedback < 1.0
    assert 0.0 <= depth <= 1.0
    assert 0.0 <= mix <= 1.0
    if width == 0.0 or depth == 0.0 or mix == 0.0:
        return x
    batch_size, n_ch, n_samples = x.shape
    if mod_sig.ndim == 2:
        mod_sig = mod_sig.unsqueeze(1).expand(-1, n_ch, -1)

    # max_delay_samples = int(((max_delay_ms / 1000.0) * sr) + 0.5)
    # delay_buf = tr.zeros((batch_size, n_ch, max_delay_samples))
    # out_buf = tr.zeros_like(x)

    delay_write_idx = 0
    for idx in range(n_samples):
        audio_val = x[:, :, idx]
        mod_val = mod_sig[:, :, idx]
        delay_samples = max_delay_samples * width * mod_val
        delay_read_idx = (delay_write_idx - delay_samples + max_delay_samples) % max_delay_samples
        delay_read_fraction = delay_read_idx - tr.floor(delay_read_idx)
        prev_idx = tr.floor(delay_read_idx).to(tr.long).unsqueeze(-1)
        next_idx = (prev_idx + 1) % max_delay_samples
        prev_val = tr.gather(delay_buf, dim=-1, index=prev_idx).squeeze(-1)
        next_val = tr.gather(delay_buf, dim=-1, index=next_idx).squeeze(-1)
        interp_val = (delay_read_fraction * next_val) + ((1.0 - delay_read_fraction) * prev_val)
        delay_buf[:, :, delay_write_idx] = audio_val + (feedback * interp_val)
        out_buf[:, :, idx] = audio_val + (depth * interp_val)

        delay_write_idx += 1
        if delay_write_idx == max_delay_samples:
            delay_write_idx = 0

    out_buf = ((1.0 - mix) * x) + (mix * out_buf)
    return out_buf


class FlangerModule(nn.Module):
    def __init__(self,
                 batch_size: int,
                 n_ch: int,
                 n_samples: int,
                 max_delay_ms: float = 10.0,
                 sr: float = 44100) -> None:
        super().__init__()
        self.max_delay_samples = int(((max_delay_ms / 1000.0) * sr) + 0.5)
        self.register_buffer("delay_buf", tr.zeros((batch_size, n_ch, self.max_delay_samples)))
        self.register_buffer("out_buf", tr.zeros((batch_size, n_ch, n_samples)))

    def forward(self,
                x: T,
                mod_sig: T,
                feedback: float = 0.0,
                width: float = 1.0,
                depth: float = 1.0,
                mix: float = 1.0) -> T:
        with tr.no_grad():
            self.delay_buf.fill_(0)
            self.out_buf.fill_(0)
            return apply_flanger(x,
                                 mod_sig,
                                 self.delay_buf,
                                 self.out_buf,
                                 self.max_delay_samples,
                                 feedback,
                                 width,
                                 depth,
                                 mix)


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
            fx_config_path: str,
            sr: float = 44100,
            ext: str = "wav",
            num_examples_per_epoch: int = 10000,
            silence_threshold_energy: float = 1e-4,
            n_retries: int = 20,
            use_debug_mode: bool = False,
    ) -> None:
        super().__init__(input_dir, n_samples, sr, ext, num_examples_per_epoch, silence_threshold_energy, n_retries)
        self.use_debug_mode = use_debug_mode
        assert os.path.isfile(fx_config_path)
        with open(fx_config_path, "r") as in_f:
            self.fx_config = yaml.safe_load(in_f)

    def __getitem__(self, _) -> (T, T):
        audio_chunk = super().__getitem__(_)
        rate_hz = self.sample_log_uniform(self.fx_config["mod_sig"]["rate_hz"]["min"],
                                          self.fx_config["mod_sig"]["rate_hz"]["max"])
        phase = self.sample_uniform(self.fx_config["mod_sig"]["phase"]["min"],
                                    self.fx_config["mod_sig"]["phase"]["max"])
        shape = self.choice(self.fx_config["mod_sig"]["shapes"])
        mod_sig = make_mod_signal(self.n_samples, self.sr, rate_hz, phase, shape)
        return audio_chunk, mod_sig


class PhaserGeneratorDataset(RandomAudioChunkAndModSigDataset):
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
        log.info(f"dataset_min_rate_hz = {dataset_min_rate_hz}")
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


if __name__ == "__main__":
    audio, sr = torchaudio.load("../out/shoes_pad.wav")
    audio = audio.unsqueeze(0)
    # audio = audio[:, :, :1024]
    audio = audio[:, :, :88200]
    audio = audio.repeat(2, 1, 1)
    # sr = 44100
    # n_samples = 1024
    n_samples = audio.size(-1)
    n_ch = audio.size(1)
    batch_size = audio.size(0)
    # audio = tr.rand((batch_size, 2, n_samples))

    mod_sig_0 = make_mod_signal(n_samples, sr, freq=2.0, phase=0.0, shape="cos")
    mod_sig_1 = make_mod_signal(n_samples, sr, freq=0.1, phase=tr.pi, shape="cos")
    mod_sig = tr.stack([mod_sig_0, mod_sig_1], dim=0)
    wet = apply_flanger(audio, mod_sig, width=1.0, feedback=0.2, depth=1.0, mix=1.0, max_delay_ms=5.0, sr=sr)

    spectrogram = Spectrogram(n_fft=512, power=1, center=True, normalized=False)
    for idx, (w, m) in enumerate(zip(wet, mod_sig)):
        torchaudio.save(f"../out/wet_{idx}.wav", w, sr)
        plt.plot(m)
        plt.show()
        spec = tr.log(spectrogram(w[0]))
        plt.imshow(spec, aspect="auto", interpolation="none")
        plt.show()

    # tremolo_ds = PhaserGeneratorDataset(
    #     "../data/pads/test",
    #     2 * 44100,
    #     "../configs/ranges.json"
    # )
    #
    # merp = tremolo_ds[0]
    # ayy = 1
