import logging
import os

import torch as tr
from torch import Tensor as T, nn

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
    assert shape in {"cos", "rect_cos", "inv_rect_cos", "tri", "saw", "rsaw", "sqr"}
    if shape in {"rect_cos", "inv_rect_cos"}:
        # Rectified sine waves have double the frequency
        freq /= 2.0
    argument = tr.cumsum(2 * tr.pi * tr.full((n_samples,), freq) / sr, dim=0) + phase

    if shape == "cos":
        return (tr.cos(argument + tr.pi) + 1.0) / 2.0
    if shape == "rect_cos":
        return tr.abs(tr.cos(argument + tr.pi))
    if shape == "inv_rect_cos":
        return -tr.abs(tr.cos(argument + tr.pi)) + 1.0
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
