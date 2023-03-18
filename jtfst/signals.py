import logging
import math
import os

import torch as tr
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def make_pulse(n_samples: int, center_loc: float = 0.5, dur_samples: int = 4, amp: float = 1.0) -> T:
    with tr.no_grad():
        center_idx = int(n_samples * center_loc)
        start_idx = center_idx - (dur_samples // 2)
        end_idx = center_idx + (dur_samples // 2)
        assert start_idx >= 0
        assert end_idx < n_samples
        y = tr.zeros((n_samples,))
        y[start_idx:end_idx] = amp
        return y


def make_pure_sine(n_samples: int, sr: float, freq: float, amp: float = 1.0) -> T:
    assert freq <= sr / 2.0
    with tr.no_grad():
        dt = 1.0 / sr
        x = tr.arange(n_samples) * dt
        y = amp * tr.sin(2 * tr.pi * freq * x)
        return y


def make_exp_chirp(n_samples: int,
                   sr: float,
                   start_freq: float = 1.0,
                   end_freq: float = 20000.0,
                   amp: float = 1.0) -> T:
    assert 1.0 <= start_freq <= sr / 2.0
    assert 1.0 <= end_freq <= sr / 2.0
    with tr.no_grad():
        if start_freq == end_freq:
            return make_pure_sine(n_samples, sr, start_freq, amp)

        # TODO(cm): figure out what exactly is happening here
        dt = 1.0 / sr
        x = tr.arange(n_samples) * dt
        x_end = x[-1]
        k = x_end / math.log(end_freq / start_freq)
        phase = 2 * tr.pi * k * start_freq * (tr.pow(end_freq / start_freq, x / x_end) - 1.0)
        y = amp * tr.sin(phase)
        return y


def make_hann_window(n_samples: int) -> T:
    with tr.no_grad():
        x = tr.arange(n_samples)
        y = tr.sin(tr.pi * x / n_samples) ** 2
        return y
