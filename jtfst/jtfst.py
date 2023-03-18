import logging
import os
from typing import Optional, List, Tuple

import torch as tr
from torch import Tensor as T, nn

from dwt import average_td
from scattering_1d import ScatTransform1D, ScatTransform1DJagged
from scattering_2d import ScatTransform2DSubsampling

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


# TODO(cm): figure out why reflect_f has no effect
class JTFST1D(nn.Module):
    def __init__(self,
                 sr: float,
                 J_1: int = 12,
                 J_2_f: int = 4,
                 J_2_t: int = 12,
                 Q_1: int = 16,
                 Q_2_f: int = 1,
                 Q_2_t: int = 1,
                 should_avg_f: bool = True,
                 should_avg_t: bool = True,
                 avg_win_f: Optional[int] = None,
                 avg_win_t: Optional[int] = None,
                 reflect_f: bool = False) -> None:
        super().__init__()
        self.should_avg_f = should_avg_f
        self.should_avg_t = should_avg_t
        self.avg_win_f = avg_win_f
        self.avg_win_t = avg_win_t
        self.audio_to_scalogram = ScatTransform1D(sr, J_1, Q_1, should_avg=False, squeeze_channels=True)
        self.conv_t = ScatTransform1DJagged(sr, J_2_t, Q_2_t, should_avg=False, should_pad=True)
        self.conv_f = ScatTransform1D(sr,
                                      J_2_f,
                                      Q_2_f,
                                      should_avg=should_avg_f,
                                      avg_win=avg_win_f,
                                      squeeze_channels=False,
                                      reflect_t=reflect_f)

    def forward(self, x: T) -> (T, List[float], T, List[Tuple[float, float, int]]):
        scalogram, freqs_1, _ = self.audio_to_scalogram(x)
        y_t, freqs_t, _ = self.conv_t(scalogram, freqs_1)

        jtfst_s = []
        freqs_2 = []
        for y, freq_t in zip(y_t, freqs_t):
            y = tr.swapaxes(y, 1, 2)
            jtfst, freqs_f, orientations = self.conv_f(y)
            jtfst = tr.swapaxes(jtfst, 2, 3)
            jtfst_s.append(jtfst)
            for freq_f, orientation in zip(freqs_f, orientations):
                freqs_2.append((freq_f, freq_t, orientation))

        jtfst = tr.cat(jtfst_s, dim=1)
        if self.should_avg_t:
            avg_win_t = self.avg_win_t
            if avg_win_t is None:
                lowest_freq_t = freqs_t[-1]
                assert self.sr % lowest_freq_t == 0
                avg_win_t = int(self.sr / lowest_freq_t)
                log.info(f"defaulting avg_win_t to {avg_win_t} samples ({lowest_freq_t:.2f} Hz at {self.sr:.0f} SR)")

            jtfst = average_td(jtfst, avg_win_t, dim=-1)

        return scalogram, freqs_1, jtfst, freqs_2


class JTFST2D(nn.Module):
    def __init__(self,
                 sr: float,
                 J_1: int = 12,
                 J_2_f: int = 4,
                 J_2_t: int = 12,
                 Q_1: int = 16,
                 Q_2_f: int = 1,
                 Q_2_t: int = 1,
                 should_avg_f: bool = True,
                 should_avg_t: bool = True,
                 avg_win_f: Optional[int] = None,
                 avg_win_t: Optional[int] = None,
                 reflect_f: bool = True) -> None:
        super().__init__()
        self.should_avg_f = should_avg_f
        self.should_avg_t = should_avg_t
        self.avg_win_f = avg_win_f
        self.avg_win_t = avg_win_t
        self.audio_to_scalogram = ScatTransform1D(sr, J_1, Q_1, should_avg=False, squeeze_channels=True)
        self.scalogram_to_jtfst = ScatTransform2DSubsampling(sr,
                                                             J_2_f,
                                                             J_2_t,
                                                             Q_2_f,
                                                             Q_2_t,
                                                             should_avg_f,
                                                             should_avg_t,
                                                             avg_win_f,
                                                             avg_win_t,
                                                             reflect_f)

    def forward(self, x: T) -> (T, List[float], T, List[Tuple[float, float, int]]):
        scalogram, freqs_1, _ = self.audio_to_scalogram(x)
        jtfst, freqs_2 = self.scalogram_to_jtfst(scalogram)
        return scalogram, freqs_1, jtfst, freqs_2
