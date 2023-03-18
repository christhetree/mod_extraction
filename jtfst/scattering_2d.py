import logging
import os
from typing import Optional, List, Tuple, Union

import torch as tr
from torch import Tensor as T, nn

from jtfst.dwt import dwt_2d, average_td
from jtfst.filterbanks import make_wavelet_bank
from jtfst.wavelets import MorletWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ScatTransform2D(nn.Module):
    def __init__(self,
                 sr: float,
                 J_f: int,
                 J_t: int,
                 Q_f: int = 1,
                 Q_t: int = 1,
                 should_avg_f: bool = False,
                 should_avg_t: bool = True,
                 avg_win_f: Optional[int] = None,
                 avg_win_t: Optional[int] = None,
                 highest_freq_f: Optional[float] = None,
                 highest_freq_t: Optional[float] = None,
                 reflect_f: bool = True) -> None:
        super().__init__()
        self.sr = sr
        self.J_f = J_f
        self.J_t = J_t
        self.Q_f = Q_f
        self.Q_t = Q_t
        self.should_avg_f = should_avg_f
        self.should_avg_t = should_avg_t
        self.avg_win_f = avg_win_f
        self.avg_win_t = avg_win_t
        self.highest_freq_f = highest_freq_f
        self.highest_freq_t = highest_freq_t
        self.reflect_f = reflect_f

        mw = MorletWavelet(sr_t=sr)
        wavelet_bank, freqs_f, freqs_t, orientations = make_wavelet_bank(mw,
                                                                         J_t,
                                                                         Q_t,
                                                                         highest_freq_t,
                                                                         n_octaves_f=J_f,
                                                                         steps_per_octave_f=Q_f,
                                                                         highest_freq_f=highest_freq_f,
                                                                         reflect_f=reflect_f)
        self.wavelet_bank = nn.ParameterList(wavelet_bank)
        self.freqs = list(zip(freqs_f, freqs_t, orientations))

    def forward(self, x: T) -> (T, List[Tuple[float, float, int]]):
        with tr.no_grad():
            y = ScatTransform2D.calc_scat_transform_2d(x,
                                                       self.sr,
                                                       self.wavelet_bank,
                                                       self.freqs,
                                                       self.should_avg_f,
                                                       self.should_avg_t,
                                                       self.avg_win_f,
                                                       self.avg_win_t)
            assert y.size(1) == len(self.freqs)
            return y, self.freqs

    @staticmethod
    def calc_scat_transform_2d(x: T,
                               sr: float,
                               wavelet_bank: Union[List[T], nn.ParameterList],
                               freqs: List[Tuple[float, float, int]],
                               should_avg_f: bool = False,
                               should_avg_t: bool = True,
                               avg_win_f: Optional[int] = None,
                               avg_win_t: Optional[int] = None) -> T:
        assert x.ndim == 3
        assert len(wavelet_bank) == len(freqs)
        y = dwt_2d(x, wavelet_bank, take_modulus=True)

        if not should_avg_f and not should_avg_t:
            return y

        lowest_freq_f, lowest_freq_t, _ = freqs[-1]
        if should_avg_t:
            if avg_win_t is None:
                assert sr % lowest_freq_t == 0
                avg_win_t = int(sr / lowest_freq_t)
                log.info(f"defaulting avg_win_t to {avg_win_t} samples ({lowest_freq_t:.2f} Hz at {sr:.0f} SR)")

            max_wavelet_len_t = max([w.size(1) for w in wavelet_bank])
            if avg_win_t > (max_wavelet_len_t + 1) // 6:
                log.warning(
                    "Time averaging window is suspiciously large (probably greater than the lowest central freq)")
            y = average_td(y, avg_win_t, dim=-1)

        if should_avg_f:
            if avg_win_f is None:
                log.info(f"should_avg_f is True, but avg_win_f is None, using a heuristic value of 2")
                avg_win_f = 2

            log.info(f"avg_win_f = {avg_win_f}")
            max_wavelet_len_f = max([w.size(0) for w in wavelet_bank])
            if avg_win_f > (max_wavelet_len_f + 1) // 6:
                log.warning(
                    "Freq averaging window is suspiciously large (probably greater than the lowest central freq)")
            y = average_td(y, avg_win_f, dim=-2)

        return y


# TODO(cm): implement ability to set highest frequency
class ScatTransform2DSubsampling(nn.Module):
    def __init__(self,
                 sr: float,
                 J_f: int,
                 J_t: int,
                 Q_f: int = 1,
                 Q_t: int = 1,
                 should_avg_f: bool = False,
                 should_avg_t: bool = True,
                 avg_win_f: Optional[int] = None,
                 avg_win_t: Optional[int] = None,
                 reflect_f: bool = True) -> None:
        super().__init__()
        self.sr = sr
        self.J_f = J_f
        self.J_t = J_t
        self.Q_f = Q_f
        self.Q_t = Q_t
        self.should_avg_f = should_avg_f
        self.should_avg_t = should_avg_t
        self.avg_win_t = avg_win_t
        self.reflect_f = reflect_f

        if should_avg_f:
            if avg_win_f is None:
                log.info(f"should_avg_f is True, but avg_win_f is None, using a heuristic value of 2")
                avg_win_f = 2
        self.avg_win_f = avg_win_f

        curr_sr_t = sr
        curr_highest_freq_t = sr / 2

        if should_avg_t:
            if avg_win_t is None:
                curr_avg_win_t = 2 ** J_t  # TODO(cm): check
                log.info(f"defaulting avg_win_t to {curr_avg_win_t} samples "
                         f"({sr / curr_avg_win_t:.2f} Hz at {sr:.0f} SR)")
            else:
                curr_avg_win_t = avg_win_t
        else:
            curr_avg_win_t = 1

        wavelet_banks = []
        avg_wins_t = []
        freqs_all = []
        for curr_j_t in range(J_t):
            if curr_j_t == J_t - 1:
                include_lowest_octave_t = True
            else:
                include_lowest_octave_t = False

            mw = MorletWavelet(sr_t=curr_sr_t, sr_f=sr)
            wavelet_bank, freqs_f, freqs_t, orientations = make_wavelet_bank(
                mw,
                n_octaves_t=1,
                steps_per_octave_t=Q_t,
                highest_freq_t=curr_highest_freq_t,
                include_lowest_octave_t=include_lowest_octave_t,
                n_octaves_f=J_f,
                steps_per_octave_f=Q_f,
                reflect_f=reflect_f
            )
            wavelet_bank = nn.ParameterList(wavelet_bank)
            wavelet_banks.append(wavelet_bank)
            avg_wins_t.append(curr_avg_win_t)
            freqs = list(zip(freqs_f, freqs_t, orientations))
            freqs_all.extend(freqs)
            if curr_avg_win_t > 1:
                curr_sr_t /= 2
                assert curr_avg_win_t % 2 == 0
                curr_avg_win_t //= 2
            curr_highest_freq_t /= 2

        self.wavelet_banks = nn.ParameterList(wavelet_banks)
        self.avg_wins_t = avg_wins_t
        self.freqs = freqs_all

    def forward(self, x: T) -> (T, List[Tuple[float, float, int]]):
        with tr.no_grad():
            octaves = []
            for wavelet_bank, avg_win_t in zip(self.wavelet_banks, self.avg_wins_t):
                octave = dwt_2d(x, wavelet_bank, take_modulus=True)
                octave = average_td(octave, avg_win_t, dim=-1)
                if self.should_avg_f:
                    octave = average_td(octave, self.avg_win_f, dim=-2)
                octaves.append(octave)
                if avg_win_t > 1:
                    x = average_td(x, avg_win=2, dim=-1)
            y = tr.cat(octaves, dim=1)
            assert y.size(1) == len(self.freqs)
            return y, self.freqs
