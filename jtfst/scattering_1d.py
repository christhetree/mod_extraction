import logging
import os
from typing import Optional, List, Union

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T, nn

from jtfst.dwt import average_td, dwt_1d
from jtfst.filterbanks import make_wavelet_bank
from jtfst.wavelets import MorletWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ScatTransform1D(nn.Module):
    def __init__(self,
                 sr: float,
                 J: int = 12,
                 Q: int = 12,
                 should_avg: bool = False,
                 avg_win: Optional[int] = None,
                 highest_freq: Optional[float] = None,
                 squeeze_channels: bool = True,
                 reflect_t: bool = False) -> None:
        super().__init__()
        self.sr = sr
        self.J = J
        self.Q = Q
        self.should_avg = should_avg
        self.avg_win = avg_win
        self.highest_freq = highest_freq
        self.squeeze_channels = squeeze_channels
        self.reflect_t = reflect_t

        mw = MorletWavelet(sr_t=sr)
        wavelet_bank, _, freqs_t, orientations = make_wavelet_bank(mw, J, Q, highest_freq, reflect_t=reflect_t)
        self.wavelet_bank = nn.ParameterList(wavelet_bank)
        self.freqs_t = freqs_t
        self.orientations = orientations

    def forward(self, x: T) -> (T, List[float], List[int]):
        with tr.no_grad():
            y = ScatTransform1D.calc_scat_transform_1d(x,
                                                       self.sr,
                                                       self.wavelet_bank,
                                                       self.freqs_t,
                                                       self.should_avg,
                                                       self.avg_win,
                                                       self.squeeze_channels)
            assert y.size(1) == len(self.freqs_t) == len(self.orientations)
            return y, self.freqs_t, self.orientations

    @staticmethod
    def calc_scat_transform_1d(x: T,
                               sr: float,
                               wavelet_bank: Union[List[T], nn.ParameterList],
                               freqs: List[float],
                               should_avg: bool = False,
                               avg_win: Optional[int] = None,
                               squeeze_channels: bool = True) -> T:
        assert x.ndim == 3
        assert len(wavelet_bank) == len(freqs)
        y = dwt_1d(x, wavelet_bank, take_modulus=True, squeeze_channels=squeeze_channels)

        if not should_avg:
            return y

        if avg_win is None:
            lowest_freq = freqs[-1]
            assert sr % lowest_freq == 0
            avg_win = int(sr / lowest_freq)
            log.info(f"defaulting avg_win to {avg_win} samples ({lowest_freq:.2f} Hz at {sr:.0f} SR)")

        y = average_td(y, avg_win, dim=-1)
        return y


class ScatTransform1DJagged(nn.Module):
    def __init__(self,
                 sr: float,
                 J: int = 12,
                 Q: int = 12,
                 should_avg: bool = False,
                 avg_win: Optional[int] = None,
                 highest_freq: Optional[float] = None,
                 reflect_t: bool = False,
                 should_pad: bool = False) -> None:
        super().__init__()
        self.sr = sr
        self.J = J
        self.Q = Q
        self.should_avg = should_avg
        if should_avg:
            assert avg_win is not None, "ScatTransform1DJagged cannot infer the averaging window automatically"
        self.avg_win = avg_win
        self.highest_freq = highest_freq
        self.reflect_t = reflect_t
        self.should_pad = should_pad

        mw = MorletWavelet(sr_t=sr)
        wavelet_bank, _, freqs_t, orientations = make_wavelet_bank(mw, J, Q, highest_freq, reflect_t=reflect_t)
        self.wavelet_bank = nn.ParameterList(wavelet_bank)
        self.freqs_t = freqs_t
        self.orientations = orientations

    def forward(self, x: T, freqs_x: List[float]) -> (List[T], List[float], List[int]):
        with tr.no_grad():
            assert x.ndim == 3
            assert x.size(1) == len(freqs_x)
            y_s = []
            freqs_t = []
            orientations = []
            for wavelet, freq_t, orientation in zip(self.wavelet_bank, self.freqs_t, self.orientations):
                # TODO(cm): check what condition is correct
                band_freqs = [f_x for f_x in freqs_x if f_x >= freq_t]
                n_bands = len(band_freqs)
                if n_bands == 0:
                    continue
                curr_x = x[:, :n_bands, :]
                curr_wb = [wavelet]
                curr_freqs = [freq_t]
                y = ScatTransform1D.calc_scat_transform_1d(curr_x,
                                                           self.sr,
                                                           curr_wb,
                                                           curr_freqs,
                                                           self.should_avg,
                                                           self.avg_win,
                                                           squeeze_channels=False)
                if self.should_pad:
                    pad_n = x.size(1) - n_bands
                    y = F.pad(y, (0, 0, 0, pad_n))
                y = y.squeeze(1)
                y_s.append(y)
                freqs_t.append(freq_t)
                orientations.append(orientation)
            assert len(y_s) == len(freqs_t) == len(orientations)
            return y_s, freqs_t, orientations


# TODO(cm): add support for avg_win
class ScatTransform1DSubsampling(nn.Module):
    def __init__(self,
                 sr: float,
                 J: int = 12,
                 Q: int = 12,
                 squeeze_channels: bool = True,
                 reflect_t: bool = False) -> None:
        super().__init__()
        self.sr = sr
        self.J = J
        self.Q = Q
        self.squeeze_channels = squeeze_channels
        self.reflect_t = reflect_t

        curr_sr = sr
        curr_avg_win = 2 ** J

        wavelet_banks = []
        avg_wins = []
        freqs_all = []
        orientations_all = []
        for curr_j in range(J):
            if curr_j == J - 1:
                include_lowest_octave = True
            else:
                include_lowest_octave = False
            mw = MorletWavelet(sr_t=curr_sr)
            wavelet_bank, _, freqs_t, orientations = make_wavelet_bank(mw,
                                                                       n_octaves_t=1,
                                                                       steps_per_octave_t=Q,
                                                                       include_lowest_octave_t=include_lowest_octave,
                                                                       reflect_t=reflect_t)
            wavelet_bank = nn.ParameterList(wavelet_bank)
            wavelet_banks.append(wavelet_bank)
            avg_wins.append(curr_avg_win)
            freqs_all.extend(freqs_t)
            orientations_all.extend(orientations)
            curr_sr /= 2
            curr_avg_win //= 2

        self.wavelet_banks = nn.ParameterList(wavelet_banks)
        self.avg_wins = avg_wins
        self.freqs_t = freqs_all
        self.orientations = orientations_all

    def forward(self, x: T) -> (T, List[float], List[int]):
        with tr.no_grad():
            octaves = []
            for wavelet_bank, avg_win in zip(self.wavelet_banks, self.avg_wins):
                octave = dwt_1d(x, wavelet_bank, take_modulus=True, squeeze_channels=self.squeeze_channels)
                octave = average_td(octave, avg_win)
                octaves.append(octave)
                x = average_td(x, avg_win=2, hop_size=2)
            y = tr.cat(octaves, dim=1)
            assert y.size(1) == len(self.freqs_t) == len(self.orientations)
            return y, self.freqs_t, self.orientations
