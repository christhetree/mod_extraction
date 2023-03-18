"""
Methods for creating discrete wavelet filterbanks.
"""

import logging
import os
from typing import Optional, List

from torch import Tensor as T

from jtfst.wavelets import DiscreteWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def calc_scales_and_freqs(n_octaves: int,
                          steps_per_octave: int,
                          sr: float,
                          dw: DiscreteWavelet,
                          highest_freq: Optional[float] = None,
                          include_lowest_octave: bool = True) -> (List[float], List[float]):
    """
    Calculates the scales and corresponding frequencies for a wavelet filterbank.

    Args:
        :param n_octaves: How many octaves to include in the filterbank.
        :param steps_per_octave: How many steps per octave to include in the filterbank.
        :param sr: Sampling rate of the wavelets in the filterbank.
        :param dw: Discrete wavelet class to use for the filterbank.
        :param highest_freq: Starting frequency of the filterbank. If omitted, defaults to the Nyquist frequency.
        :param include_lowest_octave: For completeness, include the lowest octave or not
                                      (e.g. if true, results in (n_octaves * steps_per_octave) + 1 number of filters).
    """
    assert n_octaves >= 1
    assert steps_per_octave >= 1

    if highest_freq is None:
        smallest_period = 2.0 / sr  # Default to Nyquist
    else:
        # Ensure highest frequency is not higher than Nyquist
        smallest_period = 1.0 / highest_freq
        assert smallest_period * sr >= 2.0

    scales = []
    periods = []

    for j in range(n_octaves):
        for q in range(steps_per_octave):
            exp = j + (q / steps_per_octave)
            curr_period = smallest_period * (2 ** exp)
            s = dw.period_to_scale(curr_period)
            scales.append(s)
            periods.append(curr_period)

        if include_lowest_octave and j == n_octaves - 1:
            curr_period = smallest_period * (2 ** (j + 1))
            s = dw.period_to_scale(curr_period)
            scales.append(s)
            periods.append(curr_period)

    freqs = [1.0 / p for p in periods]
    return scales, freqs


def make_wavelet_bank(dw: DiscreteWavelet,
                      n_octaves_t: int,
                      steps_per_octave_t: int = 1,
                      highest_freq_t: Optional[float] = None,
                      include_lowest_octave_t: bool = True,
                      reflect_t: bool = False,
                      n_octaves_f: Optional[int] = None,
                      steps_per_octave_f: int = 1,
                      highest_freq_f: Optional[float] = None,
                      include_lowest_octave_f: bool = True,
                      reflect_f: bool = True,
                      normalize: bool = True) -> (List[T], List[float], List[float], List[int]):
    """
    Creates a 1D or 2D wavelet filterbank along with each wavelet's corresponding frequency axis frequency, time axis
    frequency, and orientation (if 1D for the time axis and if 2D for the frequency axis).

    For easier to read notation and consistency with scalograms and convolution outputs, we always define the last
    dimension (dim=-1) as the time dimension "t" and the second last dimension (dim=-2) as the frequency dimension "f".

    We consider 1D wavelet filterbanks as operating on the time axis and 2D wavelet filterbanks as operating on both.

    Args:
        :param dw: Discrete wavelet class to use for the filterbank.
        :param n_octaves_t: How many octaves to include in the filterbank for the time axis.
        :param steps_per_octave_t: How many steps per octave to include in the filterbank for the time axis.
        :param highest_freq_t: Starting frequency of the filterbank for the time axis. If omitted, defaults to the
                               Nyquist frequency.
        :param include_lowest_octave_t: For completeness, includes the lowest octave or not for the time axis.
        :param reflect_t: Reflects each wavelet along the time axis (results in twice as many filters).
        :param n_octaves_f: How many octaves to include in the filterbank for the frequency axis.
        :param steps_per_octave_f: How many octaves to include in the filterbank for the frequency axis.
        :param highest_freq_f: Starting frequency of the filterbank for the frequency axis. If omitted, defaults to the
                               Nyquist frequency.
        :param include_lowest_octave_f: For completeness, includes the lowest octave or not for the frequency axis.
        :param reflect_f: Reflects each wavelet along the frequency axis (results in twice as many filters).
        :param normalize: Normalizes the wavelets to unit energy.
    """
    if n_octaves_f is not None:
        scales_f, freqs_f = calc_scales_and_freqs(
            n_octaves_f, steps_per_octave_f, dw.sr_f, dw, highest_freq_f, include_lowest_octave_f)
        log.debug(f"freqs_f highest = {freqs_f[0]:.2f}")
        log.debug(f"freqs_f lowest  = {freqs_f[-1]:.2f}")
    else:
        scales_f = None
        freqs_f = None

    scales_t, freqs_t = calc_scales_and_freqs(
        n_octaves_t, steps_per_octave_t, dw.sr_t, dw, highest_freq_t, include_lowest_octave_t)
    log.debug(f"freqs_t highest = {freqs_t[0]:.2f}")
    log.debug(f"freqs_t lowest  = {freqs_t[-1]:.2f}")

    wavelet_bank = []
    freqs_f_out = []
    freqs_t_out = []
    orientations = []
    if scales_f:
        for s_t, freq_t in zip(scales_t, freqs_t):
            for s_f, freq_f in zip(scales_f, freqs_f):
                wavelet = dw.create_2d_wavelet_from_scale(s_f, s_t, reflect=False, normalize=normalize)
                wavelet_bank.append(wavelet)
                freqs_f_out.append(freq_f)
                freqs_t_out.append(freq_t)
                orientations.append(1)
                if reflect_f:
                    wavelet_reflected = dw.create_2d_wavelet_from_scale(s_f,
                                                                        s_t,
                                                                        reflect=True,
                                                                        normalize=normalize)
                    wavelet_bank.append(wavelet_reflected)
                    freqs_f_out.append(freq_f)
                    freqs_t_out.append(freq_t)
                    orientations.append(-1)
    else:
        for s_t, freq_t in zip(scales_t, freqs_t):
            wavelet = dw.create_1d_wavelet_from_scale(s_t, reflect=False, normalize=normalize)
            wavelet_bank.append(wavelet)
            freqs_t_out.append(freq_t)
            orientations.append(1)
            if reflect_t:
                wavelet_reflected = dw.create_1d_wavelet_from_scale(s_t, reflect=True, normalize=normalize)
                wavelet_bank.append(wavelet_reflected)
                freqs_t_out.append(freq_t)
                orientations.append(-1)

    return wavelet_bank, freqs_f_out, freqs_t_out, orientations
