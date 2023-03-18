"""
Wavelet base class and Morlet wavelet implementation.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import torch as tr
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class DiscreteWavelet(ABC):
    """
    Abstract base class for defining a discrete wavelet to be used with scattering transforms.

    For easier to read notation and consistency with scalograms and convolution outputs, we always define the last
    dimension (dim=-1) as the time dimension "t" and (in this case if the wavelet is 2-dimensional) the second last
    dimension (dim=-2) as the frequency dimension "f".

    Args:
        :param sr_t: Sampling rate of the time dimension of the wavelet.
        :param sr_f: Sampling rate of the frequency dimension of the wavelet (only needs to be specified if different).
    """
    def __init__(self,
                 sr_t: float = 44100,
                 sr_f: Optional[float] = None) -> None:
        self.sr_t = sr_t
        if sr_f is None:
            self.sr_f = sr_t
        else:
            self.sr_f = sr_f

        self.dt = 1.0 / self.sr_t
        self.df = 1.0 / self.sr_f

        nyquist_t = self.sr_t / 2.0
        self.min_scale_t = self.freq_to_scale(nyquist_t)
        nyquist_f = self.sr_f / 2.0
        self.min_scale_f = self.freq_to_scale(nyquist_f)

    @abstractmethod
    def period_to_scale(self, period: float) -> float:
        """Given a period in seconds, returns the corresponding wavelet scale."""
        pass

    @abstractmethod
    def make_t_from_scale(self, s: float, dt: float) -> T:
        """
        Given a wavelet scale and the duration of one sample, returns a 1D tensor of timesteps representing the x axis
        for a 1D wavelet. This is usually symmetrical around 0 and will be fed into the wavelet function to calculate
        the corresponding y values.
        """
        pass

    @abstractmethod
    def y_1d(self, t: T, s: float) -> T:
        """
        Given a wavelet scale and 1D tensor of timesteps representing the x axis for a 1D wavelet, returns a 1D tensor
        of the corresponding y values.
        """
        pass

    def freq_to_scale(self, freq: float) -> float:
        """Given a frequency in Hz, returns the corresponding wavelet scale."""
        return self.period_to_scale(1.0 / freq)

    def normalize_to_unit_energy(self, wavelet: T) -> T:
        """Normalizes a discrete wavelet to unit energy."""
        energy = DiscreteWavelet.calc_energy(wavelet)
        wavelet *= energy ** -0.5
        wavelet *= wavelet.size(-1) ** -0.5
        if wavelet.ndim == 2:
            wavelet *= wavelet.size(-2) ** -0.5
        return wavelet

    def y_2d(self, t_1: T, t_2: T, s_1: float = 1.0, s_2: float = 1.0) -> T:
        """Given 2 wavelet scales and 2 x-axis, creates a 2D wavelet by taking the outer product of 2 1D wavelets."""
        assert t_1.ndim == 1
        assert t_2.ndim == 1
        assert t_1.size(0) * t_2.size(0) <= 2 ** 26  # Larger than this is too big for reasonable compute times
        y_1 = self.normalize_to_unit_energy(self.y_1d(t_1, s_1))
        y_2 = self.normalize_to_unit_energy(self.y_1d(t_2, s_2))
        y = tr.outer(y_1, y_2)
        return y

    def create_1d_wavelet_from_scale(self, s_t: float = 1.0, reflect: bool = False, normalize: bool = True) -> T:
        """
        Creates a 1D wavelet for the time dimension.

        Args:
            :param s_t: Wavelet scale.
            :param reflect: Reflects the wavelet across the y axis.
            :param normalize: Normalizes the wavelet to unit energy.
        """
        with tr.no_grad():
            assert s_t >= self.min_scale_t
            t = self.make_t_from_scale(s_t, self.dt)
            if reflect:
                t = -t
            wavelet = self.y_1d(t, s_t)
            if normalize:
                wavelet = self.normalize_to_unit_energy(wavelet)
            return wavelet

    def create_2d_wavelet_from_scale(self,
                                     s_f: float = 1.0,
                                     s_t: float = 1.0,
                                     reflect: bool = False,
                                     normalize: bool = True) -> T:
        """
        Creates a 2D wavelet for the frequency and time dimension.

        Args:
            :param s_f: Wavelet scale for the frequency dimension.
            :param s_t: Wavelet scale for the time dimension.
            :param reflect: Reflects the wavelet across the y axis.
            :param normalize: Normalizes the wavelet to unit energy.
        """
        with tr.no_grad():
            assert s_f >= self.min_scale_f
            assert s_t >= self.min_scale_t
            t_f = self.make_t_from_scale(s_f, self.df)
            t_t = self.make_t_from_scale(s_t, self.dt)
            if reflect:
                t_t = -t_t
            wavelet = self.y_2d(t_f, t_t, s_f, s_t)
            if normalize:
                wavelet = self.normalize_to_unit_energy(wavelet)
            return wavelet

    @staticmethod
    def calc_energy(signal: T) -> float:
        """Calculates the energy of a tensor."""
        return tr.sum(tr.abs(signal) ** 2).item()


class MorletWavelet(DiscreteWavelet):
    """
    Creates a discrete Morlet wavelet.
    More information about the underlying math and hyperparameters can be found here:
    https://psl.noaa.gov/people/gilbert.p.compo/Torrence_compo1998.pdf
    https://www.biorxiv.org/content/10.1101/397182v1.full.pdf

    Args:
        :param sr_t: Sampling rate of the time dimension of the wavelet.
        :param sr_f: Sampling rate of the frequency dimension of the wavelet (only needs to be specified if different).
        :param n_sig: Number of standard deviations to use for the gaussian envelope. Contains >= 99.7% of the wavelet
                      if >= 3.0
        :param w: Morlet wavelet hyperparameter, usually set to be > 5. If not provided it is automatically calculated
                  such that scale is equal to the fourier period (e.g. scale of 0.5 corresponds to 2 Hz)
    """
    def __init__(self,
                 sr_t: float = 44100,
                 sr_f: Optional[float] = None,
                 n_sig: float = 3.0,
                 w: Optional[float] = None):
        if w is None:
            w = MorletWavelet.freq_to_w_at_s(freq=1.0, s=1.0)  # For convenience let scale and fourier period be equal
        self.w = tr.tensor(w)
        self.n_sig = n_sig  # Contains >= 99.7% of the wavelet if >= 3.0

        # These are pre-calculated constants
        self.a = tr.tensor(tr.pi ** -0.25)
        self.b = tr.exp(-0.5 * (self.w ** 2))
        super().__init__(sr_t, sr_f)

    def period_to_scale(self, period: float) -> float:
        return period * (self.w + ((2.0 + (self.w ** 2)) ** 0.5)) / (4 * tr.pi)

    def make_t_from_scale(self, s: float, dt: float) -> T:
        M = int((self.n_sig * s) / dt)
        t = tr.arange(-M, M + 1) * dt
        return t

    def y_1d(self, t: T, s: float = 1.0) -> T:
        assert t.ndim == 1
        x = t / s
        y = self.a * (tr.exp(1j * self.w * x) - self.b) * tr.exp(-0.5 * (x ** 2))
        return y

    def scale_to_period(self, s: float) -> float:
        """Convenience method rearranged for period."""
        return (4 * tr.pi * s) / (self.w + ((2.0 + (self.w ** 2)) ** 0.5))

    def scale_to_freq(self, s: float) -> float:
        """Convenience method rearranged for frequency."""
        period = self.scale_to_period(s)
        return 1.0 / period

    @staticmethod
    def period_to_w_at_s(period: float, s: float = 1.0) -> float:
        """
        Calculates the Morlet wavelet hyperparameter w such that it corresponds to a desired fourier period and scale.
        For convenience it makes sense to set w such that a fourier period of 1 corresponds to a scale of 1.
        """
        return (((4 * tr.pi * s) ** 2) - (2 * (period ** 2))) / (8 * tr.pi * period * s)

    @staticmethod
    def freq_to_w_at_s(freq: float, s: float = 1.0) -> float:
        """Convenience method rearranged for frequency."""
        return MorletWavelet.period_to_w_at_s(1.0 / freq, s)
