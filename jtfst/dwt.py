import logging
import os
from typing import Optional, List, Union

import torch as tr
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import ParameterList
from tqdm import tqdm

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def dwt_1d_td(x: T, wavelet_bank: Union[List[T], ParameterList], take_modulus: bool = True) -> T:
    assert x.ndim == 3
    assert wavelet_bank
    n_ch = x.size(1)
    assert n_ch == 1  # TODO(cm): slight modification required for multiple channels (adds them currently)

    x_complex = x.to(wavelet_bank[0].dtype)
    convs = []
    for wavelet in tqdm(wavelet_bank):
        assert wavelet.ndim == 1
        kernel = wavelet.view(1, 1, -1).repeat(1, n_ch, 1)
        out = F.conv1d(x_complex, kernel, stride=(1,), padding="same")
        convs.append(out)

    y = tr.cat(convs, dim=1)
    if take_modulus:
        y = tr.abs(y)

    return y


def dwt_1d(x: T,
           wavelet_bank: Union[List[T], ParameterList],
           take_modulus: bool = True,
           squeeze_channels: bool = True) -> T:
    assert x.ndim == 3
    n_ch = x.size(1)

    max_wavelet_len = max([len(w) for w in wavelet_bank])
    max_padding = max_wavelet_len // 2
    # TODO(cm): check why we can get away with only padding the front
    x = F.pad(x, (max_padding, 0))
    x_fd = tr.fft.fft(x).unsqueeze(1)

    kernels = []
    for wavelet in wavelet_bank:
        assert wavelet.ndim == 1
        left_padding = max_padding - wavelet.size(-1) // 2
        right_padding = x_fd.size(-1) - wavelet.size(-1) - left_padding
        kernel = wavelet.view(1, 1, -1).expand(-1, n_ch, -1)
        kernel = F.pad(kernel, (left_padding, right_padding))
        kernels.append(kernel)

    kernels = tr.cat(kernels, dim=0).unsqueeze(0)
    kernels_fd = tr.fft.fft(kernels)
    kernels_fd.imag *= -1  # PyTorch does cross-correlation instead of convolution
    y_fd = kernels_fd * x_fd
    y = tr.fft.ifft(y_fd)
    # TODO(cm): check why removing padding from the end works empirically after IFFT
    y = y[:, :, :, :-max_padding]
    if squeeze_channels:
        y = y.squeeze(-2)
    if take_modulus:
        y = tr.abs(y)
    return y


def dwt_2d_td(x: T, wavelet_bank: Union[List[T], ParameterList], take_modulus: bool = True) -> T:
    assert x.ndim == 3
    assert wavelet_bank

    x = x.unsqueeze(1)  # Image with 1 channel
    scalogram_complex = x.to(wavelet_bank[0].dtype)
    convs = []
    for wavelet in tqdm(wavelet_bank):
        assert wavelet.ndim == 2
        kernel = wavelet.view(1, 1, *wavelet.shape)
        out = F.conv2d(scalogram_complex, kernel, stride=(1, 1), padding="same")
        convs.append(out)

    y = tr.cat(convs, dim=1)
    if take_modulus:
        y = tr.abs(y)

    return y


def dwt_2d(x: T,
           wavelet_bank: Union[List[T], ParameterList],
           max_f_dim: Optional[int] = None,
           max_t_dim: Optional[int] = None,
           take_modulus: bool = True) -> T:
    assert x.ndim == 3
    if max_f_dim is None:
        max_f_dim = max([w.size(0) for w in wavelet_bank])
    if max_t_dim is None:
        max_t_dim = max([w.size(1) for w in wavelet_bank])
    max_f_padding = max_f_dim // 2
    max_t_padding = max_t_dim // 2
    # TODO(cm): check why we can get away with only padding the front
    x = F.pad(x, (max_t_padding, 0, max_f_padding, 0))
    log.debug("scalogram fft")
    x_fd = tr.fft.fft2(x).unsqueeze(1)

    log.debug("making kernels")
    kernels = []
    for wavelet in wavelet_bank:
        assert wavelet.ndim == 2
        top_padding = max_f_padding - wavelet.size(-2) // 2
        bottom_padding = x_fd.size(-2) - wavelet.size(-2) - top_padding
        left_padding = max_t_padding - wavelet.size(-1) // 2
        right_padding = x_fd.size(-1) - wavelet.size(-1) - left_padding
        kernel = wavelet.view(1, 1, *wavelet.shape)
        kernel = F.pad(kernel, (left_padding, right_padding, top_padding, bottom_padding))
        kernels.append(kernel)

    kernels = tr.cat(kernels, dim=1)
    log.debug("kernel fft")
    kernels_fd = tr.fft.fft2(kernels)
    log.debug("matmult")
    kernels_fd.imag *= -1  # PyTorch does cross-correlation instead of convolution
    y_fd = kernels_fd * x_fd
    log.debug("y ifft")
    y = tr.fft.ifft2(y_fd)
    # TODO(cm): check why removing padding from the end works empirically after IFFT
    y = y[:, :, :-max_f_padding, :-max_t_padding]
    if take_modulus:
        y = tr.abs(y)
    return y


def average_td(x: T, avg_win: int, dim: int = -1, hop_size: Optional[int] = None) -> T:
    assert x.ndim >= 2
    assert avg_win >= 1
    assert x.size(dim) >= avg_win

    if hop_size is None:
        hop_size = avg_win

    # TODO(cm): add padding for last frame
    unfolded = x.unfold(dimension=dim, size=avg_win, step=hop_size)
    out = tr.mean(unfolded, dim=-1, keepdim=False)
    return out
