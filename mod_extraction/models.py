import logging
import math
import os
from typing import Optional, List, Tuple, Dict

import torch as tr
from torch import Tensor as T
from torch import nn
from torchaudio.transforms import Spectrogram, MelSpectrogram, FrequencyMasking, TimeMasking

from mod_extraction.modulations import make_rand_mod_signal
from mod_extraction.s4d import S4D, DropoutNd
from mod_extraction.tcn import TCN

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class RandomLFO(nn.Module):
    def __init__(self,
                 n_samples: int,
                 sr: float,
                 use_shape_gt: bool = False,
                 use_phase_gt: bool = False,
                 use_freq_gt: bool = False,
                 shapes: Optional[List[str]] = None,
                 freq_min: float = 0.5,
                 freq_max: float = 3.0,
                 phase_error: float = 0.0,
                 freq_error: float = 0.0) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.sr = sr
        self.use_shape_gt = use_shape_gt
        self.use_phase_gt = use_phase_gt
        self.use_freq_gt = use_freq_gt
        self.shapes = shapes
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.phase_error = phase_error
        self.freq_error = freq_error

    def forward(self, batch_size: int, fx_params: Optional[Dict[str, T]] = None) -> T:
        shapes_gt = None
        phase_gt = None
        freq_gt = None
        if self.use_shape_gt:
            assert fx_params is not None and "shape" in fx_params
            shapes_gt = fx_params["shape"]
        if self.use_phase_gt:
            assert fx_params is not None and "phase" in fx_params
            phase_gt = fx_params["phase"]
        if self.use_freq_gt:
            assert fx_params is not None and "rate_hz" in fx_params
            freq_gt = fx_params["rate_hz"]

        return make_rand_mod_signal(
            batch_size,
            self.n_samples,
            self.sr,
            self.freq_min,
            self.freq_max,
            shapes_gt,
            self.shapes,
            phase_gt,
            self.phase_error,
            freq_gt,
            self.freq_error,
        ).unsqueeze(1)


class SpectralTCN(nn.Module):
    def __init__(self,
                 n_samples: int = 88200,
                 n_fft: int = 1024,
                 hop_len: int = 256,
                 kernel_size: int = 13,
                 out_channels: Optional[List[int]] = None,
                 dilations: Optional[List[int]] = None,
                 latent_dim: int = 1,
                 use_ln: bool = True,
                 use_res: bool = True,
                 eps: float = 1e-7) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.use_ln = use_ln
        self.use_res = use_res
        self.eps = eps
        if out_channels is None:
            out_channels = [96] * 5
        self.out_channels = out_channels
        if dilations is None:
            dilations = [2 ** idx for idx in range(len(out_channels))]
        self.dilations = dilations

        self.spectrogram = Spectrogram(n_fft, hop_length=hop_len, normalized=False)
        in_ch = n_fft // 2 + 1
        n_frames = n_samples // hop_len + 1
        temporal_dims = [n_frames] * len(out_channels)

        self.tcn = TCN(out_channels,
                       dilations,
                       in_ch,
                       kernel_size,
                       padding=None,
                       use_ln=use_ln,
                       temporal_dims=temporal_dims,
                       use_res=use_res,
                       is_causal=False)
        self.receptive_field = self.tcn.calc_receptive_field()
        log.info(f"Receptive field = {self.receptive_field} samples")
        self.output = nn.Conv1d(out_channels[-1], self.latent_dim, kernel_size=(1,))

    def forward(self, x: T) -> T:
        assert x.ndim == 3
        x = self.spectrogram(x).squeeze(1)
        x = tr.clip(x, min=self.eps)
        x = tr.log(x)
        x = self.tcn(x)
        x = self.output(x)
        x = tr.sigmoid(x)
        return x


class Spectral2DCNN(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 n_samples: int = 88200,
                 sr: float = 44100,
                 n_fft: int = 1024,
                 hop_len: int = 256,
                 n_mels: int = 256,
                 kernel_size: Tuple[int, int] = (5, 13),
                 out_channels: Optional[List[int]] = None,
                 bin_dilations: Optional[List[int]] = None,
                 temp_dilations: Optional[List[int]] = None,
                 pool_size: Tuple[int, int] = (3, 1),
                 latent_dim: int = 1,
                 freq_mask_amount: float = 0.0,
                 time_mask_amount: float = 0.0,
                 use_ln: bool = True,
                 eps: float = 1e-7) -> None:
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.kernel_size = kernel_size
        assert pool_size[1] == 1
        self.pool_size = pool_size
        self.latent_dim = latent_dim
        self.freq_mask_amount = freq_mask_amount
        self.time_mask_amount = time_mask_amount
        self.use_ln = use_ln
        self.eps = eps
        if out_channels is None:
            out_channels = [64] * 5
        self.out_channels = out_channels
        if bin_dilations is None:
            bin_dilations = [1] * len(out_channels)
        self.bin_dilations = bin_dilations
        if temp_dilations is None:
            temp_dilations = [2 ** idx for idx in range(len(out_channels))]
        self.temp_dilations = temp_dilations
        assert len(out_channels) == len(bin_dilations) == len(temp_dilations)

        self.spectrogram = MelSpectrogram(sample_rate=int(sr),
                                          n_fft=n_fft,
                                          hop_length=hop_len,
                                          normalized=False,
                                          n_mels=n_mels,
                                          center=True)
        n_bins = n_mels
        n_frames = n_samples // hop_len + 1
        temporal_dims = [n_frames] * len(out_channels)

        self.freq_masking = FrequencyMasking(freq_mask_param=int(freq_mask_amount * n_bins))
        self.time_masking = TimeMasking(time_mask_param=int(time_mask_amount * n_frames))

        layers = []
        for out_ch, b_dil, t_dil, temp_dim in zip(out_channels, bin_dilations, temp_dilations, temporal_dims):
            if use_ln:
                layers.append(nn.LayerNorm([n_bins, temp_dim], elementwise_affine=False))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=(1, 1), dilation=(b_dil, t_dil), padding="same"))
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
            layers.append(nn.PReLU(num_parameters=out_ch))
            in_ch = out_ch
            n_bins = n_bins // pool_size[0]
        self.cnn = nn.Sequential(*layers)

        # TODO(cm): change from regression to classification
        self.output = nn.Conv1d(out_channels[-1], self.latent_dim, kernel_size=(1,))

    def forward(self, x: T) -> (T, T):
        assert x.ndim == 3
        x = self.spectrogram(x)

        if self.training:
            if self.freq_mask_amount > 0:
                x = self.freq_masking(x)
            if self.time_mask_amount > 0:
                x = self.time_masking(x)

        x = tr.clip(x, min=self.eps)
        x = tr.log(x)
        x = self.cnn(x)
        x = tr.mean(x, dim=-2)
        latent = x

        x = self.output(x)
        x = tr.sigmoid(x)
        return x, latent


class SpectralDSTCN(nn.Module):
    def __init__(self,
                 n_samples: int = 88200,
                 n_fft: int = 1024,
                 hop_len: int = 256,
                 kernel_size: int = 13,
                 out_channels: Optional[List[int]] = None,
                 dilations: Optional[List[int]] = None,
                 strides: Optional[List[int]] = None,
                 n_fc_units: int = 48,
                 latent_dim: int = 2,
                 use_ln: bool = True,
                 use_res: bool = True,
                 eps: float = 1e-7) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.kernel_size = kernel_size
        self.n_fc_units = n_fc_units
        self.latent_dim = latent_dim
        self.use_ln = use_ln
        self.use_res = use_res
        self.eps = eps

        if out_channels is None:
            out_channels = [96] * 5
        self.out_channels = out_channels
        if dilations is None:
            dilations = [2 ** idx for idx in range(len(out_channels))]
        self.dilations = dilations
        if strides is None:
            strides = [2] * len(out_channels)
        self.strides = strides

        self.spectrogram = Spectrogram(n_fft, hop_length=hop_len, normalized=False)
        in_ch = n_fft // 2 + 1

        n_frames = n_samples // hop_len + 1
        temporal_dims = [n_frames]
        curr_n_frames = n_frames
        for stride in strides[:-1]:
            curr_n_frames = math.ceil(curr_n_frames / stride)
            temporal_dims.append(curr_n_frames)

        self.tcn = TCN(out_channels,
                       dilations,
                       in_ch,
                       kernel_size,
                       strides,
                       padding=None,
                       use_ln=use_ln,
                       temporal_dims=temporal_dims,
                       use_res=use_res,
                       is_causal=False)
        self.fc = nn.Linear(out_channels[-1], self.n_fc_units)
        self.fc_act = nn.PReLU(self.n_fc_units)
        self.output = nn.Linear(self.n_fc_units, self.latent_dim)

    def forward(self, x: T) -> T:
        assert x.ndim == 3
        x = self.spectrogram(x).squeeze(1)
        x = tr.clip(x, min=self.eps)
        x = tr.log(x)

        x = self.tcn(x)
        x = tr.mean(x, dim=-1)

        x = self.fc(x)
        x = self.fc_act(x)
        x = self.output(x)
        x = tr.sigmoid(x)
        return x


class HiddenStateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden: Tuple[T, T] = (tr.zeros((1,)), tr.zeros((1,)))  # Must be initialized as a tensor for torchscript tracing
        self.is_hidden_init = False

    def update_hidden(self, hidden: Tuple[T, T]) -> None:
        self.hidden = hidden
        self.is_hidden_init = True

    def detach_hidden(self) -> None:
        if self.is_hidden_init:
            # TODO(cm): check whether clone is required or not
            self.hidden = tuple((h.detach().clone() for h in self.hidden))

    def clear_hidden(self) -> None:
        self.is_hidden_init = False


class LSTMEffectModel(HiddenStateModel):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
                 n_hidden: int = 64,
                 latent_dim: int = 1) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_hidden = n_hidden
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(in_ch + latent_dim, n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, out_ch)

    def forward(self, x: T, latent: T) -> T:
        assert x.ndim == 3
        assert latent.shape == (x.size(0), self.latent_dim, x.size(-1))
        lstm_in = tr.cat([latent, x], dim=1)
        lstm_in = tr.swapaxes(lstm_in, 1, 2)
        if self.is_hidden_init:
            lstm_out, new_hidden = self.lstm(lstm_in, self.hidden)
        else:
            lstm_out, new_hidden = self.lstm(lstm_in)
        fc_out = self.fc(lstm_out)
        fc_out = tr.swapaxes(fc_out, 1, 2)
        y_hat = fc_out + x
        y_hat = tr.tanh(y_hat)
        self.update_hidden(new_hidden)
        return y_hat


class S4DEffectModel(HiddenStateModel):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
                 n_hidden: int = 64,
                 latent_dim: int = 1,
                 n_layers: int = 4,
                 dropout: float = 0.2,
                 prenorm: bool = False,
                 lr: float = 0.0001) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_hidden = n_hidden
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.prenorm = prenorm
        self.lr = lr

        self.enc = nn.Linear(in_ch + latent_dim, n_hidden)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(n_hidden, dropout=dropout, transposed=True, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(n_hidden))
            self.dropouts.append(DropoutNd(dropout))
        self.dec = nn.Linear(n_hidden, out_ch)

    def forward(self, x: T, latent: T) -> T:
        assert x.ndim == 3
        assert latent.shape == (x.size(0), self.latent_dim, x.size(-1))
        in_audio = x
        x = tr.cat([latent, x], dim=1)
        x = tr.swapaxes(x, 1, 2)  # (B, L, C)
        x = self.enc(x)
        x = tr.swapaxes(x, 1, 2)  # (B, C, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)  # (B, L, C)
        # Decode the outputs
        x = self.dec(x)
        x = x.transpose(-1, -2)  # (B, C, L)
        y_hat = x + in_audio
        y_hat = tr.tanh(y_hat)
        return y_hat

    def update_hidden(self, hidden: Tuple[T, T]) -> None:
        return

    def detach_hidden(self) -> None:
        return

    def clear_hidden(self) -> None:
        return


if __name__ == "__main__":
    model = Spectral2DCNN()
    audio = tr.rand((3, 1, 88200))
    y = model(audio)
    log.info(y[0].shape)
