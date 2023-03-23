import logging
import math
import os
from typing import Optional, List, Tuple

import torch as tr
from torch import Tensor as T
from torch import nn
from torchaudio.transforms import Spectrogram

from lfo_tcn.tcn import TCN

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class SpectralTCN(nn.Module):
    def __init__(self,
                 n_samples: int = 88200,
                 n_fft: int = 1024,
                 hop_len: int = 256,
                 kernel_size: int = 13,
                 out_channels: Optional[List[int]] = None,
                 dilations: Optional[List[int]] = None,
                 latent_dim: int = 1,
                 smooth_n_frames: int = 8,
                 use_ln: bool = True,
                 use_res: bool = True,
                 eps: float = 1e-7) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.smooth_n_frames = smooth_n_frames
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

        # lstm_dim = 64
        # self.lstm = nn.LSTM(out_channels[-1], lstm_dim, batch_first=True, bidirectional=True)
        # self.output = nn.Conv1d(2 * lstm_dim, self.latent_dim, kernel_size=(1,))

    def forward(self, x: T) -> T:
        assert x.ndim == 3
        x = self.spectrogram(x).squeeze(1)
        x = tr.clip(x, min=self.eps)
        x = tr.log(x)

        x = self.tcn(x)

        # lstm_in = tr.swapaxes(x, 1, 2)
        # lstm_out, _ = self.lstm(lstm_in)
        # x = tr.swapaxes(lstm_out, 1, 2)

        x = self.output(x)
        x = tr.sigmoid(x)

        if self.smooth_n_frames > 1:
            x = x.unfold(dimension=-1, size=self.smooth_n_frames, step=1)
            x = tr.mean(x, dim=-1, keepdim=False)
        return x


class Spectral2DCNN(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 n_samples: int = 88200,
                 n_fft: int = 1024,
                 hop_len: int = 256,
                 kernel_size: Tuple[int, int] = (5, 13),
                 out_channels: Optional[List[int]] = None,
                 bin_dilations: Optional[List[int]] = None,
                 temp_dilations: Optional[List[int]] = None,
                 pool_size: Tuple[int, int] = (3, 1),
                 latent_dim: int = 1,
                 smooth_n_frames: int = 8,
                 use_ln: bool = True,
                 scale_output: bool = False,
                 eps: float = 1e-7) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.kernel_size = kernel_size
        assert pool_size[1] == 1
        self.pool_size = pool_size
        self.latent_dim = latent_dim
        self.smooth_n_frames = smooth_n_frames
        self.use_ln = use_ln
        self.scale_output = scale_output
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

        self.spectrogram = Spectrogram(n_fft, hop_length=hop_len, normalized=False)
        n_bin = n_fft // 2 + 1
        n_frames = n_samples // hop_len + 1
        temporal_dims = [n_frames] * len(out_channels)

        layers = []
        for out_ch, b_dil, t_dil, temp_dim in zip(out_channels, bin_dilations, temp_dilations, temporal_dims):
            if use_ln:
                layers.append(nn.LayerNorm([in_ch, n_bin, temp_dim], elementwise_affine=False))
                # layers.append(nn.BatchNorm2d(in_ch, affine=False))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=(1, 1), dilation=(b_dil, t_dil), padding="same"))
            # padding = (kernel_size[0] // 2 * b_dil, kernel_size[1] // 2 * t_dil)
            # layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=pool_size, dilation=(b_dil, t_dil), padding=padding))
            # n_bin = math.ceil(n_bin / pool_size[0])
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
            layers.append(nn.PReLU(num_parameters=out_ch))
            in_ch = out_ch
            n_bin = n_bin // pool_size[0]
        self.cnn = nn.Sequential(*layers)

        self.output = nn.Conv1d(out_channels[-1], self.latent_dim, kernel_size=(1,))

        # fc_dim = out_channels[-1] // 2
        # self.phase_fc = nn.Linear(out_channels[-1], fc_dim)
        # self.phase_act = nn.PReLU()
        # self.phase_out = nn.Linear(fc_dim, 1)
        # self.freq_fc = nn.Linear(out_channels[-1], fc_dim)
        # self.freq_act = nn.PReLU()
        # self.freq_out = nn.Linear(fc_dim, 1)
        # self.shape_fc = nn.Linear(out_channels[-1], fc_dim)
        # self.shape_act = nn.PReLU()
        # self.shape_out = nn.Linear(fc_dim, 4)

    def forward(self, x: T) -> T:
        assert x.ndim == 3
        x = self.spectrogram(x)

        x = tr.clip(x, min=self.eps)
        x = tr.log(x)

        # plt.imshow(x[0, 1, :, :].detach().numpy())
        # plt.show()
        # n_bins = x.size(2)
        # min_n = int(0.05 * n_bins)
        # n = tr.randint(min_n, n_bins, size=(1,)).item()
        # n = int(RandomAudioChunkDataset.sample_log_uniform(float(min_n), float(n_bins)))
        # x = x[:, :, :n, :]
        # x = nn.functional.interpolate(x, (n_bins, x.size(3)), mode="bicubic", align_corners=True)
        # plt.imshow(x[0, 1, :, :].detach().numpy())
        # plt.show()

        x = self.cnn(x)
        x = tr.mean(x, dim=-2)
        # latent = tr.mean(x, dim=-2)

        x = self.output(x)
        # x = self.output(latent)
        x = tr.sigmoid(x)

        if self.smooth_n_frames > 1:
            x = x.unfold(dimension=-1, size=self.smooth_n_frames, step=1)
            x = tr.mean(x, dim=-1, keepdim=False)

        if self.scale_output:
            x_min = tr.min(x.squeeze(1), dim=-1).values.view(-1, 1, 1)
            x_max = tr.max(x.squeeze(1), dim=-1).values.view(-1, 1, 1)
            x -= x_min
            x *= (1.0 / (x_max - x_min))

        # mod_sig_hat = x
        #
        # x = tr.chunk(latent, 15, dim=-1)
        # x = tr.stack(x, dim=1)
        # latent_2 = tr.mean(x, dim=-1)
        #
        # x = self.phase_fc(latent_2)
        # x = self.phase_act(x)
        # x = self.phase_out(x)
        # phase_hat = tr.relu(x)
        # x = self.freq_fc(latent_2)
        # x = self.freq_act(x)
        # x = self.freq_out(x)
        # freq_hat = tr.relu(x)
        # x = self.shape_fc(latent_2)
        # x = self.shape_act(x)
        # x = self.shape_out(x)
        # x = tr.softmax(x, dim=-1)
        # shape_hat = x.swapaxes(1, 2)

        return x
        # return mod_sig_hat, phase_hat.squeeze(-1), freq_hat.squeeze(-1), shape_hat


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
        self.hidden: Optional[Tuple[T, ...]] = None

    def detach_hidden(self) -> None:
        if self.hidden is not None:
            # TODO: check whether clone is required or not
            self.hidden = tuple((h.detach().clone() for h in self.hidden))

    def clear_hidden(self) -> None:
        self.hidden = None


class LSTMEffectModel(HiddenStateModel):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
                 n_hidden: int = 48,
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
        lstm_out, new_hidden = self.lstm(lstm_in, self.hidden)
        fc_out = self.fc(lstm_out)
        fc_out = tr.swapaxes(fc_out, 1, 2)
        y_hat = fc_out + x
        y_hat = tr.tanh(y_hat)
        self.hidden = new_hidden
        return y_hat


if __name__ == "__main__":
    # model = SpectralTCN(kernel_size=5,
    #                     out_channels=[96] * 5,
    #                     dilations=[1, 3, 9, 27, 81],
    #                     use_ln=False)
    # print(model.tcn.calc_receptive_field())
    model = Spectral2DCNN()
    # model = SpectralDSTCN()
    # model = LSTMEffectModel()

    # latent = tr.rand((3, 1, 1024))
    audio = tr.rand((3, 1, 88200))
    y = model(audio)
    print(y.shape)
