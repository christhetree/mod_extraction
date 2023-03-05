import logging
import os
from typing import Optional, List

import torch as tr
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torchaudio.transforms import Spectrogram

from erase_fx import tcn
from erase_fx.jtfst.scattering_1d import ScatTransform1D
from erase_fx.tcn import causal_crop, center_crop, TCN

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class AnalysisModel(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 kernel_size: int = 13,
                 out_channels: Optional[List[int]] = None,
                 strides: Optional[List[int]] = None,
                 latent_dim: int = 1,
                 norm_type: str = "identity",
                 causal: bool = False) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.norm_type = norm_type
        self.causal = causal
        log.info(f"Is causal: {self.causal}")
        assert not causal

        if out_channels is None:
            # out_channels = [8, 16, 32, 64, 128, 256, 512, 1024]
            out_channels = [108] * 8
            log.info(f"Setting out_channels automatically to: {out_channels}")
        self.n_blocks = len(out_channels)
        self.out_channels = out_channels

        if strides is None:
            # strides = [2, 2, 2, 2, 2, 4, 4, 4]
            strides = [2] * self.n_blocks
            # strides[0] = 1
            log.info(f"Setting strides automatically to: {strides}")
        self.strides = strides

        self.dilations = [1] * self.n_blocks
        # self.dilations = [5 ** idx for idx in range(self.n_blocks)]
        # self.dilations = [4 ** idx for idx in range(self.n_blocks)]

        if self.causal:
            self.crop_fn = causal_crop
        else:
            self.crop_fn = center_crop

        # TCN
        self.tcn = TCN(out_channels,
                       self.dilations,
                       in_ch,
                       kernel_size,
                       strides,
                       padding=None,
                       crop_fn=self.crop_fn)
        # self.receptive_field = self.tcn.calc_receptive_field()
        # log.info(f"Receptive field = {self.receptive_field} samples")
        # self.output = nn.Conv1d(out_channels[-1], self.latent_dim, kernel_size=(1,))

        # # Projection
        self.proj = nn.Linear(out_channels[-1], latent_dim)
        # if norm_type == "layer":
        #     self.norm = nn.LayerNorm(latent_dim)
        # elif norm_type == "batch":
        #     self.norm = nn.BatchNorm1d(latent_dim)
        # elif norm_type == "identity":
        #     self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.tcn(x)
        y = tr.mean(y, dim=-1)
        c = self.proj(y)
        # c = self.norm(c)
        # c = self.output(y)
        c = tr.sigmoid(c)
        return c


class Spectral1DCNNModel(nn.Module):
    def __init__(self,
                 n_fft: int = 1024,
                 hop_len: Optional[int] = None,
                 kernel_size: int = 5,
                 out_channels: Optional[List[int]] = None,
                 n_fc_units: int = 128,
                 latent_dim: int = 1) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.kernel_size = kernel_size
        self.n_fc_units = n_fc_units
        self.latent_dim = latent_dim

        # J = 8
        # Q = 16
        # self.scalogram = ScatTransform1D(sr=44100,
        #                                  J=J,
        #                                  Q=Q,
        #                                  should_avg=True,
        #                                  avg_win=256,
        #                                  highest_freq=20000,
        #                                  squeeze_channels=True)
        # log.info(f"scalogram lowest freq: {self.scalogram.freqs_t[-1]:.2f}")
        # in_ch = J * Q + 1
        # out_channels = [128] * 5

        if hop_len is None:
            hop_len = n_fft // 4
        self.hop_len = hop_len
        self.spectrogram = Spectrogram(n_fft, hop_length=hop_len, normalized=False)
        in_ch = n_fft // 2 + 1
        out_channels = [96] * 5

        self.out_channels = out_channels
        dilations = [2 ** idx for idx in range(len(out_channels))]
        self.cnn = TCN(out_channels,
                       dilations,
                       in_ch,
                       kernel_size,
                       padding=None,
                       use_ln=True,
                       # temporal_dims=[344] * len(out_channels),
                       temporal_dims=[345] * len(out_channels),
                       use_res=True,
                       crop_fn=tcn.center_crop)
        self.receptive_field = self.cnn.calc_receptive_field()
        log.info(f"Receptive field = {self.receptive_field} samples")
        self.output = nn.Conv1d(out_channels[-1], self.latent_dim, kernel_size=(1,))
        # self.output = nn.Conv1d(out_channels[-1], self.latent_dim, kernel_size=(8,), padding=0)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        # x, _, _ = self.scalogram(x)

        x = self.spectrogram(x).squeeze(1)
        x = tr.clip(x, min=1e-7)
        x = tr.log(x)

        x = self.cnn(x)
        x = self.output(x)
        x = tr.sigmoid(x)

        x = x.unfold(dimension=-1, size=8, step=1)
        x = tr.mean(x, dim=-1, keepdim=False)
        return x


class EffectModel(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
                 n_blocks: int = 4,
                 channel_growth: int = 0,
                 channel_width: int = 32,
                 kernel_size: int = 13,
                 dilation_growth: int = 10,
                 latent_dim: int = 2,
                 cond_dim: int = 32,
                 cond_hidden: int = 256,
                 causal: bool = False) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_blocks = n_blocks
        self.channel_growth = channel_growth
        self.channel_width = channel_width
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.cond_hidden = cond_hidden
        self.causal = causal
        log.info(f"Is causal: {self.causal}")

        if channel_growth > 1:
            out_channels = [in_ch * (channel_growth ** (idx + 1)) for idx in range(n_blocks)]
        else:
            out_channels = [channel_width] * n_blocks
        log.info(f"Setting out_channels to: {out_channels}")
        self.out_channels = out_channels

        self.dilations = [dilation_growth ** idx for idx in range(n_blocks)]
        log.info(f"Setting dilations to: {self.dilations}")

        if self.causal:
            self.crop_fn = causal_crop
        else:
            self.crop_fn = center_crop

        # TCN
        self.tcn = TCN(out_channels,
                       self.dilations,
                       in_ch,
                       kernel_size,
                       cond_dim=cond_dim,
                       crop_fn=self.crop_fn)
        self.receptive_field = self.tcn.calc_receptive_field()
        log.info(f"Receptive field = {self.receptive_field} samples")
        self.output = nn.Conv1d(out_channels[-1], out_ch, kernel_size=(1,))

        # Conditioning mapping
        self.cond_mapping = nn.Sequential(
            nn.Linear(latent_dim, cond_hidden),
            nn.PReLU(),
            nn.Linear(cond_hidden, cond_hidden),
            nn.PReLU(),
            nn.Linear(cond_hidden, cond_dim),
        )

    def forward(self, x: Tensor, latent: Tensor) -> Tensor:
        assert latent.shape == (x.size(0), self.latent_dim)
        x_in = x
        cond = self.cond_mapping(latent)  # Latent dim to cond dim
        x = self.tcn(x, cond)
        x = self.output(x)
        y_hat = tr.tanh(x)

        # TODO(cm): figure out which is most appropriate here
        # x_in = self.crop_fn(x_in, x.size(-1))
        # y_hat = tr.tanh(x * x_in)
        return y_hat


class LSTMEffectModel(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, n_hidden: int = 32, latent_dim: int = 1) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_hidden = n_hidden
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(in_ch + latent_dim, n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, out_ch)

    def forward(self, x: Tensor, latent: Tensor) -> Tensor:
        # assert latent.shape == (x.size(0), self.latent_dim, x.size(2))
        assert latent.ndim == 3
        assert latent.size(0) == x.size(0)
        assert latent.size(1) == self.latent_dim
        n_samples = x.size(2)
        latent_upsampled = F.interpolate(latent, size=n_samples, mode="linear", align_corners=True)
        lstm_in = tr.cat([latent_upsampled, x], dim=1)
        lstm_in = tr.swapaxes(lstm_in, 1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        fc_out = self.fc(lstm_out)
        fc_out = tr.swapaxes(fc_out, 1, 2)
        y_hat = fc_out + x
        y_hat = tr.tanh(y_hat)
        return y_hat


if __name__ == "__main__":
    model = Spectral1DCNNModel()
    audio = tr.rand((3, 1, 88200))
    y = model(audio)
