import logging
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
                 n_samples: int = 88100,
                 n_fft: int = 1024,
                 hop_len: int = 256,
                 kernel_size: int = 13,
                 out_channels: Optional[List[int]] = None,
                 dilations: Optional[List[int]] = None,
                 n_fc_units: int = 128,
                 latent_dim: int = 1,
                 smooth_n_frames: int = 8,
                 use_ln: bool = True,
                 use_res: bool = True) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.kernel_size = kernel_size
        self.n_fc_units = n_fc_units
        self.latent_dim = latent_dim
        self.smooth_n_frames = smooth_n_frames
        self.use_ln = use_ln
        self.use_res = use_res
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
        x = tr.clip(x, min=1e-7)  # TODO(cm)
        x = tr.log(x)

        x = self.tcn(x)
        x = self.output(x)
        x = tr.sigmoid(x)

        if self.smooth_n_frames > 1:
            x = x.unfold(dimension=-1, size=self.smooth_n_frames, step=1)
            x = tr.mean(x, dim=-1, keepdim=False)
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
    model = LSTMEffectModel()
    audio = tr.rand((3, 1, 1024))
    latent = tr.rand((3, 1, 1024))
    y = model(audio, latent)
