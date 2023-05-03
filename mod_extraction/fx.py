import logging
import os
from typing import Union

import torch as tr
from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def apply_tremolo(x: T, mod_sig: T, mix: Union[float, T] = 1.0) -> T:
    assert x.ndim == 3
    assert x.size(0) == mod_sig.size(0)
    assert x.size(-1) == mod_sig.size(-1)
    if mod_sig.ndim == 2:
        mod_sig = mod_sig.unsqueeze(1).expand(-1, x.size(1), -1)
    if isinstance(mix, T):
        assert mix.size(0) == x.size(0)
    assert 0.0 <= mix <= 1.0
    return ((1.0 - mix) * x) + (mix * mod_sig * x)


class MonoFlangerChorusModule(nn.Module):
    def __init__(self,
                 batch_size: int,
                 n_ch: int,
                 n_samples: int,
                 sr: float,
                 max_min_delay_ms: float,
                 max_lfo_delay_ms: float) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.n_ch = n_ch
        self.n_samples = n_samples
        self.sr = sr
        self.max_min_delay_ms = max_min_delay_ms
        self.max_lfo_delay_ms = max_lfo_delay_ms
        self.max_min_delay_samples = int(((max_min_delay_ms / 1000.0) * sr) + 0.5)
        self.max_lfo_delay_samples = int(((max_lfo_delay_ms / 1000.0) * sr) + 0.5)
        self.max_delay_samples = self.max_min_delay_samples + self.max_lfo_delay_samples
        self.register_buffer("delay_buf", tr.zeros((batch_size, n_ch, self.max_delay_samples)))
        self.register_buffer("out_buf", tr.zeros((batch_size, n_ch, n_samples)))

    def check_param(self,
                    param: Union[float, T],
                    bs: int,
                    out_n_dim: int = 2,
                    can_be_one: bool = True) -> Union[float, T]:
        if isinstance(param, T):
            assert param.shape == (bs,)
            assert param.min() >= 0
            if can_be_one:
                assert param.max() <= 1.0
            else:
                assert param.max() < 1.0
            if out_n_dim == 2:
                param = param.view(-1, 1)
            elif out_n_dim == 3:
                param = param.view(-1, 1, 1)
            else:
                raise ValueError
        else:
            assert param >= 0
            if can_be_one:
                assert param <= 1.0
            else:
                assert param < 1.0
        return param

    def apply_effect(self,
                     x: T,
                     mod_sig: T,
                     feedback: Union[float, T],
                     min_delay_width: Union[float, T],
                     width: Union[float, T],
                     depth: Union[float, T],
                     mix: Union[float, T]) -> T:
        assert x.ndim == 3
        batch_size, n_ch, n_samples = x.shape
        assert mod_sig.size(0) == batch_size
        assert mod_sig.size(-1) == n_samples
        if mod_sig.ndim == 2:
            mod_sig = mod_sig.unsqueeze(1).expand(-1, n_ch, -1)
        feedback = self.check_param(feedback, batch_size, out_n_dim=2, can_be_one=False)
        min_delay_width = self.check_param(min_delay_width, batch_size, out_n_dim=3, can_be_one=True)
        width = self.check_param(width, batch_size, out_n_dim=3, can_be_one=True)
        depth = self.check_param(depth, batch_size, out_n_dim=2, can_be_one=True)
        mix = self.check_param(mix, batch_size, out_n_dim=3, can_be_one=True)

        self.delay_buf.fill_(0)
        self.out_buf.fill_(0)

        delay_write_idx_all = tr.arange(0, n_samples) % self.max_delay_samples
        delay_write_idx_all = delay_write_idx_all.view(1, 1, -1).expand(batch_size, n_ch, -1)
        min_delay_samples = min_delay_width * self.max_min_delay_samples
        delay_samples_all = (self.max_lfo_delay_samples * width * mod_sig) + min_delay_samples
        delay_read_idx_all = (delay_write_idx_all - delay_samples_all + self.max_delay_samples) % self.max_delay_samples
        delay_read_fraction_all = delay_read_idx_all - tr.floor(delay_read_idx_all)
        prev_idx_all = tr.floor(delay_read_idx_all).to(tr.long)
        next_idx_all = (prev_idx_all + 1) % self.max_delay_samples

        for idx in range(n_samples):
            audio_val = x[:, :, idx]
            prev_idx = prev_idx_all[:, :, idx].unsqueeze(-1)
            next_idx = next_idx_all[:, :, idx].unsqueeze(-1)
            delay_read_fraction = delay_read_fraction_all[:, :, idx]
            delay_write_idx = delay_write_idx_all[0, 0, idx]

            prev_val = tr.gather(self.delay_buf, dim=-1, index=prev_idx).squeeze(-1)
            next_val = tr.gather(self.delay_buf, dim=-1, index=next_idx).squeeze(-1)
            interp_val = (delay_read_fraction * next_val) + ((1.0 - delay_read_fraction) * prev_val)
            self.delay_buf[:, :, delay_write_idx] = audio_val + (feedback * interp_val)
            self.out_buf[:, :, idx] = audio_val + (depth * interp_val)

        out_buf = ((1.0 - mix) * x) + (mix * self.out_buf)
        out_buf = tr.clip(out_buf, -1.0, 1.0)  # TODO(cm): should clip flag
        return out_buf

    def forward(self,
                x: T,
                mod_sig: T,
                feedback: Union[float, T] = 0.0,
                min_delay_width: Union[float, T] = 1.0,
                width: Union[float, T] = 1.0,
                depth: Union[float, T] = 1.0,
                mix: Union[float, T] = 1.0) -> T:
        with tr.no_grad():
            return self.apply_effect(x, mod_sig, feedback, min_delay_width, width, depth, mix)
