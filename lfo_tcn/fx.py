import logging
import os
from typing import Union

import torch as tr
from matplotlib import pyplot as plt
from torch import Tensor as T, nn

from lfo_tcn.util import linear_interpolate_last_dim

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def make_mod_signal(n_samples: int,
                    sr: float,
                    freq: float,
                    phase: float = 0.0,
                    shape: str = "cos",
                    exp: float = 1.0) -> T:
    assert n_samples > 0
    assert 0.0 < freq < sr / 2.0
    assert -2 * tr.pi <= phase <= 2 * tr.pi
    assert shape in {"cos", "rect_cos", "inv_rect_cos", "tri", "saw", "rsaw", "sqr"}
    if shape in {"rect_cos", "inv_rect_cos"}:
        # Rectified sine waves have double the frequency
        freq /= 2.0
    assert exp > 0
    argument = tr.cumsum(2 * tr.pi * tr.full((n_samples,), freq) / sr, dim=0) + phase
    saw = tr.remainder(argument, 2 * tr.pi) / (2 * tr.pi)

    if shape == "cos":
        mod_sig = (tr.cos(argument + tr.pi) + 1.0) / 2.0
    elif shape == "rect_cos":
        mod_sig = tr.abs(tr.cos(argument + tr.pi))
    elif shape == "inv_rect_cos":
        mod_sig = -tr.abs(tr.cos(argument + tr.pi)) + 1.0
    elif shape == "sqr":
        cos = tr.cos(argument + tr.pi)
        sqr = tr.sign(cos)
        mod_sig = (sqr + 1.0) / 2.0
    elif shape == "saw":
        mod_sig = saw
    elif shape == "rsaw":
        mod_sig = 1.0 - saw
    elif shape == "tri":
        tri = 2 * saw
        mod_sig = tr.where(tri > 1.0, 2.0 - tri, tri)
    else:
        raise ValueError("Unsupported shape")

    if exp != 1.0:
        mod_sig = mod_sig ** exp
    return mod_sig


def mod_sig_to_corners(mod_sig: T, n_frames: int) -> (T, T):
    assert mod_sig.ndim == 2
    mod_sig = linear_interpolate_last_dim(mod_sig, n_frames, align_corners=True)
    return find_corners(mod_sig)


def find_corners(mod_sig: T) -> (T, T):
    m_r = mod_sig[:, 1:]
    m_l = mod_sig[:, :-1]
    diff = m_r - m_l
    diff_r = diff[:, 1:]
    diff_l = diff[:, :-1]
    diff_pos_l = (diff_l > 0) * diff_l
    diff_neg_l = (diff_l < 0) * diff_l
    top_corners = diff_pos_l * (diff_r + 1e-16)
    top_corners = -tr.floor(top_corners).long()
    bottom_corners = diff_neg_l * (diff_r + 1e-16)
    bottom_corners = -tr.floor(bottom_corners).long()
    tmp = tr.zeros_like(mod_sig)
    tmp[:, 1:-1] = top_corners
    top_corners = tmp
    tmp = tr.zeros_like(mod_sig)
    tmp[:, 1:-1] = bottom_corners
    bottom_corners = tmp
    return top_corners, bottom_corners


def corners_to_mod_sig(top_corners: T, bottom_corners: T) -> T:
    assert top_corners.ndim == 1
    assert top_corners.shape == bottom_corners.shape
    mod_sig = tr.zeros_like(top_corners).float()
    if top_corners.max() == 0 or bottom_corners.max() == 0:
        return mod_sig
    top_indices = tr.where(top_corners == 1)[0]
    top_indices = [(v.item(), 1) for v in top_indices]
    bottom_indices = tr.where(bottom_corners == 1)[0]
    bottom_indices = [(v.item(), 0) for v in bottom_indices]
    indices = top_indices + bottom_indices
    indices.sort(key=lambda x: x[0])
    for idx in range(len(indices) - 1):
        l_idx, l_v = indices[idx]
        r_idx, r_v = indices[idx + 1]
        mod_sig[l_idx:r_idx + 1] = tr.linspace(l_v, r_v, r_idx - l_idx + 1)
    return mod_sig


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


def smoothen(x: T, smooth_n_frames: int) -> T:
    if smooth_n_frames > 1:
        x = x.unfold(dimension=-1, size=smooth_n_frames, step=1)
        x = tr.mean(x, dim=-1, keepdim=False)
    return x


def _stretch_corners(mod_sig: T, top: T, bottom: T, top_val: float = 1.0, bot_val: float = 0.0) -> T:
    assert mod_sig.ndim == 1
    assert mod_sig.shape == top.shape == bottom.shape
    top_indices = (top == 1).nonzero(as_tuple=True)[0]
    top = [(idx.item(), top_val) for idx in top_indices]
    bottom_indices = (bottom == 1).nonzero(as_tuple=True)[0]
    bottom = [(idx.item(), bot_val) for idx in bottom_indices]
    indices = top + bottom + [(mod_sig.size(0) - 1, mod_sig[-1])]
    if not indices:
        return mod_sig
    indices.sort(key=lambda x: x[0])

    prev_mod_idx = 0
    prev_anchor = mod_sig[0]
    stretched = tr.clone(mod_sig)
    for curr_mod_idx, target_val in indices:
        segment = stretched[prev_mod_idx + 1:curr_mod_idx + 1]
        curr_val = mod_sig[curr_mod_idx]
        orig_prev_anchor = mod_sig[prev_mod_idx]
        if prev_anchor != target_val:
            curr_range = abs(orig_prev_anchor - curr_val)
            target_range = abs(prev_anchor - target_val)
            scale_amount = target_range / curr_range
            segment -= segment.min()
            segment *= scale_amount
            segment += target_val - segment[-1]
            stretched[prev_mod_idx + 1:curr_mod_idx + 1] = segment
        prev_mod_idx = curr_mod_idx
        prev_anchor = target_val

    # assert stretched.max() == top_val or stretched.min() == bot_val
    return stretched


def stretch_corners(mod_sig: T, max_n_corners: int = 10, smooth_n_frames: int = 32) -> T:
    assert mod_sig.ndim == 2
    mod_sig = smoothen(mod_sig, smooth_n_frames)
    top_corners, bottom_corners = find_corners(mod_sig)
    stretched_all = []
    for m, t, b in zip(mod_sig, top_corners, bottom_corners):
        n_corners = t.sum() + b.sum()
        if n_corners > max_n_corners:
            stretched = m
        else:
            stretched = _stretch_corners(m, t, b)
        stretched_all.append(stretched)
    stretched = tr.stack(stretched_all, dim=0)
    return stretched


if __name__ == "__main__":
    # audio, sr = torchaudio.load("/Users/puntland/local_christhetree/aim/lfo_tcn/data/idmt_4/train/Ibanez 2820__pop_2_140BPM.wav")
    # # audio, sr = torchaudio.load("/Users/puntland/local_christhetree/aim/lfo_tcn/data/idmt_4/train/Ibanez 2820__latin_1_160BPM.wav")
    # n_samples = sr * 4
    # audio = audio[:, :n_samples]
    # # audio = make_mod_signal(n_samples, sr, 220.0, shape="saw", exp=1.0)
    # audio = audio.view(1, 1, -1).repeat(3, 1, 1)

    n_samples = 12
    sr = 6

    mod_sig_a = make_mod_signal(n_samples, sr, 1.0, phase=0, shape="saw", exp=0.1)
    mod_sig_b = make_mod_signal(n_samples, sr, 0.5, phase=tr.pi, shape="tri", exp=1.0)
    mod_sig_c = make_mod_signal(n_samples, sr, 0.5, phase=tr.pi, shape="saw", exp=1.0)

    mod_sig = tr.stack([mod_sig_a, mod_sig_b, mod_sig_c], dim=0)
    mod_sig *= 0.5
    mod_sig += 0.2
    stretched = stretch_corners(mod_sig)

    idx = 0
    plt.plot(mod_sig[idx])
    plt.plot(stretched[idx])

    # top_corners, bottom_corners = find_corners(mod_sig)
    # rec_mod_sig = corners_to_mod_sig(top_corners[idx], bottom_corners[idx])
    # plt.plot(top_corners[idx])
    plt.show()

    exit()
    # mod_sig = tr.stack([mod_sig_b, mod_sig_b, mod_sig_b], dim=0)
    flanger = MonoFlangerChorusModule(3, 1, n_samples, sr, 0.0, 5.0)
    chorus = MonoFlangerChorusModule(3, 1, n_samples, sr, 30.0, 10.0)

    feedback = tr.Tensor([0.7, 0.4, 0.0])
    # depth = tr.Tensor([0.0, 0.5, 1.0])
    # width = tr.Tensor([0.0, 0.5, 1.0])
    # mix = tr.Tensor([0.0, 0.5, 1.0])
    # width = 1.0

    print(f"audio in min = {tr.min(audio):.3f}")
    print(f"audio in max = {tr.max(audio):.3f}")
    y_flanger = flanger(audio, mod_sig, )
    print(f"y    out min = {tr.min(y_flanger.squeeze(1), dim=-1)[0]}")
    print(f"y    out max = {tr.max(y_flanger.squeeze(1), dim=-1)[0]}")

    # y_chorus = chorus(audio, mod_sig, feedback=feedback)

    from lfo_tcn.plotting import plot_spectrogram
    for idx in range(3):
        y_f = y_flanger[idx]
        plot_spectrogram(y_f, title=f"flanger_{idx}", sr=sr, save_name=f"flanger_{idx}")
        # y_c = y_chorus[idx]
        # plot_spectrogram(y_c, title=f"chorus_{idx}", sr=sr, save_name=f"chorus_{idx}")
