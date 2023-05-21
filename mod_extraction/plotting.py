import io
import logging
import os
from typing import Optional, List

import PIL
import librosa
import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from matplotlib.axes import Subplot
from matplotlib.figure import Figure
from torch import Tensor as T
from torchaudio.transforms import Spectrogram, Fade
from torchvision.transforms import ToTensor

from mod_extraction.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def fig2img(fig: Figure, format: str = "png", dpi: int = 120) -> T:
    """Convert a matplotlib figure to tensor."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close("all")
    return image


def plot_spectrogram(audio: T,
                     ax: Optional[Subplot] = None,
                     title: Optional[str] = None,
                     save_name: Optional[str] = None,
                     save_dir: str = OUT_DIR,
                     sr: float = 44100,
                     fade_n_samples: int = 64) -> T:
    assert audio.ndim < 3
    audio = audio.detach()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    assert audio.size(0) == 1
    spectrogram = Spectrogram(n_fft=512, power=1, center=True, normalized=False)
    spec = tr.log(spectrogram(audio).squeeze(0))
    if ax is None:
        plt.imshow(spec, aspect="auto", interpolation="none")
        plt.title(title)
        plt.show()
    else:
        ax.imshow(spec, aspect="auto", interpolation="none")
        if title is not None:
            ax.set_title(title)

    if save_name is not None:
        sr = int(sr)
        if not save_name.endswith(".wav"):
            save_name = f"{save_name}.wav"
        if fade_n_samples:
            transform = Fade(fade_in_len=fade_n_samples, fade_out_len=fade_n_samples, fade_shape="linear")
            audio = transform(audio)
        save_path = os.path.join(save_dir, save_name)
        torchaudio.save(save_path, audio, sr)

    return spec


def plot_mod_sig(mod_sig_hat: T,
                 mod_sig: Optional[T] = None,
                 mod_sig_hat_c: str = "orange",
                 mod_sig_c: str = "black",
                 linewidth: float = 3.0,
                 save_name: Optional[str] = None,
                 save_dir: str = OUT_DIR,
                 l1_error_title: bool = True) -> None:
    if mod_sig is not None:
        plt.plot(mod_sig, c=mod_sig_c, linewidth=linewidth)
    plt.plot(mod_sig_hat, c=mod_sig_hat_c, linewidth=linewidth)
    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([0.0, 0.5, 1.0])
    plt.yticks(fontsize=18)

    if l1_error_title and mod_sig is not None:
        assert mod_sig.shape == mod_sig_hat.shape
        mae = tr.mean(tr.abs(mod_sig - mod_sig_hat))
        plt.title(f"{mae * 100:.1f}% L1 Error", fontsize=28)

    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(save_dir, f"{save_name}.svg"))
    plt.show()


def plot_mod_sig_callback(ax: Subplot,
                          mod_sig_hat: T,
                          mod_sig: T,
                          title: Optional[str] = None) -> None:
    assert mod_sig_hat.ndim == mod_sig.ndim == 1
    mod_sig_hat = mod_sig_hat.detach()
    mod_sig = mod_sig.detach()
    ax.plot(mod_sig)
    ax.plot(mod_sig_hat)
    if title is not None:
        ax.set_title(title)


def plot_waveforms_stacked(waveforms: List[T],
                           sr: float,
                           title: Optional[str] = None,
                           waveform_labels: Optional[List[str]] = None,
                           show: bool = False) -> Figure:
    assert waveforms
    if waveform_labels is None:
        waveform_labels = [None] * len(waveforms)
    assert len(waveform_labels) == len(waveforms)

    fig, axs = plt.subplots(
        nrows=len(waveforms),
        sharex="all",
        sharey="all",
        figsize=(7, 2 * len(waveforms)),
        squeeze=True,
    )

    for idx, (ax, w, label) in enumerate(zip(axs, waveforms, waveform_labels)):
        assert 0 < w.ndim <= 2
        if w.ndim == 2:
            assert w.size(0) == 1
            w = w.squeeze(0)
        w = w.detach().float().cpu().numpy()
        if idx == len(waveforms) - 1:
            axis = "time"
        else:
            axis = None
        librosa.display.waveshow(w, axis=axis, sr=sr, label=label, ax=ax)
        ax.set_title(label)
        ax.grid(color="lightgray", axis="x")
        # ax.set_xticks([])
        # ax.set_yticks([])

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    # fig.savefig(os.path.join(OUT_DIR, f"3.svg"))

    if show:
        fig.show()
    return fig
