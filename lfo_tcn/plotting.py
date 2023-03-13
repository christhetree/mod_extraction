import io
import logging
import os
from typing import Optional

import PIL
import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from matplotlib.axes import Subplot
from matplotlib.figure import Figure
from torch import Tensor as T
from torchaudio.transforms import Spectrogram
from torchvision.transforms import ToTensor

from lfo_tcn.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def fig2img(fig: Figure, format: str = "png", dpi: int = 120) -> T:
    """Convert a matplotlib figure to JPEG to be show in Tensorboard."""
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
                     sr: float = 44100) -> T:
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
        save_path = os.path.join(save_dir, save_name)
        torchaudio.save(save_path, audio, sr)

    return spec


def plot_mod_sig(ax: Subplot,
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
