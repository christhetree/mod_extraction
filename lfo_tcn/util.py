import io
import logging
import os
import random
import shutil
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
                     sr: float = 44100) -> None:
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


def split_egfx(root_dir: str, val_split: float = 0.3) -> None:
    train_names = None
    val_names = None
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    for dir_name in os.listdir(root_dir):
        dir = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir) and dir_name not in {"train", "val"}:
            if val_names is None:
                names = os.listdir(dir)
                names = [n for n in names if n.endswith(".wav") and not n.startswith(".")]
                val_names = random.sample(names, int(val_split * len(names)))
                for n in val_names:
                    names.remove(n)
                train_names = names

            train_dest_dir = os.path.join(train_dir, dir_name)
            os.makedirs(train_dest_dir, exist_ok=True)
            for n in train_names:
                file_path = os.path.join(dir, n)
                assert os.path.isfile(file_path)
                dest_path = os.path.join(train_dest_dir, n)
                shutil.copyfile(file_path, dest_path)

            val_dest_dir = os.path.join(val_dir, dir_name)
            os.makedirs(val_dest_dir, exist_ok=True)
            for n in val_names:
                file_path = os.path.join(dir, n)
                assert os.path.isfile(file_path)
                dest_path = os.path.join(val_dest_dir, n)
                shutil.copyfile(file_path, dest_path)


if __name__ == "__main__":
    from lfo_tcn.paths import DATA_DIR
    split_egfx(os.path.join(DATA_DIR, "egfx_clean"))
