import logging
import os

import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from torch import Tensor as T
from torchaudio.transforms import Spectrogram

from lfo_tcn.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def display_spectrogram(audio: T,
                        save_audio: bool = True,
                        name: str = "audio",
                        sr: float = 44100,
                        save_dir: str = OUT_DIR,
                        idx: int = 0) -> None:
    assert audio.ndim < 3
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    assert audio.size(0) == 1
    spectrogram = Spectrogram(n_fft=512, power=1, center=True, normalized=False)
    spec = tr.log(spectrogram(audio).squeeze(0))
    plt.imshow(spec, aspect="auto", interpolation="none")
    plt.show()
    if save_audio:
        sr = int(sr)
        save_path = os.path.join(save_dir, f"{name}_{idx}.wav")
        torchaudio.save(save_path, audio, sr)
