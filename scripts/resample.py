import logging
import os

import torchaudio
from torchaudio.transforms import Resample

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    src_path = "../data/marco/vocalset/vocalset_input.wav"
    dest_path = "../data/marco/vocalset/vocalset_input_44100.wav"
    new_sr = 44100
    src_audio, src_sr = torchaudio.load(src_path)
    resampler = Resample(src_sr, new_sr)
    new_audio = resampler(src_audio)
    torchaudio.save(dest_path, new_audio, new_sr)
