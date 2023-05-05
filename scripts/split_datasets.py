import logging
import os
import random
import shutil

import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

from mod_extraction.paths import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def save_dir_structure(root_dir: str, out_path: str, ext: str = ".wav") -> None:
    file_paths = []
    for curr_dir, _, file_names in os.walk(os.path.join(root_dir)):
        for file_name in file_names:
            if file_name.endswith(ext) and not file_name.startswith("."):
                file_path = os.path.join(curr_dir, file_name)
                rel_path = os.path.relpath(file_path, root_dir)
                file_paths.append(rel_path)
    file_paths = sorted(file_paths)
    with open(out_path, "w") as out_f:
        for file_path in file_paths:
            out_f.write(f"{file_path}\n")


def split_idmt_4(root_dir: str, val_split: float = 0.25, offset_n_bars: int = 3, seed: int = 42) -> None:
    random.seed(seed)
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    dir_level_1 = ["acoustic_mic", "acoustic_pickup", "Career SG", "Ibanez 2820"]
    song_names = set()
    val_names = []
    for dir_name in dir_level_1:
        input_paths = []
        for root_dir_2, _, file_names in os.walk(os.path.join(root_dir, dir_name)):
            for file_name in file_names:
                if file_name.endswith(".wav") and not file_name.startswith("."):
                    input_paths.append(os.path.join(root_dir_2, file_name))
        if len(input_paths) == 128 and not song_names:
            for input_path in input_paths:
                file_name = os.path.basename(input_path)
                tokens = file_name.split("_")
                song_name = "_".join(tokens[:2])
                song_names.add(song_name)
        assert len(song_names) == 64
        song_names = sorted(list(song_names))
        if not val_names:
            val_names = random.sample(song_names, int(val_split * len(song_names)))
        for src_path in tqdm(input_paths):
            file_name = os.path.basename(src_path)
            tokens = file_name.split("_")
            bpm_token = tokens[-1]
            bpm = int(bpm_token[:-7])
            assert 50 <= bpm <= 200, f"Bad bpm: {bpm}"
            audio, sr = torchaudio.load(src_path)
            offset_n_samples = int(1.0 / (bpm / 60.0 / 4.0) * offset_n_bars * sr)
            audio = audio[:, offset_n_samples:]
            dest_name = f"{dir_name}__{file_name}"
            if any([n in file_name for n in val_names]):
                dest_path = os.path.join(val_dir, dest_name)
            else:
                dest_path = os.path.join(train_dir, dest_name)
            torchaudio.save(dest_path, audio, sr)


def split_egfx(root_dir: str,
               val_split: float = 0.18,
               test_split: float = 0.12,
               new_sr: int = 44100,
               seed: int = 42) -> None:
    random.seed(seed)
    train_names = None
    val_names = None
    test_names = None
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)
    for dir_name in tqdm(os.listdir(root_dir)):
        dir = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir) and dir_name not in {"train", "val", "test"}:
            if train_names is None:
                names = os.listdir(dir)
                names = [n for n in names if n.endswith(".wav") and not n.startswith(".")]
                n_names = len(names)
                val_names = random.sample(names, int(val_split * n_names))
                for n in val_names:
                    names.remove(n)
                test_names = random.sample(names, int(test_split * n_names))
                for n in test_names:
                    names.remove(n)
                train_names = names

            train_dest_dir = os.path.join(train_dir, dir_name)
            val_dest_dir = os.path.join(val_dir, dir_name)
            test_dest_dir = os.path.join(test_dir, dir_name)
            os.makedirs(train_dest_dir, exist_ok=True)
            os.makedirs(val_dest_dir, exist_ok=True)
            os.makedirs(test_dest_dir, exist_ok=True)
            for names, dest_dir in [(train_names, train_dest_dir),
                                    (val_names, val_dest_dir),
                                    (test_names, test_dest_dir)]:
                for n in names:
                    src_path = os.path.join(dir, n)
                    assert os.path.isfile(src_path)
                    dest_path = os.path.join(dest_dir, n)
                    src_sr = torchaudio.info(src_path).sample_rate
                    if src_sr != new_sr:
                        src_audio, src_sr = torchaudio.load(src_path)
                        resampler = Resample(src_sr, new_sr)
                        new_audio = resampler(src_audio)
                        torchaudio.save(dest_path, new_audio, new_sr)
                    else:
                        shutil.copyfile(src_path, dest_path)


if __name__ == "__main__":
    # split_idmt_4(os.path.join(DATA_DIR, "idmt_4"))
    # split_egfx(os.path.join(DATA_DIR, "egfx_clean"))
    # split_egfx(os.path.join(DATA_DIR, "egfx_phaser"))
    # split_egfx(os.path.join(DATA_DIR, "egfx_flanger"))
    # split_egfx(os.path.join(DATA_DIR, "egfx_chorus"))
    pass
