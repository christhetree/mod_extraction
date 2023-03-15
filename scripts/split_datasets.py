import logging
import os
import random
import shutil

import torchaudio
from tqdm import tqdm

from lfo_tcn.paths import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


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


def split_egfx_from_existing(root_dir: str, new_name: str, prev_name: str = "clean") -> None:
    existing_paths = []
    for root_dir_2, _, file_names in os.walk(os.path.join(root_dir)):
        for file_name in file_names:
            if file_name.endswith(".wav") and not file_name.startswith("."):
                existing_paths.append(os.path.join(root_dir_2, file_name))

    dest_paths = list(existing_paths)
    dest_paths = [p.replace(prev_name, new_name) for p in dest_paths]
    src_paths = list(dest_paths)
    src_paths = [p.replace("train/", "") for p in src_paths]
    src_paths = [p.replace("val/", "") for p in src_paths]

    for src_path, dest_path in zip(src_paths, dest_paths):
        assert os.path.basename(src_path) == os.path.basename(dest_path)
        os.makedirs(os.path.dirname(src_path), exist_ok=True)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copyfile(src_path, dest_path)


def split_idmt_4(root_dir: str, val_split: float = 0.3, offset_n_bars: int = 3) -> None:
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
        song_names = list(song_names)
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


if __name__ == "__main__":
    # split_egfx(os.path.join(DATA_DIR, "egfx_phaser"))
    # split_idmt_4(os.path.join(DATA_DIR, "idmt_4"), 0.25)
    # split_egfx_from_existing(os.path.join(DATA_DIR, "egfx_clean"), "phaser")
    # split_egfx_from_existing(os.path.join(DATA_DIR, "egfx_clean"), "flanger")
    # split_egfx_from_existing(os.path.join(DATA_DIR, "egfx_clean"), "chorus")
    pass
