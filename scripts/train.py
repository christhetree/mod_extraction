import logging
import os

import torch

from lfo_tcn.cli import run_custom_cli
from lfo_tcn.paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    # config_name = "train_lfo_tcn_phaser.yml"
    config_name = "train_lfo_tcn_flanger.yml"
    config_path = os.path.join(CONFIGS_DIR, config_name)
    run_custom_cli(config_path)
