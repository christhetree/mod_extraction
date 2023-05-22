import logging
import os

import torch

from mod_extraction.cli import CustomLightningCLI
from mod_extraction.paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    # Train a model for just one effect
    # config_name = "train_lfo_phaser.yml"
    # config_name = "train_lfo_flanger.yml"

    # To use this config, make preprocessed datasets for flanger and chorus effects called gen_idmt_fl and gen_idmt_ch.
    # This can be done by writing the batches of the dataloader of the previous config to disk.
    # See scripts/scratch.py for code to help do this.
    # config_name = "train_lfo_interwoven_all.yml"

    # Train effect models and baselines
    # config_name = "train_baseline_em_dry_wet.yml"
    config_name = "train_em_dry_wet.yml"

    config_path = os.path.join(CONFIGS_DIR, config_name)
    cli = CustomLightningCLI(args=["fit", "-c", config_path],
                             trainer_defaults=CustomLightningCLI.trainer_defaults)
