import logging
import os

from lfo_tcn.cli import CustomLightningCLI
from lfo_tcn.paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    # config_name = "eval_lfo_quasi.yml"
    # config_name = "eval_lfo_distorted.yml"
    # config_name = "eval_lfo_combined.yml"
    # config_name = "eval_lfo_unseen_data.yml"
    config_name = "eval_lfo_val.yml"

    # config_name = "val_lfo_flanger.yml"
    # config_name = "val_lfo_phaser.yml"
    # config_name = "val_lfo_preproc.yml"
    # config_name = "val_lfo_dry_wet.yml"
    # config_name = "val_lfo_interwoven_all.yml"
    config_path = os.path.join(CONFIGS_DIR, config_name)
    cli = CustomLightningCLI(args=["validate", "--config", config_path],
                             trainer_defaults=CustomLightningCLI.trainer_defaults)
