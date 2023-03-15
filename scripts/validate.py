import logging
import os

import yaml

from lfo_tcn.cli import CustomLightningCLI
from lfo_tcn.paths import MODELS_DIR, CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    model_dir = MODELS_DIR
    # model_name = "lstm_48_fr_l1_tbp_1024__ph_rate_egfx_clean__epoch_81_step_1176864"
    # model_name = "lstm_48_gt_l1_tbp_1024__ph_rate_egfx_clean__epoch_24_step_358800"
    # model_name = "lstm_48_gt_l1_tbp_1024__ph_rate_idmt_4__epoch_79_step_1060800"
    model_name = "lstm_48_fr_l1_tbp_1024__ph_rate_idmt_4__epoch_26_step_358020"

    config_path = os.path.join(model_dir, f"{model_name}.yml")
    ckpt_path = os.path.join(model_dir, f"{model_name}.ckpt")
    with open(config_path, "r") as in_f:
        config = yaml.safe_load(in_f)
    assert os.path.abspath(config["ckpt_path"]) == os.path.abspath(ckpt_path)

    # Manually override
    config_name = "val_lfo_tcn_phaser.yml"
    config_path = os.path.join(CONFIGS_DIR, config_name)

    cli = CustomLightningCLI(args=["validate", "-c", config_path],
                             trainer_defaults=CustomLightningCLI.trainer_defaults)
