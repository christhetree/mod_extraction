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
    model_name = "lfo_2dcnn_32_64_96_io__ph_all2_idmt_4_fl_ch_idmt_4_ibanez_fast__epoch_182_step_18300"

    config_path = os.path.join(model_dir, f"{model_name}.yml")
    ckpt_path = os.path.join(model_dir, f"{model_name}.ckpt")
    with open(config_path, "r") as in_f:
        config = yaml.safe_load(in_f)
    if config.get("ckpt_path"):
        assert os.path.abspath(config["ckpt_path"]) == os.path.abspath(ckpt_path)

    cli = CustomLightningCLI(args=["validate", "--config", config_path, "--ckpt_path", ckpt_path],
                             trainer_defaults=CustomLightningCLI.trainer_defaults)
