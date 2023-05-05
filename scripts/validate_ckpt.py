import logging
import os

import yaml

from mod_extraction.cli import CustomLightningCLI
from mod_extraction.paths import MODELS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    model_dir = MODELS_DIR
    model_name = "lfo_2dcnn_io_sa_25_25_no_ch_ln__ph_fl_ch_all_2__idmt_4__epoch_197_step_15840"

    config_path = os.path.join(model_dir, f"{model_name}.yml")
    ckpt_path = os.path.join(model_dir, f"{model_name}.ckpt")
    with open(config_path, "r") as in_f:
        config = yaml.safe_load(in_f)
    if config.get("ckpt_path"):
        assert os.path.abspath(config["ckpt_path"]) == os.path.abspath(ckpt_path)

    cli = CustomLightningCLI(args=["validate", "--config", config_path, "--ckpt_path", ckpt_path],
                             trainer_defaults=CustomLightningCLI.trainer_defaults)
