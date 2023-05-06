import logging
import os

from mod_extraction.cli import CustomLightningCLI
from mod_extraction.paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    # config_name = "eval_lfo.yml"
    # config_name = "eval_lfo_unseen_audio.yml"
    # config_name = "eval_lfo_quasi.yml"
    # config_name = "eval_lfo_distorted.yml"
    # config_name = "eval_lfo_combined.yml"
    # config_name = "eval_lfo_rand.yml"  # TODO(cm)

    # config_name = "eval_em_unseen_effect.yml"

    config_name = "prototyping_lfo_dry_wet.yml"

    config_path = os.path.join(CONFIGS_DIR, config_name)
    cli = CustomLightningCLI(args=["validate", "--config", config_path],
                             trainer_defaults=CustomLightningCLI.trainer_defaults)
