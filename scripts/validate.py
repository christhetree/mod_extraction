import logging
import os

from lfo_tcn.cli import CustomLightningCLI
from lfo_tcn.paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    config_name = "val_lfo.yml"
    config_path = os.path.join(CONFIGS_DIR, config_name)
    cli = CustomLightningCLI(args=["validate", "--config", config_path],
                             trainer_defaults=CustomLightningCLI.trainer_defaults)
