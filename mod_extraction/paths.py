import logging
import os

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../'))

CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUT_DIR = os.path.join(ROOT_DIR, "out")

assert os.path.isdir(DATA_DIR)
assert os.path.isdir(OUT_DIR)
