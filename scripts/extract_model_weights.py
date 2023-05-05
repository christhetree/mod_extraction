import logging
import os

import torch as tr

from mod_extraction.cli import CustomLightningCLI
from mod_extraction.paths import MODELS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    model_dir = MODELS_DIR
    model_name = "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_ph_quasi__epoch_241_step_803440"
    # model_name = "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_fl_quasi__epoch_207_step_690560"
    # model_name = "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_ph_irregular__epoch_199_step_664000"
    # model_name = "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_fl_irregular__epoch_202_step_673960"
    # model_name = "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__egfx_ph_2_peak__epoch_35_step_95616"
    # model_name = "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__egfx_fl_2_peak__epoch_20_step_55776"
    # model_name = "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__egfx_ch_2_peak__epoch_40_step_108896"
    pt_module_attr_name = "effect_model"

    # model_name = "lfo_2dcnn_io_sa_25_25_no_ch_ln__ph_fl_ch_all_2__idmt_4__epoch_197_step_15840"
    # pt_module_attr_name = "model"

    config_path = os.path.join(model_dir, f"{model_name}.yml")
    ckpt_path = os.path.join(model_dir, f"{model_name}.ckpt")
    ckpt_data = tr.load(ckpt_path, map_location=tr.device('cpu'))

    cli = CustomLightningCLI(args=["-c", config_path],
                             trainer_defaults=CustomLightningCLI.trainer_defaults,
                             run=False)
    lm = cli.model
    assert hasattr(lm, pt_module_attr_name)
    model = getattr(lm, pt_module_attr_name)
    state_dict = {}
    for k, v in ckpt_data["state_dict"].items():
        model_tag = f"{pt_module_attr_name}."
        if model_tag in k:
            new_k = k.replace(model_tag, "")
            state_dict[new_k] = v

    model.load_state_dict(state_dict)
    save_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    tr.save(model.state_dict(), save_path)
