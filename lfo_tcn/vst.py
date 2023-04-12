import logging
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import torch as tr
import torch.nn as nn
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model
from torch import Tensor

from lfo_tcn.models import LSTMEffectModel
from lfo_tcn.paths import OUT_DIR, MODELS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class EffectModel(nn.Module):
    def __init__(self, weights_path: Optional[str] = None, n_hidden: int = 64, sr: float = 44100) -> None:
        super().__init__()
        self.sr = sr
        self.model = LSTMEffectModel(in_ch=1, out_ch=1, n_hidden=n_hidden, latent_dim=1)
        if weights_path and os.path.isfile(weights_path):
            log.info(f"Loading effect model weights: {weights_path}")
            self.model.load_state_dict(tr.load(weights_path))
        self.prev_phase = 0.0

    def make_lfo(self, n_samples: int, freq: float, phase: float) -> Tuple[Tensor, float]:
        argument = tr.cumsum(2 * tr.pi * tr.full((n_samples,), freq) / self.sr, dim=0) + phase
        next_phase = argument[-1] % (2 * tr.pi)
        mod_sig = (tr.cos(argument) + 1.0) / 2.0
        return mod_sig, next_phase

    def forward(self,
                x: Tensor,
                lfo_shape: Tensor,
                lfo_rate: Tensor,
                lfo_mult: Tensor,
                stereo_phase_offset: Tensor) -> Tensor:
        mod_sig, next_phase = self.make_lfo(n_samples=x.size(-1), freq=lfo_rate.item(), phase=self.prev_phase)
        mod_sig *= lfo_mult
        mod_sig = mod_sig.view(1, 1, -1)
        self.prev_phase = next_phase
        x = self.model(x, mod_sig)
        return x


class EffectModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        # return "EGFx.phaser"
        return "EGFx.flanger"
        # return "EGFx.chorus"
        # return "Melda.phaser_quasi"
        # return "Melda.flanger_quasi"
        # return "Melda.phaser_irregular"
        # return "Melda.flanger_irregular"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "TBD"

    def get_model_long_description(self) -> str:
        return "TBD"

    def get_technical_description(self) -> str:
        return "TBD"

    def get_technical_links(self) -> Dict[str, str]:
        return {
            # "Code": "TBD"
        }

    def get_tags(self) -> List[str]:
        return ["TBD"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("lfo_shape", "lfo shape [cos, tri]", default_value=0.0),
            NeutoneParameter("lfo_rate", "lfo rate [0.1 to 5 Hz]", default_value=0.1),
            NeutoneParameter("lfo_mult", "lfo multiplier [0.0, 2.0]", default_value=0.5),
            NeutoneParameter("stereo_phase_offset", "stereo phase offset [0.0, 2pi]", default_value=0.0),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return True

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return True

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [44100]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        lfo_shape = params["lfo_shape"]
        lfo_rate = (params["lfo_rate"] * 4.9) + 0.1
        lfo_mult = params["lfo_mult"] * 2.0
        stereo_phase_offset = params["stereo_phase_offset"]
        x = x.unsqueeze(1)
        x = self.model.forward(x, lfo_shape, lfo_rate, lfo_mult, stereo_phase_offset)
        x = x.squeeze(1)
        return x


if __name__ == "__main__":
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__egfx_ph_2_peak__epoch_35_step_95616.pt")
    weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__egfx_fl_2_peak__epoch_20_step_55776.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__egfx_ch_2_peak__epoch_40_step_108896.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_ph_quasi__epoch_241_step_803440.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_fl_quasi__epoch_207_step_690560.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_ph_irregular__epoch_199_step_664000.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_fl_irregular__epoch_202_step_673960.pt")
    model = EffectModel(weights_path=weights_path)
    wrapper = EffectModelWrapper(model)
    root_dir = pathlib.Path(os.path.join(OUT_DIR, "neutone_models", wrapper.get_model_name()))
    save_neutone_model(wrapper, root_dir, dump_samples=True, submission=True)
