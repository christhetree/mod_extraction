import logging
import os
import pathlib
from typing import Dict, List, Optional

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
        if weights_path:
            assert os.path.isfile(weights_path)
            log.info(f"Loading effect model weights: {weights_path}")
            self.model.load_state_dict(tr.load(weights_path))
        self.prev_phase = tr.tensor(0.0)

    def make_argument(self, n_samples: int, freq: float, phase: float) -> Tensor:
        argument = tr.cumsum(2 * tr.pi * tr.full((n_samples,), freq) / self.sr, dim=0) + phase
        return argument

    def forward(self,
                x: Tensor,
                lfo_rate: Tensor,
                lfo_depth: Tensor,
                lfo_stereo_phase_offset: Tensor) -> Tensor:
        arg_l = self.make_argument(n_samples=x.size(-1), freq=lfo_rate.item(), phase=self.prev_phase.item())
        next_phase = arg_l[-1] % (2 * tr.pi)
        self.prev_phase = next_phase
        arg_r = arg_l + lfo_stereo_phase_offset.item()
        arg = tr.stack([arg_l, arg_r], dim=0)
        lfo = (tr.cos(arg) + 1.0) / 2.0
        lfo *= lfo_depth
        lfo = lfo.unsqueeze(1)
        x = self.model(x, lfo)
        return x


class EffectModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "Melda.phaser_quasi"
        # return "Melda.flanger_quasi"
        # return "Melda.phaser_irregular"
        # return "Melda.flanger_irregular"
        # return "EGFx.phaser"
        # return "EGFx.flanger"
        # return "EGFx.chorus"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "LFO extraction evaluation model."

    def get_model_long_description(self) -> str:
        return "LFO extraction evaluation model for 'Modulation Extraction for LFO-driven Audio Effects'."

    def get_technical_description(self) -> str:
        return "Wrapper for a simple conditional LSTM that models phaser, flanger, or chorus effects."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://christhetree.github.io/mod_extraction/",
            "Code": "https://github.com/christhetree/mod_extraction/",
        }

    def get_tags(self) -> List[str]:
        return ["lfo", "phaser", "flanger", "chorus"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("lfo_rate", "LFO rate [0.1 to 5 Hz]", default_value=0.2),
            NeutoneParameter("lfo_depth", "LFO depth [0.0, 1.5]", default_value=0.66666666),
            NeutoneParameter("lfo_stereo_phase_offset", "LFO stereo phase offset [0.0, 2pi]", default_value=0.0),
        ]

    @tr.jit.export
    def get_input_gain_default_value(self) -> float:
        """[0.0, 1.0] here maps to [-30.0db, +30.0db]"""
        return 0.4

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return False

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return False

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [44100]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        lfo_rate = (params["lfo_rate"] * 4.9) + 0.1
        lfo_depth = params["lfo_depth"] * 1.5
        lfo_stereo_phase_offset = params["lfo_stereo_phase_offset"] * 2 * tr.pi
        x = x.unsqueeze(1)
        x = self.model.forward(x, lfo_rate, lfo_depth, lfo_stereo_phase_offset)
        x = x.squeeze(1)
        return x


if __name__ == "__main__":
    weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_ph_quasi__epoch_241_step_803440.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_fl_quasi__epoch_207_step_690560.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_ph_irregular__epoch_199_step_664000.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__melda_fl_irregular__epoch_202_step_673960.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__egfx_ph_2_peak__epoch_35_step_95616.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__egfx_fl_2_peak__epoch_20_step_55776.pt")
    # weights_path = os.path.join(MODELS_DIR, "lstm_64__lfo_2dcnn_io_sa_25_25_no_ch_ln__egfx_ch_2_peak__epoch_40_step_108896.pt")
    model = EffectModel(weights_path=weights_path)
    wrapper = EffectModelWrapper(model)
    root_dir = pathlib.Path(os.path.join(OUT_DIR, "neutone_models", wrapper.get_model_name()))
    save_neutone_model(wrapper, root_dir, dump_samples=True, submission=True)
