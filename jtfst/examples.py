import logging
import os
import time

import torch as tr
import torchaudio
from torch import Tensor as T
from tqdm import tqdm

from jtfst.jtfst import JTFST1D, JTFST2D
from jtfst.plotting import plot_scalogram, plot_3d_tensor
from jtfst.scattering_1d import ScatTransform1D, ScatTransform1DSubsampling, ScatTransform1DJagged
from jtfst.signals import make_pure_sine, make_pulse, make_exp_chirp
from jtfst.wavelets import DiscreteWavelet

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

tr.backends.cudnn.benchmark = True
log.info(f'tr.backends.cudnn.benchmark = {tr.backends.cudnn.benchmark}')

GPU_IDX = 0

if tr.cuda.is_available():
    log.info(f"tr.cuda.device_count() = {tr.cuda.device_count()}")
    tr.cuda.set_device(GPU_IDX)
    log.info(f"tr.cuda.current_device() = {tr.cuda.current_device()}")
    device = tr.device(f"cuda:{GPU_IDX}")
else:
    log.info(f"No GPUs found")
    device = tr.device("cpu")

log.info(f"device = {device}")


def load_audio(path: str = "../data/flute.wav",
               n_samples: int = 2 ** 16,
               start_n: int = 0) -> (T, int):
    audio, sr = torchaudio.load(path)
    log.info(f"Loaded audio shape = {audio.shape}")
    audio = audio[:, start_n:n_samples]
    audio = tr.mean(audio, dim=0)
    return audio, sr


def make_chirp_pulse_sine_audio(n_samples: int = 2 ** 16, sr: int = 48000) -> (T, int):
    audio_1 = make_pure_sine(n_samples, sr, freq=4000, amp=1.0)
    audio_2 = make_pulse(n_samples, center_loc=0.5, dur_samples=128, amp=4.0)
    audio_3 = make_exp_chirp(n_samples, sr, start_freq=20, end_freq=20000, amp=1.0)
    audio = audio_1 + audio_2 + audio_3
    audio = audio.view(1, 1, -1)
    return audio, sr


def time_scat_1st_order_example() -> None:
    log.info("time_scat_1st_order_example")
    audio, sr = make_chirp_pulse_sine_audio()
    log.info(f"audio.shape = {audio.shape}")
    log.info(f"sr = {sr}")

    J_1 = 8
    Q_1 = 12
    should_avg = True
    avg_win = 2 ** 8
    highest_freq = None

    scat_1d_regular = ScatTransform1D(sr,
                                      J_1,
                                      Q_1,
                                      should_avg=should_avg,
                                      avg_win=avg_win,
                                      highest_freq=highest_freq,
                                      squeeze_channels=True)
    scat_1d_jagged = ScatTransform1DJagged(sr,
                                           J_1,
                                           Q_1,
                                           should_avg=should_avg,
                                           avg_win=avg_win,
                                           highest_freq=highest_freq,
                                           should_pad=True)
    scat_1d_subsampling = ScatTransform1DSubsampling(sr, J_1, Q_1, squeeze_channels=True)

    scalogram, freqs, _ = scat_1d_regular(audio)
    log.info(f"Regular: scalogram shape = {scalogram.shape}")
    log.info(f"Regular: scalogram energy = {DiscreteWavelet.calc_energy(scalogram):.2f}")
    log.info(f"Regular: highest freq  = {freqs[0]:.2f} Hz")
    log.info(f"Regular: lowest  freq  = {freqs[-1]:.2f} Hz")
    plot_scalogram(scalogram[0], title="scalogram regular", freqs=freqs, n_y_ticks=J_1)

    scalogram, freqs, _ = scat_1d_subsampling(audio)
    log.info(f"Subsampling: scalogram shape = {scalogram.shape}")
    log.info(f"Subsampling: scalogram energy = {DiscreteWavelet.calc_energy(scalogram):.2f}")
    log.info(f"Subsampling: highest freq  = {freqs[0]:.2f} Hz")
    log.info(f"Subsampling: lowest  freq  = {freqs[-1]:.2f} Hz")
    plot_scalogram(scalogram[0], title="scalogram subsampled", freqs=freqs, n_y_ticks=J_1)

    scalogram, freqs, _ = scat_1d_jagged(audio, [sr])
    scalogram = tr.cat(scalogram, dim=1)
    log.info(f"Jagged: scalogram shape = {scalogram.shape}")
    log.info(f"Jagged: scalogram energy = {DiscreteWavelet.calc_energy(scalogram):.2f}")
    log.info(f"Jagged: highest freq  = {freqs[0]:.2f} Hz")
    log.info(f"Jagged: lowest  freq  = {freqs[-1]:.2f} Hz")
    plot_scalogram(scalogram[0], title="scalogram jagged", freqs=freqs, n_y_ticks=J_1)


def time_scat_2nd_order_example() -> None:
    log.info("time_scat_2nd_order_example")
    audio, sr = make_chirp_pulse_sine_audio()
    log.info(f"audio.shape = {audio.shape}")
    log.info(f"sr = {sr}")

    J_1 = 12
    Q_1 = 16
    J_2 = 12
    Q_2 = 1
    should_avg_2 = False
    avg_win_2 = 2 ** 11
    highest_freq_2 = sr / 4  # Half of nyquist

    scat_1st_order = ScatTransform1D(sr,
                                     J_1,
                                     Q_1,
                                     should_avg=False,
                                     squeeze_channels=True)
    scat_2nd_order = ScatTransform1DJagged(sr,
                                           J_2,
                                           Q_2,
                                           should_avg=should_avg_2,
                                           avg_win=avg_win_2,
                                           highest_freq=highest_freq_2,
                                           should_pad=True)
    scalogram, freqs_1, _ = scat_1st_order(audio)
    log.info(f"scalogram shape = {scalogram.shape}")
    log.info(f"highest freq 1  = {freqs_1[0]:.2f} Hz")
    log.info(f"lowest  freq 1  = {freqs_1[-1]:.2f} Hz")
    plot_scalogram(scalogram[0], title="scalogram", freqs=freqs_1, n_y_ticks=J_1)

    start_t = time.perf_counter()
    y_2, freqs_2, orientations = scat_2nd_order(scalogram, freqs_1)
    end_t = time.perf_counter()
    log.info(f"elapsed time = {end_t - start_t:.2f} seconds")

    y_2 = tr.stack(y_2, dim=1)
    log.info(f"y_2 shape = {y_2.shape}")
    log.info(f"highest freq 2  = {freqs_2[0]:.2f} Hz")
    log.info(f"lowest  freq 2  = {freqs_2[-1]:.2f} Hz")

    titles = [f"freq_2 = {freq_2:.0f}, theta = {theta}" for freq_2, theta in zip(freqs_2, orientations)]
    plot_3d_tensor(y_2[0], n_cols=2, titles=titles, freqs=freqs_1, n_y_ticks=J_1)


def jtfst_example() -> None:
    log.info("jtfst_example")
    audio, sr = make_chirp_pulse_sine_audio()
    log.info(f"audio.shape = {audio.shape}")
    log.info(f"sr = {sr}")

    J_1 = 12
    J_2_f = 2
    J_2_t = 12
    Q_1 = 16
    Q_2_f = 1
    Q_2_t = 1
    should_avg_f = False
    should_avg_t = True
    avg_win_f = 4  # Average across 25% of an octave if Q_1 == 16
    avg_win_t = 2 ** 9
    reflect_f = True

    jtfst_2d = JTFST2D(sr,
                       J_1=J_1,
                       J_2_f=J_2_f,
                       J_2_t=J_2_t,
                       Q_1=Q_1,
                       Q_2_f=Q_2_f,
                       Q_2_t=Q_2_t,
                       should_avg_f=should_avg_f,
                       should_avg_t=should_avg_t,
                       avg_win_f=avg_win_f,
                       avg_win_t=avg_win_t,
                       reflect_f=reflect_f)
    start_t = time.perf_counter()
    scalogram, freqs_1, jtfst, freqs_2 = jtfst_2d(audio)
    end_t = time.perf_counter()
    log.info(f"elapsed time = {end_t - start_t:.2f} seconds")

    log.info(f"jtfst_2d scalogram shape = {scalogram.shape}")
    log.info(f"jtfst_2d highest freq 1  = {freqs_1[0]:.2f} Hz")
    log.info(f"jtfst_2d lowest  freq 1  = {freqs_1[-1]:.2f} Hz")
    plot_scalogram(scalogram[0], title="jtfst_2d scalogram", freqs=freqs_1, n_y_ticks=J_1)
    log.info(f"jtfst_2d shape = {jtfst.shape}")
    log.info(f"jtfst_2d highest freq 2  = {freqs_2[0][1]:.2f} Hz")
    log.info(f"jtfst_2d lowest  freq 2  = {freqs_2[-1][1]:.2f} Hz")
    titles = [f"jtfst_2d, freq_f = {freq_f:.0f}, freq_t = {freq_t:.0f}, theta = {theta}"
              for freq_f, freq_t, theta in freqs_2]
    plot_3d_tensor(jtfst[0], n_cols=2, titles=titles, freqs=freqs_1, n_y_ticks=J_1)

    jtfst_1d = JTFST1D(sr,
                       J_1=J_1,
                       J_2_f=J_2_f,
                       J_2_t=J_2_t,
                       Q_1=Q_1,
                       Q_2_f=Q_2_f,
                       Q_2_t=Q_2_t,
                       should_avg_f=should_avg_f,
                       should_avg_t=should_avg_t,
                       avg_win_f=avg_win_f,
                       avg_win_t=avg_win_t,
                       reflect_f=reflect_f)
    start_t = time.perf_counter()
    scalogram, freqs_1, jtfst, freqs_2 = jtfst_1d(audio)
    end_t = time.perf_counter()
    log.info(f"elapsed time = {end_t - start_t:.2f} seconds")

    log.info(f"jtfst_1d scalogram shape = {scalogram.shape}")
    log.info(f"jtfst_1d highest freq 1  = {freqs_1[0]:.2f} Hz")
    log.info(f"jtfst_1d lowest  freq 1  = {freqs_1[-1]:.2f} Hz")
    plot_scalogram(scalogram[0], title="jtfst_1d scalogram", freqs=freqs_1, n_y_ticks=J_1)
    log.info(f"jtfst_1d shape = {jtfst.shape}")
    log.info(f"jtfst_1d highest freq 2  = {freqs_2[0][1]:.2f} Hz")
    log.info(f"jtfst_1d lowest  freq 2  = {freqs_2[-1][1]:.2f} Hz")
    titles = [f"jtfst_1d, freq_f = {freq_f:.0f}, freq_t = {freq_t:.0f}, theta = {theta}"
              for freq_f, freq_t, theta in freqs_2]
    plot_3d_tensor(jtfst[0], n_cols=2, titles=titles, freqs=freqs_1, n_y_ticks=J_1)


def gpu_example() -> None:
    log.info("gpu_example")
    if not tr.cuda.is_available():
        log.info("No GPU available!")
        return

    bs = 1
    n_iter = 100

    audio, sr = make_chirp_pulse_sine_audio()
    log.info(f"audio.shape = {audio.shape}")
    log.info(f"sr = {sr}")

    J_1 = 12
    J_2_f = 2
    J_2_t = 12
    Q_1 = 16
    Q_2_f = 1
    Q_2_t = 1
    should_avg_f = False
    should_avg_t = True
    avg_win_f = 4  # Average across 25% of an octave if Q_1 == 16
    avg_win_t = 2 ** 9
    reflect_f = True

    # jtfst_module = JTFST1D
    jtfst_module = JTFST2D
    jtfst_model = jtfst_module(sr,
                               J_1=J_1,
                               J_2_f=J_2_f,
                               J_2_t=J_2_t,
                               Q_1=Q_1,
                               Q_2_f=Q_2_f,
                               Q_2_t=Q_2_t,
                               should_avg_f=should_avg_f,
                               should_avg_t=should_avg_t,
                               avg_win_f=avg_win_f,
                               avg_win_t=avg_win_t,
                               reflect_f=reflect_f)

    jtfst_model.to(device)
    batch = audio.repeat(bs, 1, 1)
    batch = batch.to(device)

    for _ in tqdm(range(n_iter)):
        scalogram, freqs_1, jtfst, freqs_2 = jtfst_model(batch)


if __name__ == "__main__":
    time_scat_1st_order_example()
    time_scat_2nd_order_example()
    jtfst_example()
    gpu_example()
