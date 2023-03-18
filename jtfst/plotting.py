import logging
import os
from typing import Optional, List

import matplotlib.pyplot as plt
import torch as tr
from matplotlib.axes import Subplot
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def populate_subplot(subplot: Subplot,
                     data: T,
                     title: str = "scalogram",
                     freqs: Optional[List[float]] = None,
                     n_y_ticks: int = 8,
                     dt: Optional[float] = None,
                     n_x_ticks: int = 8,
                     remove_outliers: bool = True,
                     interpolation: str = "none",
                     cmap: str = "OrRd") -> None:
    assert data.ndim == 2
    if remove_outliers:
        mean = tr.mean(data)
        std = tr.std(data)
        data = tr.clip(data, mean - (4 * std), mean + (4 * std))

    data = data.detach().numpy()
    subplot.imshow(data, aspect="auto", interpolation=interpolation, cmap=cmap)
    subplot.set_title(title)

    if dt:
        x_pos = list(range(data.shape[1]))
        x_labels = [pos * dt for pos in x_pos]
        x_step_size = len(x_pos) // n_x_ticks
        x_pos = x_pos[::x_step_size]
        x_labels = x_labels[::x_step_size]
        x_labels = [f"{_:.3f}" for _ in x_labels]
        subplot.set_xticks(x_pos, x_labels)
        subplot.set_xlabel("time (s)")
    else:
        subplot.set_xlabel("samples")

    if freqs is not None:
        assert data.shape[0] == len(freqs)
        y_pos = list(range(len(freqs)))
        y_step_size = len(freqs) // n_y_ticks
        y_pos = y_pos[::y_step_size]
        y_labels = freqs[::y_step_size]
        y_labels = [f"{_:.0f}" for _ in y_labels]
        subplot.set_yticks(y_pos, y_labels)
        subplot.set_ylabel("freq (Hz)")


def plot_scalogram(scalogram: T,
                   dt: Optional[float] = None,
                   freqs: Optional[List[float]] = None,
                   title: str = "scalogram",
                   n_x_ticks: int = 8,
                   n_y_ticks: int = 8,
                   remove_outliers: bool = True,
                   interpolation: str = "none",
                   cmap: str = "OrRd") -> None:
    assert scalogram.ndim == 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), squeeze=True)
    populate_subplot(ax,
                     scalogram,
                     title=title,
                     freqs=freqs,
                     n_y_ticks=n_y_ticks,
                     dt=dt,
                     n_x_ticks=n_x_ticks,
                     remove_outliers=remove_outliers,
                     interpolation=interpolation,
                     cmap=cmap)
    plt.show()


def plot_3d_tensor(data: T,
                   n_cols: int = 2,
                   titles: Optional[List[str]] = None,
                   freqs: Optional[List[float]] = None,
                   n_y_ticks: int = 8,
                   dt: Optional[float] = None,
                   n_x_ticks: int = 8,
                   remove_outliers: bool = True,
                   interpolation: str = "none",
                   cmap: str = "OrRd") -> None:
    assert data.ndim == 3
    n_plots = data.size(0)
    if titles is not None:
        assert len(titles) == n_plots

    n_rows = n_plots // n_cols
    if n_plots % n_cols != 0:
        n_rows += 1

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

    for idx in range(n_plots):
        curr_ax = ax[idx // n_cols, idx % n_cols]
        title = None
        if titles is not None:
            title = titles[idx]

        pic = data[idx, :, :]
        populate_subplot(curr_ax,
                         pic,
                         title=title,
                         freqs=freqs,
                         n_y_ticks=n_y_ticks,
                         dt=dt,
                         n_x_ticks=n_x_ticks,
                         remove_outliers=remove_outliers,
                         interpolation=interpolation,
                         cmap=cmap)
    fig.tight_layout()
    plt.show()
