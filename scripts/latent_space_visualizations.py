import logging
import os

import torch as tr
import umap
from matplotlib import pyplot as plt

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    fl_latent = tr.load("../out/fl_1hz__latent.pt")
    # fl_latent = tr.load("../out/fl_rate__latent.pt")
    fl_labels = ["blue"] * fl_latent.size(0)
    ch_latent = tr.load("../out/ch_1hz__latent.pt")
    # ch_latent = tr.load("../out/ch_rate__latent.pt")
    ch_labels = ["red"] * ch_latent.size(0)
    ph_latent = tr.load("../out/ph_1hz__latent.pt")
    # ph_latent = tr.load("../out/ph_rate__latent.pt")
    ph_labels = ["cyan"] * ph_latent.size(0)

    labels = fl_labels + ch_labels + ph_labels
    latent = tr.cat([fl_latent, ch_latent, ph_latent], dim=0)

    # reducer = umap.UMAP()
    # latent = latent.numpy()
    # latent_2d = reducer.fit_transform(latent)
    latent_2d, _, _ = tr.pca_lowrank(latent, 2)
    colors = labels

    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, zorder=3)
    plt.grid(c="lightgray", zorder=0)
    plt.axis("equal")
    plt.show()
