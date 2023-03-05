import logging
import os
from argparse import ArgumentParser
from typing import Dict, Tuple, Any

import auraloss
import pytorch_lightning as pl
import torch
import yaml
from matplotlib import pyplot as plt
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from erase_fx.jtfst.scattering_1d import ScatTransform1D
from erase_fx.models import AnalysisModel, LSTMEffectModel, Spectral1DCNNModel
from erase_fx.phaser_experiments import FlangerModule, RandomAudioChunkDataset
from erase_fx.tcn import causal_crop

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class EffectAnalysisLightning(pl.LightningModule):
    def __init__(self, **kwargs: Dict) -> None:
        super().__init__()
        self.save_hyperparameters()

        # self.analysis_model = AnalysisModel(
        #     kernel_size=self.hparams.tcn_kernel_size,
        #     latent_dim=self.hparams.tcn_latent_dim,
        #     norm_type=self.hparams.tcn_latent_norm_type,
        #     causal=False,
        # )
        # self.analysis_model = Spectral1DCNNModel(
        #     kernel_size=13, latent_dim=self.hparams.tcn_latent_dim,
        # )
        # self.effect_model = EffectModel(
        #     n_blocks=self.hparams.tcn_nblocks,
        #     channel_width=self.hparams.tcn_channel_width,
        #     kernel_size=self.hparams.tcn_kernel_size,
        #     dilation_growth=self.hparams.tcn_dilation_growth,
        #     latent_dim=self.hparams.tcn_latent_dim,
        #     causal=self.hparams.tcn_causal,
        # )
        self.effect_model = LSTMEffectModel(n_hidden=48, latent_dim=self.hparams.tcn_latent_dim)
        self.model = self.effect_model

        self.recon_losses = torch.nn.ModuleDict()
        for recon_loss in self.hparams.recon_losses:
            if recon_loss == "mrstft":
                self.recon_losses[recon_loss] = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[512, 2048, 8192],
                    hop_sizes=[256, 1024, 4096],
                    win_lengths=[512, 2048, 8192],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                )
            elif recon_loss == "l1":
                self.recon_losses[recon_loss] = torch.nn.L1Loss()
            elif recon_loss == "si-sdr":
                self.recon_losses[recon_loss] = auraloss.time.SISDRLoss(zero_mean=False)
            elif recon_loss == "dc":
                self.recon_losses[recon_loss] = auraloss.time.DCLoss()
            else:
                raise RuntimeError(f"Invalid reconstruction loss: {recon_loss}")

        self.sisdr = auraloss.time.SISDRLoss(zero_mean=False)
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 2048, 8192],
            hop_sizes=[256, 1024, 4096],
            win_lengths=[512, 2048, 8192],
            w_sc=0.0,
            w_phs=0.0,
            w_lin_mag=1.0,
            w_log_mag=1.0,
        )

    def forward(self, x: Tensor, latent: Tensor) -> Tensor:
        return self.effect_model(x, latent)

    def common_step(self, batch: Tuple[Tensor, ...], is_training: bool) -> (Tensor, Dict[str, Tensor]):
        prefix = "train" if is_training else "val"
        # dry, wet, effect_id, dry_lufs, wet_lufs = batch
        dry, wet, mod_sig = batch
        # TODO(cm): wet is not within [-1, 1]

        # generate conditioning signal
        # if self.hparams.tcn_condition:
        #     latent = self.analysis_model(wet)
        # else:
        #     assert False
        #     # use 0 conditioning when no conditioning network is present
        #     latent = torch.zeros((wet.size(0), self.hparams.tcn_latent_dim))
        #     latent = latent.type_as(wet)

        # process input audio with model
        # wet_hat = self.effect_model(dry, latent)

        # Use LFO ground truth
        mod_sig = mod_sig.unsqueeze(1)
        wet_hat = self.effect_model(dry, mod_sig)

        # crop x and y
        dry = causal_crop(dry, wet_hat.size(-1))
        wet = causal_crop(wet, wet_hat.size(-1))

        # compute loss
        loss = 0

        # compute loss on the waveform
        for loss_idx, (loss_name, loss_fn) in enumerate(self.recon_losses.items()):
            recon_loss = loss_fn(wet_hat, wet)
            loss += self.hparams.recon_loss_weights[loss_idx] * recon_loss

            self.log(
                f"{prefix}/{loss_name}",
                recon_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=self.hparams.batch_size,
            )

        # log the overall loss
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hparams.batch_size,
        )

        # log the SI-SDR error
        if "si-sdr" not in self.recon_losses:
            sisdr_error = -self.sisdr(wet_hat, wet)
            self.log(
                f"{prefix}/si-sdr",
                sisdr_error,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=self.hparams.batch_size,
            )

        # log the MR-STFT error
        # if "mrstft" not in self.recon_losses:
        #     mrstft_error = self.mrstft(wet_hat, wet)
        #     self.log(
        #         f"{prefix}/mrstft",
        #         mrstft_error,
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=False,
        #         logger=True,
        #         sync_dist=True,
        #         batch_size=self.hparams.batch_size,
        #     )

        # torchaudio.save("../out/dry.wav", dry[0], self.hparams.sample_rate)
        # torchaudio.save("../out/wet.wav", wet[0], self.hparams.sample_rate)
        # torchaudio.save("../out/wet_hat.wav", wet_hat[0], self.hparams.sample_rate)

        # for plotting down the line
        data_dict = {
            "x": dry.detach().float().cpu(),
            "y": wet.detach().float().cpu(),
            "y_hat": wet_hat.detach().float().cpu(),
            # "c": latent.detach().float().cpu(),
        }
        return loss, data_dict

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, _ = self.common_step(batch, is_training=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        loss, data_dict = self.common_step(batch, is_training=False)
        return data_dict

    def configure_optimizers(self):
        gen_opt = torch.optim.AdamW(
            # list(self.effect_model.parameters()) + list(self.analysis_model.parameters()),
            self.effect_model.parameters(),
            self.hparams.lr,
            (0.8, 0.99),
        )
        return gen_opt

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- Training  ---
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--checkpoint_dir", type=str, default="./")
        # --- Loss functions  ---
        # parser.add_argument("--recon_losses", nargs="+", default=["mrstft", "l1"])
        # parser.add_argument("--recon_loss_weights", nargs="+", default=[0.05, 0.95])
        parser.add_argument("--recon_losses", nargs="+", default=["l1"])
        parser.add_argument("--recon_loss_weights", nargs="+", default=[1.0])
        # --- Model ---
        parser.add_argument("--model_type", type=str, default="tcn")
        # --- TCN Model ---
        parser.add_argument("--tcn_nblocks", type=int, default=4)
        parser.add_argument("--tcn_dilation_growth", type=int, default=10)
        parser.add_argument("--tcn_kernel_size", type=int, default=13)
        parser.add_argument("--tcn_channel_width", type=int, default=32)
        parser.add_argument("--tcn_condition", action="store_true")
        parser.add_argument("--tcn_latent_dim", type=int, default=2)
        parser.add_argument("--tcn_latent_norm_type", type=str, default="batch")
        parser.add_argument("--tcn_estimate_loudness", action="store_true")
        parser.add_argument("--tcn_causal", action="store_true")
        return parser


class LFOLightning(pl.LightningModule):
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = AnalysisModel(
            kernel_size=self.hparams.tcn_kernel_size,
            latent_dim=self.hparams.tcn_latent_dim,
            norm_type=self.hparams.tcn_latent_norm_type,
            causal=False,
        )
        with open(self.hparams.effect_ranges_json, "r") as in_f:
            self.fx_config = yaml.safe_load(in_f)

        self.model = Spectral1DCNNModel(
            kernel_size=13,
        )
        self.flanger = FlangerModule(batch_size=self.hparams.batch_size,
                                     n_ch=1,
                                     n_samples=self.hparams.train_length,
                                     max_delay_ms=self.fx_config["flanger"]["max_delay_ms"],
                                     sr=self.hparams.sample_rate)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def common_step(self, batch: Tuple[Tensor, ...], is_training: bool) -> (Tensor, Dict[str, Tensor]):
        prefix = "train" if is_training else "val"
        # dry, wet, mod_sig = batch

        # Flanger
        dry, mod_sig = batch
        feedback = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["feedback"]["min"],
            self.fx_config["flanger"]["feedback"]["max"]
        )
        width = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["width"]["min"],
            self.fx_config["flanger"]["width"]["max"]
        )
        depth = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["depth"]["min"],
            self.fx_config["flanger"]["depth"]["max"]
        )
        mix = RandomAudioChunkDataset.sample_uniform(
            self.fx_config["flanger"]["mix"]["min"],
            self.fx_config["flanger"]["mix"]["max"]
        )
        wet = self.flanger(dry, mod_sig, feedback, width, depth, mix)

        # wet = dry

        # process input audio with model
        mod_sig_hat = self.model(wet).squeeze(1)  # STFT requirement

        mod_sig = F.interpolate(mod_sig.unsqueeze(1),
                                mod_sig_hat.size(-1),
                                mode="linear",
                                align_corners=True).squeeze(1)
        assert mod_sig.shape == mod_sig_hat.shape

        # import torchaudio
        # from torchaudio.transforms import Spectrogram
        # sr = 44100
        # spectrogram = Spectrogram(n_fft=512, power=1, center=True, normalized=False)
        # J = 10
        # Q = 16
        # scalogram = ScatTransform1D(sr=44100,
        #                             J=J,
        #                             Q=Q,
        #                             should_avg=True,
        #                             avg_win=256,
        #                             highest_freq=20000,
        #                             squeeze_channels=True)
        # for idx, (d, w, m) in enumerate(zip(dry, wet, mod_sig)):
        #     torchaudio.save(f"../out/dry_{idx}.wav", d, sr)
        #     torchaudio.save(f"../out/wet_{idx}.wav", w, sr)
        #     plt.plot(m.squeeze(0))
        #     plt.show()
        #
        #     scal, _, _ = scalogram(d.unsqueeze(0))
        #     from erase_fx.jtfst.plotting import plot_scalogram
        #     plot_scalogram(scal[0], freqs=scalogram.freqs_t, title="dry")
        #     scal, _, _ = scalogram(w.unsqueeze(0))
        #     plot_scalogram(scal[0], freqs=scalogram.freqs_t, title="wet")
        #
        #     spec = torch.log(spectrogram(d[0]))
        #     plt.imshow(spec, aspect="auto", interpolation="none")
        #     plt.show()
        #     spec = torch.log(spectrogram(w[0]))
        #     plt.imshow(spec, aspect="auto", interpolation="none")
        #     plt.show()
        # exit()

        # plt.plot(mod_sig[0].detach())
        # plt.show()
        # plt.plot(mod_sig_hat[0].detach())
        # plt.show()
        # exit()

        # compute loss
        mse = self.mse(mod_sig_hat, mod_sig)
        l1 = self.l1(mod_sig_hat, mod_sig)
        # loss = mse
        loss = l1

        # log the overall loss
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hparams.batch_size,
        )

        self.log(
            f"{prefix}/mse",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            f"{prefix}/l1",
            l1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hparams.batch_size,
        )
        # for plotting down the line
        data_dict = {
            "dry": dry.detach().float().cpu(),
            "wet": wet.detach().float().cpu(),
            "mod_sig": mod_sig.detach().float().cpu(),
            "mod_sig_hat": mod_sig_hat.detach().float().cpu(),
        }
        return loss, data_dict

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, _ = self.common_step(batch, is_training=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        loss, data_dict = self.common_step(batch, is_training=False)
        return data_dict

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            self.hparams.lr,
            (0.8, 0.99),
        )
        return opt

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- Training  ---
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--checkpoint_dir", type=str, default="./")
        # --- Model ---
        parser.add_argument("--model_type", type=str, default="tcn")
        # --- TCN Model ---
        parser.add_argument("--tcn_nblocks", type=int, default=4)
        parser.add_argument("--tcn_dilation_growth", type=int, default=10)
        parser.add_argument("--tcn_kernel_size", type=int, default=13)
        parser.add_argument("--tcn_channel_width", type=int, default=32)
        parser.add_argument("--tcn_condition", action="store_true")
        parser.add_argument("--tcn_latent_dim", type=int, default=2)
        parser.add_argument("--tcn_latent_norm_type", type=str, default="batch")
        parser.add_argument("--tcn_estimate_loudness", action="store_true")
        parser.add_argument("--tcn_causal", action="store_true")
        return parser
