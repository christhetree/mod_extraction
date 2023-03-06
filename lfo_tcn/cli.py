import logging
import os

import torch
from jsonargparse import lazy_instance
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class CustomLightningCLI(LightningCLI):
    trainer_defaults = {
        "accelerator": "gpu",
        "callbacks": [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                filename="epoch_{epoch}_step_{step}",  # Name is appended
                auto_insert_metric_name=False,
                monitor='val/loss',
                mode='min',
                save_last=True,
                save_top_k=1,
                verbose=True
            ),
        ],
        "logger": {
            "class_path": "pytorch_lightning.loggers.TensorBoardLogger",
            "init_args": {
                "save_dir": "lightning_logs",
                "name": None,
            },
        },
        "log_every_n_steps": 1,
        "precision": 32,
        "strategy": lazy_instance(DDPStrategy, find_unused_parameters=False),
    }

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("custom.project_name", default="testing")
        parser.add_argument("custom.model_name", default="testing")
        parser.add_argument("custom.dataset_name", default="testing")
        parser.add_argument("custom.cpu_batch_size", default=1)
        parser.link_arguments("custom.project_name", "trainer.logger.init_args.name")

    def before_instantiate_classes(self) -> None:
        devices = self.config.fit.trainer.devices
        if isinstance(devices, int) and devices < 2:
            self.config.fit.trainer.strategy = None
        elif isinstance(devices, list) and len(devices) < 2:
            self.config.fit.trainer.strategy = None

        if not torch.cuda.is_available():
            self.config.fit.trainer.accelerator = None
            self.config.fit.trainer.devices = None
            self.config.fit.trainer.strategy = None
            self.config.fit.data.init_args.batch_size = self.config.fit.custom.cpu_batch_size
            self.config.fit.data.init_args.num_workers = 0

    def before_fit(self) -> None:
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.filename = f"{self.config.fit.custom.model_name}__" \
                              f"{self.config.fit.custom.dataset_name}__{cb.filename}"
                log.info(f"Setting checkpoint name to: {cb.filename}")

        if torch.cuda.is_available():
            wandb_logger = WandbLogger(save_dir="wandb_logs",
                                       project=self.config.fit.custom.project_name,
                                       name=f"{self.config.fit.custom.model_name}__"
                                            f"{self.config.fit.custom.dataset_name}")
            self.trainer.loggers.append(wandb_logger)
        else:
            log.info("================ Running on CPU ================ ")

        log.info(f"================ {self.config.fit.custom.project_name} "
                 f"{self.config.fit.custom.model_name} "
                 f"{self.config.fit.custom.dataset_name} ================")


def run_custom_cli(config_path: str) -> None:
    cli = CustomLightningCLI(args=["fit", "-c", config_path],
                             trainer_defaults=CustomLightningCLI.trainer_defaults)
