import logging
import os

import torch as tr
from jsonargparse import lazy_instance
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from lfo_tcn.callbacks import LogSpecAndModSigCallback, LogAudioCallback

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class CustomLightningCLI(LightningCLI):
    trainer_defaults = {
        "accelerator": "gpu",
        "callbacks": [
            # TODO(cm): use text instead?
            LearningRateMonitor(logging_interval="step"),
            LogSpecAndModSigCallback(n_examples=4, log_wet_hat=True),
            # LogAudioCallback(n_examples=4, log_dry_audio=True),
            ModelCheckpoint(
                filename="epoch_{epoch}_step_{step}",  # Name is appended
                auto_insert_metric_name=False,
                monitor='val/loss',
                mode='min',
                save_last=True,
                save_top_k=1,
                verbose=False,
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
        parser.add_argument("custom.cpu_model_name", default="testing")
        parser.add_argument("custom.cpu_dataset_name", default="testing")
        parser.add_argument("custom.cpu_batch_size", default=5)
        parser.add_argument("custom.cpu_train_num_examples_per_epoch", default=15)
        parser.add_argument("custom.cpu_val_num_examples_per_epoch", default=10)
        parser.add_argument("custom.use_wandb", default=True)
        parser.link_arguments("custom.project_name", "trainer.logger.init_args.name")

        # parser.link_arguments("data.init_args.n_samples", "model.init_args.model.init_args.n_samples")  # TODO
        parser.link_arguments("data.init_args.shared_args.n_samples", "model.init_args.model.init_args.n_samples")  # TODO
        # parser.link_arguments("data.init_args.n_samples", "model.init_args.lfo_model.init_args.n_samples")  # TODO
        # parser.link_arguments("data.init_args.n_samples", "model.init_args.param_model.init_args.n_samples")  # TODO

        # parser.link_arguments("data.init_args.sr", "model.init_args.sr")
        parser.link_arguments("data.init_args.shared_args.sr", "model.init_args.sr")

    def before_instantiate_classes(self) -> None:
        if self.subcommand is not None:
            config = self.config[self.subcommand]
        else:
            config = self.config
        devices = config.trainer.devices
        if isinstance(devices, list):
            cuda_flag = f'{",".join([str(d) for d in devices])}'
            log.info(f"setting CUDA_VISIBLE_DEVICES = {cuda_flag}")
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_flag}"
            config.trainer.devices = len(devices)

        if config.trainer.devices < 2:
            log.info("Disabling strategy")
            config.trainer.strategy = None

        if not tr.cuda.is_available():
            config.custom.model_name = config.custom.cpu_model_name
            config.custom.dataset_name = config.custom.cpu_dataset_name
            config.trainer.accelerator = None
            config.trainer.devices = None
            config.trainer.strategy = None
            config.data.init_args.batch_size = config.custom.cpu_batch_size
            config.data.init_args.num_workers = 0
            # config.data.init_args.check_dataset = False  # TODO
            config.data.init_args.shared_args["check_dataset"] = False  # TODO
            # config.data.init_args.train_num_examples_per_epoch = config.custom.cpu_train_num_examples_per_epoch  # TODO
            config.data.init_args.shared_train_args["num_examples_per_epoch"] = config.custom.cpu_val_num_examples_per_epoch  # TODO
            # config.data.init_args.val_num_examples_per_epoch = config.custom.cpu_val_num_examples_per_epoch  # TODO
            config.data.init_args.shared_val_args["num_examples_per_epoch"] = config.custom.cpu_train_num_examples_per_epoch  # TODO
        else:
            assert not config.data.init_args.use_debug_mode

    def before_fit(self) -> None:
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.filename = f"{self.config.fit.custom.model_name}__" \
                              f"{self.config.fit.custom.dataset_name}__{cb.filename}"
                log.info(f"Setting checkpoint name to: {cb.filename}")

        if tr.cuda.is_available():
            if self.config.fit.custom.use_wandb:
                wandb_logger = WandbLogger(save_dir="wandb_logs",
                                           project=self.config.fit.custom.project_name,
                                           name=f"{self.config.fit.custom.model_name}__"
                                                f"{self.config.fit.custom.dataset_name}")
                self.trainer.loggers.append(wandb_logger)
            else:
                log.info("wandb is disabled")
        else:
            log.info("================ Running on CPU ================ ")

        log.info(f"================ {self.config.fit.custom.project_name} "
                 f"{self.config.fit.custom.model_name} "
                 f"{self.config.fit.custom.dataset_name} ================")
        log.info(f"================ Starting LR = {self.config.fit.optimizer.init_args.lr:.5f} ================ ")

    # def before_validate(self) -> None:
    #     tr.manual_seed(42)  # TODO(cm)
