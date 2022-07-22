import py
from pytorch_lightning import Trainer
from utils import load_callbacks_logger
from data import CIFAR10DataModule
from model import LitResnet
from pytorch_lightning import seed_everything
from hparams import config

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='config', config_name='default')
def train_test(cfg: DictConfig):
    seed_everything(cfg.seed)
    
    model = LitResnet(lr=cfg.model.lr)
    datamodule = CIFAR10DataModule(
        data_dir="./",
        batch_size=config["BATCH_SIZE"],
    )
    callbacks, logger = load_callbacks_logger()
    trainer = Trainer(
        max_epochs=100,
        gpus=cfg.train.num_gpus,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
