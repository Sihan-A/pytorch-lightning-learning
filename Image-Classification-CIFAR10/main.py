import py
from pytorch_lightning import Trainer
from utils import load_callbacks_logger
from data import CIFAR10DataModule
from model import LitResnet
from pytorch_lightning import seed_everything
from hparams import config

def train_test():
    seed_everything(config["RANDOM_SEED"])
    model = LitResnet(lr=config["LEARNING_RATE"])
    datamodule = CIFAR10DataModule(data_dir="./", batch_size=config["BATCH_SIZE"])
    callbacks, logger = load_callbacks_logger()
    trainer = Trainer(
    max_epochs=100,
    gpus=config["AVAIL_GPUS"],
    logger=logger,
    callbacks=callbacks,
)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)