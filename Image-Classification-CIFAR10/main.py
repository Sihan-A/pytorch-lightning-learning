import py
from pytorch_lightning import Trainer
from utils import load_callbacks_logger
from data import CIFAR10DataModule
from model import LitResnet
from pytorch_lightning import seed_everything
import torch
import os

RANDOM_SEED = 42
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)
LEARNING_RATE = 0.05

def train_test():
    seed_everything(RANDOM_SEED)
    model = LitResnet(lr=LEARNING_RATE)
    datamodule = CIFAR10DataModule(data_dir="./", batch_size=BATCH_SIZE)
    callbacks, logger = load_callbacks_logger()
    trainer = Trainer(
    max_epochs=100,
    gpus=AVAIL_GPUS,
    logger=logger,
    callbacks=callbacks,
)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)