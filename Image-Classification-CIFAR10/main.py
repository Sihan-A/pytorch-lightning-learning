import py
from pytorch_lightning import Trainer
from utils import load_callbacks_logger
from data import CIFAR10DataModule
from model import LitResnet
from pytorch_lightning import seed_everything
from hparams import RANDOM_SEED, LEARNING_RATE, BATCH_SIZE, AVAIL_GPUS

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