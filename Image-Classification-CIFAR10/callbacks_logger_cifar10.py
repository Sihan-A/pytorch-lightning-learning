from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

def load_callbacks_logger():
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoint',
        filename='cifar10-{epoch:02d}-{val_loss:.2f}',)

    class LitProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
            bar = super().init_validation_tqdm()
            bar.set_description('running validation ...')
            return bar
    progress_bar = LitProgressBar(refresh_rate=10, process_position=0)

    callbacks = [checkpoint_callback, early_stopping, lr_monitor, progress_bar]
    logger = TensorBoardLogger("lightning_logs/", name="resnet")
    return callbacks, logger