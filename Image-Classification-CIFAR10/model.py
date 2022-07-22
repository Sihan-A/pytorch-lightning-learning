from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from torchmetrics.functional import accuracy, confusion_matrix
from torchvision.models import resnet18
import torch
from hparams import config

def create_model(num_classes):
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        in_channels=3, out_channels=64, kernel_size=(3,3),
        stride=(1,1), padding=(1,1), bias=False)
    model.maxpool = nn.Identity()
    return model

class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    ### Evaluation ###
    def evaluate(self, batch, stage=None):
        """
        For validation_step and test_step
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    ### Optimizer ###
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4,)
        steps_per_epoch = 45000//config["BATCH_SIZE"]
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer=optimizer, max_lr=0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
