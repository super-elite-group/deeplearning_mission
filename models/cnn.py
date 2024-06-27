import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

class CNN(pl.LightningModule):
    def __init__(self, input_dim, output_dim, dropout_ratio, num_classes, optimizer_cfg, scheduler_cfg):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout_ratio),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout_ratio),
        )

        self.fc_layer = nn.Linear(64 * 3 * 3, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(x.size(0), -1)
        pred = self.fc_layer(out)
        pred = self.softmax(pred)
        return pred

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.nll_loss(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log additional metrics
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.nll_loss(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log additional metrics
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.nll_loss(outputs, labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log additional metrics
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_cfg, self.parameters())
        scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer)
        return [optimizer], [scheduler]
