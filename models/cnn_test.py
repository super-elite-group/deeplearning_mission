import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import hydra

class CNNTest(pl.LightningModule):
    def __init__(self, input_channels, num_classes, dropout_ratio, optimizer_cfg, scheduler_cfg):
        super(CNNTest, self).__init__()
        self.num_classes = num_classes
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5), # 28 --> 24
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5), # 24 --> 20
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 20 --> 10
            nn.Dropout(dropout_ratio),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5), # 10 --> 6
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 6 --> 3
            nn.Dropout(dropout_ratio),
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3), # 3 --> 1
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        )

    def forward(self, x):
        out = self.layer(x)
        out = torch.squeeze(out)  # Remove dimensions of size 1
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log additional metrics
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log additional metrics
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
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
