import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
from torch.nn import functional as F

class GRUModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, optimizer_cfg, scheduler_cfg):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape from (batch_size, 1, 28, 28) to (batch_size, 28, 28)
        x = x.view(x.size(0), 28, 28)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_cfg, self.parameters())
        scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer)
        return [optimizer], [scheduler]
