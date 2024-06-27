import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from models import CNN, CNNTest, GRUModel
from data import MyDataModule
from utils import random_seed
import torch
import wandb

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('medium')
    random_seed(42)
    wandb.login()
    print(cfg)
    # Set up TensorBoard logger
    tensorboard_logger = TensorBoardLogger("tb_logs", name="my_model")
    wandb_logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name)
    # Load model configuration
    if 'cnn_test' in cfg.model._target_:
        model = CNNTest(
            input_channels=cfg.model.input_channels,
            num_classes=cfg.model.num_classes,
            dropout_ratio=cfg.model.dropout_ratio,
            optimizer_cfg=cfg.optimizer,
            scheduler_cfg=cfg.scheduler
        )
    elif 'gru' in cfg.model._target_:
        model = GRUModel(
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            output_dim=cfg.model.output_dim,
            dropout=cfg.model.dropout,
            optimizer_cfg=cfg.optimizer,
            scheduler_cfg=cfg.scheduler
        )
    else:
        model = CNN(
            input_dim=cfg.model.input_dim,
            output_dim=cfg.model.output_dim,
            dropout_ratio=cfg.model.dropout_ratio,
            num_classes=cfg.model.num_classes,
            optimizer_cfg=cfg.optimizer,
            scheduler_cfg=cfg.scheduler
        )
    early_stop_callback = hydra.utils.instantiate(cfg.callbacks)
    datamodule = MyDataModule(cfg.data)


    trainer = pl.Trainer(
        logger=[tensorboard_logger, wandb_logger],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[early_stop_callback],
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()
