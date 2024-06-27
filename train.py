import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from models import CNN
from data import MyDataModule
from utils import random_seed
import torch

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    random_seed(42)
    print(cfg)
    # Load model configuration
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

    # Set up TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="my_model")

    trainer = pl.Trainer(
        logger=logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[early_stop_callback],
        max_epochs=cfg.trainer.max_epochs
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()
