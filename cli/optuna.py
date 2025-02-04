import optuna
import torch
import lightning as L
from functools import partial
from typing import Callable, Optional
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset, DistributedSampler

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.data.loader import DataLoader


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, train_dataset: Dataset, val_dataset: Optional[Dataset | list[Dataset]]):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset

        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    @staticmethod
    def get_dataloader(dataset: Dataset, dataloader_func: Callable[..., DataLoader], shuffle: bool, world_size: int,
                       batch_size: int, num_batches_per_epoch: Optional[int] = None) -> DataLoader:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=None,
                rank=None,
                shuffle=shuffle,
                seed=0,
                drop_last=False,
            ) if world_size > 1 else None
        )
        return dataloader_func(
            dataset=dataset,
            shuffle=shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            self.train_dataset,
            instantiate(self.cfg.train_dataloader, _partial_=True),
            self.cfg.train_dataloader.shuffle,
            self.trainer.world_size,
            self.train_batch_size,
            num_batches_per_epoch=self.train_num_batches_per_epoch,
        )

    def _val_dataloader(self) -> DataLoader | list[DataLoader]:
        return tree_map(
            partial(
                self.get_dataloader,
                dataloader_func=instantiate(self.cfg.val_dataloader, _partial_=True),
                shuffle=self.cfg.val_dataloader.shuffle,
                world_size=self.trainer.world_size,
                batch_size=self.val_batch_size,
                num_batches_per_epoch=None,
            ),
            self.val_dataset,
        )

    @property
    def train_batch_size(self) -> int:
        return self.cfg.train_dataloader.batch_size // (self.trainer.world_size * self.trainer.accumulate_grad_batches)

    @property
    def val_batch_size(self) -> int:
        return self.cfg.val_dataloader.batch_size // (self.trainer.world_size * self.trainer.accumulate_grad_batches)

    @property
    def train_num_batches_per_epoch(self) -> int:
        return self.cfg.train_dataloader.num_batches_per_epoch * self.trainer.accumulate_grad_batches


def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""

    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    dropout_p = trial.suggest_uniform("dropout_p", 0.0, 0.3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-3, 1e-1)

    # Modify config dynamically
    cfg.model.lr = lr
    cfg.model.dropout_p = dropout_p
    cfg.model.weight_decay = weight_decay

    if cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model: L.LightningModule = instantiate(cfg.model, _convert_="all")
    trainer: L.Trainer = instantiate(cfg.trainer)

    train_dataset = instantiate(cfg.data).load_dataset(model.train_transform_map)
    val_dataset = instantiate(cfg.val_data, _convert_="all").load_dataset(model.val_transform_map)

    L.seed_everything(cfg.seed + trainer.logger.version, workers=True)

    trainer.fit(model, datamodule=DataModule(cfg, train_dataset, val_dataset))

    val_loss = trainer.callback_metrics.get("val/PackedNLLLoss", torch.tensor(float("inf")))
    return val_loss.item()


@hydra.main(version_base="1.3", config_name="default.yaml")
def main(cfg: DictConfig):
    # Create Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)


if __name__ == "__main__":
    main()
