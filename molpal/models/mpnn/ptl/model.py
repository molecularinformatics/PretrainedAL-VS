import logging
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.nn import functional as F

from molpal.models import mpnn
from molpal.models.chemprop.nn_utils import NoamLR

# logging.getLogger("lightning").setLevel(logging.FATAL)


class LitMPNN(pl.LightningModule):
    """A message-passing neural network base class"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}

        self.mpnn = config.get("model", mpnn.MoleculeModel())
        self.uncertainty = config.get("uncertainty", "none")
        self.dataset_type = config.get("dataset_type", "regression")

        self.warmup_epochs = config.get("warmup_epochs", 2.0)
        self.max_epochs = config.get("max_epochs", 50)
        self.num_lrs = 1
        self.init_lr = config.get("init_lr", 1e-4)
        self.max_lr = config.get("max_lr", 1e-3)
        self.final_lr = config.get("final_lr", 1e-4)

        self.criterion = mpnn.utils.get_loss_func(self.dataset_type, self.uncertainty)
        self.metric = {
            "mse": lambda X, Y: F.mse_loss(X, Y, reduction="none"),
            "rmse": lambda X, Y: torch.sqrt(F.mse_loss(X, Y, reduction="none")),
        }.get(config.get("metric", "rmse"), "rmse")
        self.n_valid_steps = 0 

    def training_step(self, batch: Tuple, batch_idx) -> torch.Tensor:
        Xs, Y = batch

        mask = ~torch.isnan(Y)
        Y = torch.nan_to_num(Y, nan=0.0)
        class_weights = torch.ones_like(Y)

        Y_pred = self.mpnn(Xs)
        if self.uncertainty == "mve":
            Y_pred_mean = Y_pred[:, 0::2]
            Y_pred_var = Y_pred[:, 1::2]
            L = self.criterion(Y_pred_mean, Y_pred_var, Y)
        else:
            L = self.criterion(Y_pred, Y) * class_weights * mask
        output = L.sum() / mask.sum()
        # self.log('batch_train_loss', output, prog_bar=False, on_step=True)
        self.logger.log_metrics({"batch_train_loss": output}, step=self.global_step)
        return output

    def training_epoch_end(self, outputs):
        losses = [d["loss"] for d in outputs]
        # train_loss = losses
        train_loss = torch.stack(losses, dim=0).mean()
        # self.log("train_loss", train_loss, on_epoch=True)
        self.logger.log_metrics({"epoch_train_loss": train_loss}, step=self.current_epoch)

    def validation_step(self, batch: Tuple, batch_idx) -> List[float]:
        Xs, Y = batch

        Y_pred = self.mpnn(Xs)
        if self.uncertainty == "mve":
            Y_pred = Y_pred[:, 0::2]
        output = self.metric(Y_pred, Y)
        self.logger.log_metrics({"batch_val_loss": output.mean()}, step=self.n_valid_steps)
        self.n_valid_steps += 1
        return output

    def validation_epoch_end(self, outputs):
        val_loss = torch.cat(outputs).mean()
        """
        There is a confirmed bug here, where the current_epoch value stays at 0 for first two epochs
        Check the ref link:
        https://github.com/Lightning-AI/lightning/issues/3974
        """
        self.logger.log_metrics({"epoch_val_loss": val_loss}, step=self.current_epoch)
        self.log("val_loss", val_loss, logger=None)
        print(f"Epoch {self.current_epoch} Validation loss: {val_loss}")

    def configure_optimizers(self) -> List:
        opt = Adam([{"params": self.mpnn.parameters(), "lr": self.init_lr, "weight_decay": 0}])
        sched = NoamLR(
            optimizer=opt,
            warmup_epochs=[self.warmup_epochs],
            total_epochs=[self.trainer.max_epochs] * self.num_lrs,
            steps_per_epoch=self.num_training_steps,
            init_lr=[self.init_lr],
            max_lr=[self.max_lr],
            final_lr=[self.final_lr],
        )
        scheduler = {
            "scheduler": sched,
            "interval": "step" if isinstance(sched, NoamLR) else "batch",
        }

        return [opt], [scheduler]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())

        if isinstance(limit_batches, int):
            batches = min(batches, limit_batches)
        else:
            batches = int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        output = (batches // effective_accum) * self.trainer.max_epochs
        return output
