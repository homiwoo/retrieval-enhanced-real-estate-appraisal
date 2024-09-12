import lightning as L
from abc import ABC
from typing import Dict
import torch
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)


class Appraiser(ABC):
    def __init__(
        self,
        log_on_step: bool = False,
        show_non_train_prog_bar: bool = True,
        loss: str = "MSE",
    ):
        super().__init__()
        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_abs_percentage_error = MeanAbsolutePercentageError()
        self.mean_squared_error = MeanSquaredError()
        self.log_on_step = log_on_step
        self.show_non_train_prog_bar = show_non_train_prog_bar
        self.loss = loss

    def batch_inference(self, batch) -> torch.Tensor:
        y_hat = self.forward(
            target_features=batch["target_features"].to(self.device),
            haversine_features=batch["haversine_features"].to(self.device),
            haversine_values=batch["haversine_values"].to(self.device),
            haversine_realtive_distance=batch["haversine_relative_features"].to(
                self.device
            ),
            vector_relative_distance=batch["neighbor_relative_features"].to(
                self.device
            ),
            vector_features=batch["neighbor_features"].to(self.device),
            vector_values=batch["neighbor_values"].to(self.device),
        ).view(-1)

        return y_hat

    def _step(self, batch) -> Dict:
        y = batch["target_value"]

        y_hat = self.batch_inference(batch)

        mse = self.mean_squared_error(y_hat, y)
        ae = (y_hat - y).abs()

        return {
            "MAE": ae.mean(),
            "MdAE": ae.median(),
            "MSE": mse,
        }

    def training_step(self, batch, batch_idx):
        metrics = self._step(batch)
        [
            self.log(
                f"Training {k}",
                metrics[k],
                on_step=self.log_on_step,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            for k in metrics.keys()
        ]
        return metrics[self.loss]

    def validation_step(self, batch, batch_idx):
        metrics = self._step(batch)
        [
            self.log(
                f"Val {k}",
                metrics[k],
                on_step=self.log_on_step,
                on_epoch=True,
                prog_bar=self.show_non_train_prog_bar,
                logger=True,
            )
            for k in metrics.keys()
        ]
        return metrics[self.loss]

    def test_step(self, batch, batch_idx):
        metrics = self._step(batch)
        [
            self.log(
                f"Test {k}",
                metrics[k],
                on_step=self.log_on_step,
                on_epoch=True,
                prog_bar=self.show_non_train_prog_bar,
                logger=True,
            )
            for k in metrics.keys()
        ]
        return metrics[self.loss]
