from typing import Dict

import torch
from torch import nn


class MLP(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        activation_function: str = "ReLU",
        activation_kwargs: Dict = {},
    ):
        super().__init__()
        if activation_function is None:
            self.sequence = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Dropout(dropout),
            )
        else:
            self.sequence = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                getattr(nn, activation_function)(**activation_kwargs),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequence(x)
