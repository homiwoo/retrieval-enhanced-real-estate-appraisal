from retrieval_enhanced_real_estate_appraisal.model.modules import mlp

import lightning as L
import torch


class Gate(L.LightningModule):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate = torch.nn.Sequential(
            mlp.MLP(
                input_dim,
                hidden_dim,
                activation_function="SELU",
                dropout=dropout,
            ),
            mlp.MLP(hidden_dim, 1, activation_function="Sigmoid", dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)
