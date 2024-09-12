from retrieval_enhanced_real_estate_appraisal.model.modules import mlp

import lightning as L
import torch


class BiEncoder(L.LightningModule):

    def __init__(self, feature_nb: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            mlp.MLP(
                feature_nb, hidden_dim, activation_function="SELU", dropout=dropout
            ),
            mlp.MLP(hidden_dim, hidden_dim, activation_function=None, dropout=dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)
