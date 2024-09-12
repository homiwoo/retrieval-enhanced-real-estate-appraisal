import torch
import lightning as L
from torch import optim

from retrieval_enhanced_real_estate_appraisal.model.rea import REA
from retrieval_enhanced_real_estate_appraisal.model.modules.gate import Gate
from retrieval_enhanced_real_estate_appraisal.model.modules.adjuster import Adjuster


class EREA(REA, L.LightningModule):

    def __init__(
        self,
        feature_nb: int,
        hidden_dim: int,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__(
            feature_nb=feature_nb, hidden_dim=hidden_dim, dropout=dropout, **kwargs
        )
        input_dim = 2 * feature_nb + 8

        self.gate = Gate(input_dim, hidden_dim, dropout)
        self.adjuster = Adjuster(input_dim, hidden_dim, dropout)

    def prepare_features(
        self,
        target_features,
        haversine_features,
        haversine_values,
        vector_features,
        vector_values,
        haversine_realtive_distance,
        vector_relative_distance,
    ):
        features = []
        values = []
        gate_features = []

        if self.use_geographic:
            features.append(haversine_features)
            values.append(haversine_values)
            gate_features.append(
                torch.cat(
                    (
                        target_features.unsqueeze(-2).expand_as(haversine_features),
                        haversine_features,
                        haversine_realtive_distance,
                        (haversine_values.unsqueeze(-1) - self.price_mean)
                        / self.price_std,
                    ),
                    dim=-1,
                )
            )

        if self.use_vector:
            features.append(vector_features)
            values.append(vector_values)
            gate_features.append(
                torch.cat(
                    (
                        target_features.unsqueeze(-2).expand_as(vector_features),
                        vector_features,
                        vector_relative_distance,
                        (vector_values.unsqueeze(-1) - self.price_mean)
                        / self.price_std,
                    ),
                    dim=-1,
                )
            )

        features = torch.cat(features, dim=1)
        values = torch.cat(values, dim=1).unsqueeze(-1)
        gate_features = torch.cat(gate_features, dim=1)
        return features, values, gate_features

    def forward(
        self,
        target_features,
        haversine_features,
        haversine_values,
        vector_features,
        vector_values,
        haversine_realtive_distance,
        vector_relative_distance,
    ):
        features, values, gate_features = self.prepare_features(
            target_features,
            haversine_features,
            haversine_values,
            vector_features,
            vector_values,
            haversine_realtive_distance,
            vector_relative_distance,
        )

        encoded_features = self.encoder(features)
        encoded_target = self.encoder(target_features.unsqueeze(-2))

        gates = self.gate(gate_features)

        aggregated_values = self.attention(
            encoded_target, encoded_features, values, gates
        ).squeeze(-2)

        comparable_features = self.attention(
            encoded_target, encoded_features, gate_features, gates
        ).squeeze(-2)

        adjustment_alpha = self.adjuster(comparable_features)

        return aggregated_values * (1 + adjustment_alpha)

    def configure_optimizers(self):
        groups = [
            {
                "params": self.encoder.parameters(),
                "lr": self.lr,
                "name": "encoder",
            },
            {
                "params": self.gate.parameters(),
                "lr": self.lr,
                "name": "gate",
            },
            {
                "params": self.adjuster.parameters(),
                "lr": self.lr,
                "name": "adjustment_decoder",
            },
        ]

        optimizer = optim.Adam(groups, lr=self.lr)
        return {"optimizer": optimizer}
