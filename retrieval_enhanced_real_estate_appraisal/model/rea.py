from retrieval_enhanced_real_estate_appraisal.model.appraiser import Appraiser
import lightning as L
import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader

from retrieval_enhanced_real_estate_appraisal.dataset.dataset import Dataset
from retrieval_enhanced_real_estate_appraisal.model.modules.gated_attention import (
    GatedAttention,
)

from retrieval_enhanced_real_estate_appraisal.model.modules.bi_encoder import BiEncoder
from retrieval_enhanced_real_estate_appraisal.dataset.dataset import Dataset


class REA(Appraiser, L.LightningModule):

    def __init__(
        self,
        feature_nb: int,
        batch_size: int = 64,
        n_neighbors: int = 20,
        n_haversine_neighbors: int = 20,
        hidden_dim: int = 10,
        log_on_step: bool = True,
        show_non_train_prog_bar: bool = True,
        num_workers: int = 7,
        lr: float = 1e-3,
        train_set: Dataset = None,
        validation_set: Dataset = None,
        loss: str = "MSE",
        rerank_factor: int = 10,
        rerank_supplement: int = 0,
        dropout: float = 0.0,
        decay: float = 1.0,
        price_mean: float = 0,
        price_std: float = 1,
        use_vector: bool = True,
        use_geographic: bool = True,
        **kwargs,
    ):
        super().__init__(
            log_on_step=log_on_step,
            show_non_train_prog_bar=show_non_train_prog_bar,
            loss=loss,
        )
        self.price_mean = price_mean
        self.price_std = price_std
        self.lr = lr
        self.decay = decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rerank_factor = rerank_factor
        self.rerank_supplement = rerank_supplement
        self.n_neighbors = n_neighbors
        self.n_haversine_neighbors = n_haversine_neighbors
        self.ts = train_set
        self.vs = validation_set
        self.use_geographic = use_geographic
        self.use_vector = use_vector
        self.save_hyperparameters(ignore=["train_set", "validation_set", "scaler"])

        self.encoder = BiEncoder(feature_nb, hidden_dim, dropout)
        self.attention = GatedAttention()

    def prepare_features(
        self,
        haversine_features,
        haversine_values,
        vector_features,
        vector_values,
        **kwargs,
    ):
        features = []
        values = []

        if self.use_geographic:
            features.append(haversine_features)
            values.append(haversine_values)

        if self.use_vector:
            features.append(vector_features)
            values.append(vector_values)

        features = torch.cat(features, dim=1)
        values = torch.cat(values, dim=1).unsqueeze(-1)
        return features, values

    def forward(
        self,
        target_features,
        haversine_features,
        haversine_values,
        vector_features,
        vector_values,
    ):
        assert len(target_features.shape) == 2
        features, values = self.prepare_features(
            target_features,
            haversine_features,
            haversine_values,
            vector_features,
            vector_values,
        )

        encoded_features = self.encoder(features)
        encoded_target = self.encoder(target_features.unsqueeze(-2))

        aggregated_values = self.attention(
            encoded_target, encoded_features, values
        ).squeeze(-2)

        return aggregated_values

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        embeddings = self.compute_embeddings(self.ts.dataset)
        self.ts.dataset.reset_indexer(
            embeddings=embeddings,
            rerank_factor=self.rerank_factor,
            rerank_supplement=self.rerank_supplement,
        )
        self.optimizers().param_groups[0]["lr"] = (
            self.optimizers().param_groups[0]["lr"] * self.decay
        )

    def compute_embeddings(self, dataset: Dataset) -> torch.Tensor:
        """Computes all the embeddings using the full dataset & the model's encoder only."""
        self.encoder.eval()
        target_features = dataset.feature_matrix.to(self.device)
        embeddings = self.encoder(target_features)
        self.encoder.train()
        return embeddings.squeeze().detach().cpu()

    def configure_optimizers(self):
        groups = [
            {
                "params": self.encoder.parameters(),
                "lr": self.lr,
                "name": "encoder",
            },
        ]

        optimizer = optim.Adam(groups, lr=self.lr)
        return {"optimizer": optimizer}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ts, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.vs, batch_size=self.batch_size, num_workers=self.num_workers
        )
