from typing import Tuple
import torch


class VectorIndexer:
    def __init__(
        self, n_neighbors: int, rerank_factor: int = 1, device: str = "cpu", **kwargs
    ):
        self.n_neighbors = n_neighbors
        self.rerank_factor = rerank_factor
        self.device = device

    def compute_adjacency_matrix(
        self,
        X_index: torch.Tensor,
        X_distances: torch.Tensor,
        embeddings: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Filter nearest neighbor by embedding.
        X_index = X_index.to(self.device)
        X_distances = X_distances.to(self.device)
        comparable_embeddings = embeddings[X_index]
        X_similarity = torch.einsum(
            "lkd, ld -> lk", comparable_embeddings, embeddings
        ).to(self.device)
        indexes = torch.argsort(X_similarity, dim=1, descending=True).to(self.device)
        X_index = torch.gather(X_index, 1, indexes)[:, : self.n_neighbors]
        X_distances = torch.gather(X_distances, 1, indexes)[:, : self.n_neighbors]

        return X_index, X_distances
