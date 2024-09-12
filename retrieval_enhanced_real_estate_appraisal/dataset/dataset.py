from typing import Dict, List

import pandas as pd
import torch

from retrieval_enhanced_real_estate_appraisal.dataset.geographic_indexer import (
    GeographicIndexer,
)
from retrieval_enhanced_real_estate_appraisal.dataset.vector_indexer import (
    VectorIndexer,
)


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_path: str,
        features: List = [str(x) for x in range(23)],
        target_column: str = "target",
        use_flags: bool = True,
        embedding_size: int = 8,
        rerank_factor: int = 1,
        rerank_supplement: int = 0,
        n_vector_neighbors: int = 30,
        n_geographic_neighbors: int = 30,
        device: str = "cpu",
        sigma: int = 1,
    ):
        self.device = device
        self.sigma = sigma
        self.use_flags = use_flags
        self.rerank_factor = rerank_factor
        self.rerank_supplement = rerank_supplement
        self.target_column = target_column
        self.n_geographic_neighbors = n_geographic_neighbors
        self.n_vector_neighbors = n_vector_neighbors
        self.features = features
        self.df = pd.read_csv(
            dataset_path,
        )
        self.init_geographic_indexer()
        self.init_embeddings(embedding_size)
        self.indexer = VectorIndexer(
            n_neighbors=n_vector_neighbors,
            df=self.df,
            hidden_dim=8,
            device=self.device,
        )
        self.init_static_tensors()
        self.init_geographic_tensors()
        self.init_dynamic_tensors()

    def init_embeddings(self, size: int):
        self.embeddings = torch.zeros(size=(self.__len__(), size))

    def set_device(self, device):
        self.device = device

    def reset_indexer(
        self,
        embeddings: torch.Tensor = None,
        rerank_factor: int = 1,
        rerank_supplement: int = 0,
        **kwargs,
    ):
        self.embeddings = embeddings
        dim = embeddings.shape[-1]
        self.indexer = VectorIndexer(
            n_neighbors=self.n_vector_neighbors,
            hidden_dim=dim,
            df=self.df,
            embeddings=embeddings,
            device=self.device,
            **kwargs,
        )
        self.init_dynamic_tensors(rerank_factor, rerank_supplement)

    def init_static_tensors(self):
        self.feature_matrix = self.compute_feature_matrix()
        self.target_matrix = self.compute_target_matrix()

    def init_dynamic_tensors(self, rerank_factor: int = 1, rerank_supplement: int = 0):
        (
            self.neighbor_index_adjacency_matrix,
            self.neighbor_distance_adjacency_matrix,
        ) = self.indexer.compute_adjacency_matrix(
            df=self.df,
            embeddings=self.embeddings,
            X_index=self.geographic_index_complete_adjacency_matrix[
                :, : rerank_factor * self.n_vector_neighbors + rerank_supplement
            ],
            X_distances=self.geographic_distance_complete_adjacency_matrix[
                :, : rerank_factor * self.n_vector_neighbors + rerank_supplement
            ],
        )
        self.neighbor_distance_adjacency_matrix = (
            self.neighbor_distance_adjacency_matrix - self.distance_mean
        ) / self.distance_std
        if self.use_flags:
            self.flag_matrix = self.compute_flag_matrix(
                self.neighbor_index_adjacency_matrix
            )

    def init_geographic_indexer(self):
        n_neighbors = max(
            self.n_geographic_neighbors,
            self.n_vector_neighbors * self.rerank_factor + self.rerank_supplement,
        )
        indexer = GeographicIndexer(n_neighbors=n_neighbors)
        (
            self.geographic_index_complete_adjacency_matrix,
            self.geographic_distance_complete_adjacency_matrix,
        ) = indexer.compute_adjacency_matrix(df=self.df)
        self.geographic_index_adjacency_matrix = (
            self.geographic_index_complete_adjacency_matrix[
                :, : self.n_geographic_neighbors
            ]
        )
        self.geographic_distance_adjacency_matrix = (
            self.geographic_distance_complete_adjacency_matrix[
                :, : self.n_geographic_neighbors
            ]
        )

    def init_geographic_tensors(self):
        self.distance_mean = torch.mean(self.geographic_distance_adjacency_matrix)
        self.distance_std = torch.std(self.geographic_distance_adjacency_matrix)
        self.geographic_distance_adjacency_matrix = (
            self.geographic_distance_adjacency_matrix - self.distance_mean
        ) / self.distance_std
        self.geographic_index_adjacency_matrix = self.geographic_index_adjacency_matrix
        self.geographic_flag_matrix = self.compute_flag_matrix(
            self.geographic_index_adjacency_matrix
        )

    def compute_flag_matrix(self, adjacency) -> torch.Tensor:
        categories = ["r1", "r2", "r3", "r4"]
        matrices = [
            self.has_same_value(adjacency, column=category) for category in categories
        ]
        if len(matrices) > 0:
            return torch.stack(matrices, dim=-1)
        else:
            return torch.Tensor()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.tensor]:
        target_features = self.feature_matrix[index]
        neighbor_indexes = self.neighbor_index_adjacency_matrix[index]
        distances = self.neighbor_distance_adjacency_matrix[index]
        neighbor_features = self.feature_matrix[neighbor_indexes]
        target_value = self.target_matrix[index]
        vector_values = self.target_matrix[neighbor_indexes]
        vector_flags = self.flag_matrix[index]
        attention_mask = neighbor_indexes != -1
        neighbor_relative_distance = distances / (torch.max(distances) + 1e-3)
        neighbor_exp_distance = torch.exp(-distances * self.sigma)
        neighbor_relative_features = torch.concat(
            (
                vector_flags,
                distances.unsqueeze(-1),
                neighbor_relative_distance.unsqueeze(-1),
                neighbor_exp_distance.unsqueeze(-1),
            ),
            dim=-1,
        )
        tensors = {
            "index": index,
            "neighbor_indexes": neighbor_indexes.sort(),
            "neighbor_features": neighbor_features,
            "distances": distances,
            "target_value": target_value,
            "target_features": target_features,
            "neighbor_values": vector_values,
            "attention_mask": attention_mask,
            "neighbor_relative_features": neighbor_relative_features,
        }

        geographic_indexes = self.geographic_index_adjacency_matrix[index]
        geographic_distances = self.geographic_distance_adjacency_matrix[index]
        geographic_relative_distance = geographic_distances / (
            torch.max(geographic_distances) + 1e-3
        )
        geographic_exp_distance = torch.exp(-geographic_distances * self.sigma)
        geographic_values = self.target_matrix[geographic_indexes]
        geographic_features = self.feature_matrix[geographic_indexes]
        geographic_flags = self.geographic_flag_matrix[index]
        geographic_relative_features = torch.cat(
            (
                geographic_flags,
                geographic_distances.unsqueeze(-1),
                geographic_relative_distance.unsqueeze(-1),
                geographic_exp_distance.unsqueeze(-1),
            ),
            dim=-1,
        )

        tensors = {
            **tensors,
            "haversine_indexes": geographic_indexes.sort(),
            "haversine_distances": geographic_distances,
            "haversine_values": geographic_values,
            "haversine_features": geographic_features,
            "haversine_relative_features": geographic_relative_features,
        }

        return tensors

    def has_same_value(
        self, adjacency_matrix: torch.Tensor, column: str
    ) -> torch.Tensor:
        target = self.df[column].values
        neighbors = target[adjacency_matrix.cpu()]
        target = target.repeat(neighbors.shape[-1]).reshape(-1, neighbors.shape[-1])
        return torch.tensor(neighbors == target, dtype=int)

    def compute_feature_matrix(self) -> torch.Tensor:
        return torch.tensor(self.df[self.features].astype("float64").values).float()

    def compute_target_matrix(self) -> torch.Tensor:
        return torch.tensor(
            self.df[self.target_column].astype("float64").values
        ).float()
