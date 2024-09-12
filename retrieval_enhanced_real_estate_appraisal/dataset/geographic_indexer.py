from math import radians
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def radian_vector(x):
    return np.vectorize(radians)(x)


class GeographicIndexer:

    def __init__(self, n_neighbors: int):
        self.n = n_neighbors
        self.index = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="haversine")

    def compute_adjacency_matrix(self, df: pd.DataFrame):
        PADDING = -1
        D, I = [], []

        for timestamp, subset in tqdm(
            df.groupby("date"), desc="Retrieving geographic neighbors."
        ):
            pool = df[df["date"] < timestamp][["lat", "lon"]].values
            if len(pool) > self.n:
                pool = radian_vector(pool)
                data = radian_vector(subset[["lat", "lon"]].values)
                distances, input_indexes = self.index.fit(pool).kneighbors(data, self.n)
                distances = distances * 6371
                D.append(torch.tensor(distances))
                I.append(torch.tensor(input_indexes))
            else:
                D.append(torch.tensor([[PADDING] * self.n] * len(subset)))
                I.append(torch.tensor([[PADDING] * self.n] * len(subset)))

        X_distances = torch.concat(D).float()
        X_index = torch.concat(I).int()

        return X_index, X_distances
