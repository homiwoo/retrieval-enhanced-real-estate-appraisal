from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DataLoader, Subset

from retrieval_enhanced_real_estate_appraisal.dataset.dataset import Dataset


@dataclass
class Split:
    train: Subset
    validation: Subset
    test: Subset
    remainder_set: Optional[Subset] = None
    num_workers: Optional[int] = 1

    @property
    def dataset(self) -> Dataset:
        return self.train.dataset

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.train, batch_size=batch_size, num_workers=self.num_workers
        )

    def validation_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.validation, batch_size=batch_size, num_workers=self.num_workers
        )

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.test, batch_size=batch_size, num_workers=self.num_workers
        )

    @classmethod
    def from_file(cls, path: str, **kwargs):

        dataset = Dataset(path, **kwargs)

        return cls(
            train=Subset(dataset, dataset.df[dataset.df["split"] == "train"].index),
            validation=Subset(
                dataset, dataset.df[dataset.df["split"] == "validation"].index
            ),
            test=Subset(dataset, dataset.df[dataset.df["split"] == "test"].index),
            remainder_set=Subset(
                dataset, dataset.df[dataset.df["split"] == "offset"].index
            ),
        )
