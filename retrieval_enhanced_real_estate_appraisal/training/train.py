import torch
import lightning as L
from typing import Tuple
from retrieval_enhanced_real_estate_appraisal.model.appraiser import Appraiser
from retrieval_enhanced_real_estate_appraisal.dataset.split import Split


def train(
    model_config: dict, training_config: dict, dataset_path: str
) -> Tuple[Appraiser, Split]:

    split = Split.from_file(
        dataset_path,
        n_geographic_neighbors=model_config["n_geographic_neighbors"],
        n_vector_neighbors=model_config["n_vector_neighbors"],
        features=model_config.pop("features"),
        rerank_factor=model_config["rerank_factor"],
        rerank_supplement=model_config["rerank_supplement"],
    )

    price_mean = torch.mean(split.dataset.target_matrix[split.train.indices])
    price_std = torch.std(split.dataset.target_matrix[split.train.indices])

    model_class = model_config.pop("class")

    model = model_class(
        **model_config,
        price_mean=price_mean,
        price_std=price_std,
        train_set=split.train,
        validation_set=split.validation,
    )

    model = fit(model, split, training_config)

    return model, split


def fit(model: Appraiser, split: Split, training_config: dict) -> Appraiser:

    train_loader = split.train_loader(batch_size=training_config["batch_size"])
    val_loader = split.validation_loader(batch_size=training_config["batch_size"])

    trainer = L.Trainer(
        max_epochs=training_config["epochs"],
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.fit(model, train_loader, val_loader)

    return model
