import click
import torch
from pathlib import Path

from retrieval_enhanced_real_estate_appraisal.model.rea import REA
from retrieval_enhanced_real_estate_appraisal.model.erea import EREA
from retrieval_enhanced_real_estate_appraisal.evaluation.evaluate import evaluate
from retrieval_enhanced_real_estate_appraisal.training.train import train

DATASET_PATH = Path(__file__).parent / "data/iv.csv"


@click.command()
@click.option("--dataset-path", default=DATASET_PATH)
@click.option("--batch-size", default=64)
@click.option("--epochs", default=25)
@click.option("--features", default=[str(x) for x in range(23)], multiple=True)
@click.option("--model-class", default="EREA")
@click.option("--decay", default=1)
@click.option("--rerank-supplement", default=25)
@click.option("--rerank-factor", default=3)
@click.option("--lr", default=1e-3)
@click.option("--hidden-dim", default=10)
@click.option("--n-vector-neighbors", default=30)
@click.option("--n-geographic-neighbors", default=30)
@click.option("--seed", default=1)
def main(
    dataset_path: str,
    batch_size: int,
    epochs: int,
    features: list,
    model_class: str,
    decay: float,
    rerank_supplement: int,
    rerank_factor: int,
    lr: float,
    hidden_dim: int,
    n_vector_neighbors: int,
    n_geographic_neighbors: int,
    seed: int,
):
    torch.manual_seed(seed)

    training_config = {
        "batch_size": batch_size,
        "epochs": epochs,
    }

    model_config = {
        "features": list(features),
        "feature_nb": len(features),
        "class": model_factory(model_class),
        "decay": decay,
        "rerank_supplement": rerank_supplement,
        "rerank_factor": rerank_factor,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "n_vector_neighbors": n_vector_neighbors,
        "n_geographic_neighbors": n_geographic_neighbors,
    }

    model, split = train(model_config, training_config, dataset_path)
    error = evaluate(model, split)
    click.echo(f"Model achieved {round(error.item() * 100, 2)}% MdABRE.")


def model_factory(name: str):
    if name == "EREA":
        return EREA
    elif name == "REA":
        return REA
    raise ValueError(f"Unkown model class: {name}.")


if __name__ == "__main__":
    main()
