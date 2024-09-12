from retrieval_enhanced_real_estate_appraisal.model.rea import REA
from retrieval_enhanced_real_estate_appraisal.model.erea import EREA
from retrieval_enhanced_real_estate_appraisal.dataset.split import Split

from typing import Union
from tqdm import tqdm
import torch


def evaluate(
    model: Union[REA, EREA], split: Split, batch_size: int = 64
) -> torch.Tensor:
    """Computes MdABRE for the given model and split."""
    test_loader = split.test_loader(batch_size=batch_size)
    metrics = []
    for batch in tqdm(test_loader, desc="Running inferences."):
        y = batch["target_value"]
        y_hat = model.batch_inference(batch)
        metrics.append(abre(y, y_hat))
    return torch.median(torch.concat(metrics))


def abre(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Computes absolute relative error."""
    return abs(y - y_hat) / torch.min(y, y_hat)
