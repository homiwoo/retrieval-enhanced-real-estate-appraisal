import torch


class GatedAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gates: torch.Tensor = None,
    ):
        alphas = self.attention_weights(query, key, gates)
        return torch.einsum("bnv,bqn->bqv", value, alphas)

    def attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        gates: torch.Tensor = None,
    ) -> torch.tensor:
        alphas = torch.einsum("bnd,bqd->bqn", key, query)
        if not gates is None:
            alphas = alphas * gates.transpose(-1, -2)
        alphas = torch.nn.functional.softmax(alphas, dim=-1)
        return alphas
