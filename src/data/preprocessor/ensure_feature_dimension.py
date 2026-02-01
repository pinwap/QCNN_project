import torch
from torch import Tensor

from .base import BasePreprocessor


class EnsureFeatureDimension(BasePreprocessor):
    """Trim or zero-pad features to a fixed length (e.g., match qubit count)."""

    def __init__(self, target_dim: int):
        self.target_dim = target_dim

    def __call__(self, data: Tensor) -> Tensor:
        flat = data.view(data.shape[0], -1)
        current = flat.shape[1]
        if current == self.target_dim:
            return flat
        if current > self.target_dim:
            return flat[:, : self.target_dim]
        pad = torch.zeros(
            (flat.shape[0], self.target_dim - current), device=flat.device, dtype=flat.dtype
        )
        return torch.cat([flat, pad], dim=1)
