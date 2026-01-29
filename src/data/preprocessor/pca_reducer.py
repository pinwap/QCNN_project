import torch
from torch import Tensor

from .base import BasePreprocessor
from .min_max_scale import MinMaxScale


class PCAReducer(BasePreprocessor):
    def __init__(self, target_dim: int):
        self.target_dim = target_dim

    def __call__(self, data: Tensor) -> Tensor:
        flat = data.view(data.shape[0], -1)
        u, s, vh = torch.pca_lowrank(flat, q=self.target_dim)
        # u: left singular vectors (shape: num_samples, q)
        # s: singular values (shape: q)
        # vh: right singular vectors (principal components/directions, shape: num_features, q)
        # Transform the data to the new principal components space
        projected = torch.matmul(flat - flat.mean(dim=0), vh[:, : self.target_dim])
        scaler = MinMaxScale()
        return scaler(projected)
