import torch
from torch import Tensor

from .base import BasePreprocessor

class PCAReducer(BasePreprocessor):
    def __init__(self, target_dim: int):
        self.target_dim = target_dim
        
    def MinMaxScale(self, data: Tensor) -> Tensor:
        min_val = data.min(dim=1, keepdim=True)[0]
        max_val = data.max(dim=1, keepdim=True)[0]
        range_val = max_val - min_val

        # Avoid division by zero
        range_val[range_val == 0] = 1.0

        return (data - min_val) / range_val

    def __call__(self, data: Tensor) -> Tensor:
        flat = data.view(data.shape[0], -1)
        u, s, vh = torch.pca_lowrank(flat, q=self.target_dim)
        # u: left singular vectors (shape: num_samples, q)
        # s: singular values (shape: q)
        # vh: right singular vectors (principal components/directions, shape: num_features, q)
        
        # Transform the data to the new principal components space
        projected = torch.matmul(flat - flat.mean(dim=0), vh[:, : self.target_dim])
        
        return self.MinMaxScale(projected)
