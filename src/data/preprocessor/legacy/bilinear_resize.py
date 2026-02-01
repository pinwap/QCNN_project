import torch
from torch import Tensor

from .base import BasePreprocessor


class BilinearResize(BasePreprocessor):
    """Resize input images to specific dimensions."""

    def __init__(self, size: tuple[int, int] = (4, 4)):
        self.size = size

    def __call__(self, data: Tensor) -> Tensor:
        # data shape: (N, 1, H, W) or (N, H, W)
        if data.dim() == 3:
            data = data.unsqueeze(1)  # add channel dimension for interpolation

        resized = torch.nn.functional.interpolate(
            data, size=self.size, mode="bilinear", align_corners=False
        )
        # Output shape: (N, 1, H_new, W_new) -> Flatten to (N, H*W)
        return resized.view(data.shape[0], -1)
