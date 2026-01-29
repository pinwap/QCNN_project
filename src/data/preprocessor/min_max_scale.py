from torch import Tensor

from .base import BasePreprocessor


class MinMaxScale(BasePreprocessor):
    """Normalize data to [0, 1] range per sample."""

    def __call__(self, data: Tensor) -> Tensor:
        # data shape: (N, Features)
        min_val = data.min(dim=1, keepdim=True)[0]
        max_val = data.max(dim=1, keepdim=True)[0]
        range_val = max_val - min_val

        # Avoid division by zero
        range_val[range_val == 0] = 1.0

        return (data - min_val) / range_val
