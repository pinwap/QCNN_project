from torch import Tensor

from .base import BasePreprocessor


class Flatten(BasePreprocessor):
    def __call__(self, data: Tensor) -> Tensor:
        return data.view(data.shape[0], -1)
