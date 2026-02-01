from torch import Tensor

from ..base import BasePreprocessor


class Identity(BasePreprocessor):
    def __call__(self, data: Tensor) -> Tensor:
        return data
