from abc import ABC, abstractmethod

from torch import Tensor


class BasePreprocessor(ABC):
    @abstractmethod
    def __call__(self, data: Tensor) -> Tensor:
        """Transform input tensor and return it."""
        raise NotImplementedError
