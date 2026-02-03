from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import torch


class EvaluationStrategy(ABC):
    """
    Abstract interface for model evaluation strategies (e.g., used by EA).
    A Strategy wraps an Engine with specific configurations.
    """

    @abstractmethod
    def evaluate(
        self,
        structure_code: List[int],
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Any]:
        """
        Evaluate a model structure and return its fitness score and trained model state.
        """
        pass
