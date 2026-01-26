from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

from .preprocessors import Preprocessor


@dataclass
class DataConfig:
    dataset: str = "mnist"
    data_path: str = "../data"
    n_train: int = 400
    n_test: int = 100
    digits: tuple[int, int] | None = None  # for digit filtering datasets
    preprocessors: Sequence[Union[str, Preprocessor]] = field(
        default_factory=lambda: ["bilinear_resize_4x4", "flatten"]
    )
    target_dim: int | None = None  # feature dimension enforcement (e.g., match qubits)
