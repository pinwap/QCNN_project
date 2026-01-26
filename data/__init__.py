"""Central data package providing datasets, preprocessors, and a unified DataManager."""

from __future__ import annotations

from typing import Tuple, Optional
import logging

import torch

from .config import DataConfig
from .datasets import resolve_dataset, BaseDataset
from .preprocessors import (
    resolve_preprocessors,
    Preprocessor,
    EnsureFeatureDimension,
)

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, config: DataConfig):
        self.config = config
        self.dataset: BaseDataset = resolve_dataset(
            config.dataset,
            data_path=config.data_path,
            n_train=config.n_train,
            n_test=config.n_test,
            digits=config.digits,
        )
        self.preprocessors = resolve_preprocessors(config.preprocessors)
        if config.target_dim:
            already = any(isinstance(p, EnsureFeatureDimension) for p in self.preprocessors)
            if not already:
                self.preprocessors.append(EnsureFeatureDimension(config.target_dim))

    def _apply_preprocessors(self, data: torch.Tensor) -> torch.Tensor:
        processed = data
        for step in self.preprocessors:
            processed = step(processed)
        return processed

    def get_data(
        self,
        as_numpy: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        data, labels = self.dataset.load()
        data = self._apply_preprocessors(data)
        x_train, y_train, x_test, y_test = self.dataset.split(data, labels)

        if as_numpy:
            x_train = x_train.numpy()
            y_train = y_train.numpy()
            x_test = x_test.numpy()
            y_test = y_test.numpy()

        logger.info(
            "Data loaded: %s | Train %d, Test %d, Feature dim %d",
            self.config.dataset,
            x_train.shape[0],
            x_test.shape[0],
            x_train.shape[1],
        )
        return x_train, y_train, x_test, y_test


# Convenience backward-compatible aliases
class MNISTDataManager(DataManager):
    def __init__(self, data_path: str = "../data", n_train: int = 400, n_test: int = 100, preprocessors=None, target_dim: int = 16, digits: tuple[int, int] | None = None):
        cfg = DataConfig(
            dataset="mnist",
            data_path=data_path,
            n_train=n_train,
            n_test=n_test,
            preprocessors=preprocessors or ["bilinear_resize_4x4", "flatten"],
            target_dim=target_dim,
            digits=digits,
        )
        super().__init__(cfg)

__all__ = [
    "DataManager",
    "DataConfig",
    "Preprocessor",
    "EnsureFeatureDimension",
    "MNISTDataManager",
]
