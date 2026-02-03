import logging
from typing import Tuple

import torch
from torchvision import datasets, transforms

from .base import BaseDataset

logger = logging.getLogger(__name__)


class FashionMNISTDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        n_train: int,
        n_test: int,
        n_val: int = 0,
        target_labels: tuple[int, int] | None = None,
        random_seed: int = 42,
    ):
        super().__init__(data_dir, n_train, n_test, n_val, target_labels, random_seed)

    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Loading FashionMNIST data from {self.data_dir}")
        dataset = datasets.FashionMNIST(
            root=self.data_dir, train=True, download=True, transform=transforms.ToTensor()
        )

        # Use provided target_labels or default to (0, 1)
        labels = self.target_labels or (0, 1)

        idx = (dataset.targets == labels[0]) | (dataset.targets == labels[1])
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

        targets = dataset.targets.clone().detach()
        # Map classes to -1.0 and 1.0
        new_targets = torch.where(targets == labels[0], torch.tensor(-1.0), torch.tensor(1.0))
        data = dataset.data.float().unsqueeze(1) / 255.0

        return data, new_targets
