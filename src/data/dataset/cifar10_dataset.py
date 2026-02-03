import logging
from typing import Tuple

import torch
from torchvision import datasets, transforms

from .base import BaseDataset

logger = logging.getLogger(__name__)


class CIFAR10Dataset(BaseDataset):
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
        logger.info(f"Loading CIFAR10 data from {self.data_dir}")
        dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        # Use provided target_labels or default to (0, 1)
        labels = self.target_labels or (0, 1)

        targets_tensor = torch.tensor(dataset.targets)
        idx = (targets_tensor == labels[0]) | (targets_tensor == labels[1])

        data = dataset.data[idx]
        targets = targets_tensor[idx]

        new_targets = torch.where(targets == labels[0], torch.tensor(-1.0), torch.tensor(1.0))
        # NHWC -> NCHW and normalize
        data = torch.tensor(data).permute(0, 3, 1, 2).float() / 255.0

        return data, new_targets
