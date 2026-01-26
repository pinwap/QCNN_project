from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Callable
import logging

import numpy as np
import torch
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    def __init__(self, data_path: str, n_train: int, n_test: int, digits: tuple[int, int] | None = None):
        self.data_path = data_path
        self.n_train = n_train
        self.n_test = n_test
        self.digits = digits

    @abstractmethod
    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full dataset tensors (data, labels)."""

    def split(
        self, data: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_train, y_train = data[: self.n_train], labels[: self.n_train]
        x_test, y_test = (
            data[self.n_train : self.n_train + self.n_test],
            labels[self.n_train : self.n_train + self.n_test],
        )
        return x_train, y_train, x_test, y_test


class MNISTDataset(BaseDataset):
    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Loading MNIST data from {self.data_path}")
        dataset = datasets.MNIST(
            root=self.data_path, train=True, download=True, transform=transforms.ToTensor()
        )
        digits = self.digits or (3, 6)
        idx = (dataset.targets == digits[0]) | (dataset.targets == digits[1])
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

        targets = dataset.targets.clone().detach()
        new_targets = torch.where(targets == digits[0], torch.tensor(-1.0), torch.tensor(1.0))
        data = dataset.data.float().unsqueeze(1) / 255.0
        return data, new_targets


class FashionMNISTDataset(BaseDataset):
    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Loading FashionMNIST data from {self.data_path}")
        dataset = datasets.FashionMNIST(
            root=self.data_path, train=True, download=True, transform=transforms.ToTensor()
        )
        digits = self.digits or (0, 1)  # default two classes
        idx = (dataset.targets == digits[0]) | (dataset.targets == digits[1])
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

        targets = dataset.targets.clone().detach()
        new_targets = torch.where(targets == digits[0], torch.tensor(-1.0), torch.tensor(1.0))
        data = dataset.data.float().unsqueeze(1) / 255.0
        return data, new_targets


class CIFAR10Dataset(BaseDataset):
    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Loading CIFAR10 data from {self.data_path}")
        dataset = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        classes = self.digits or (0, 1)  # use two classes for binary
        idx = (torch.tensor(dataset.targets) == classes[0]) | (torch.tensor(dataset.targets) == classes[1])
        data = dataset.data[idx]
        targets = torch.tensor(dataset.targets)[idx]

        new_targets = torch.where(targets == classes[0], torch.tensor(-1.0), torch.tensor(1.0))
        data = torch.tensor(data).permute(0, 3, 1, 2).float() / 255.0  # NHWC -> NCHW
        return data, new_targets


class SyntheticStripesDataset(BaseDataset):
    """Binary synthetic stripes dataset (horizontal vs vertical) used by qiskit demos."""

    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        total = self.n_train + self.n_test
        images = []
        labels = []

        hor_array = np.zeros((6, 8))
        ver_array = np.zeros((4, 8))

        j = 0
        for i in range(0, 7):
            if i != 3:
                hor_array[j][i] = np.pi / 2
                hor_array[j][i + 1] = np.pi / 2
                j += 1

        j = 0
        for i in range(0, 4):
            ver_array[j][i] = np.pi / 2
            ver_array[j][i + 4] = np.pi / 2
            j += 1

        rng = np.random.default_rng()
        for _ in range(total):
            bit = rng.integers(0, 2)
            if bit == 0:
                labels.append(-1)
                base_image = np.array(hor_array[rng.integers(0, 6)])
            else:
                labels.append(1)
                base_image = np.array(ver_array[rng.integers(0, 4)])
            noise = rng.uniform(0, np.pi / 4, size=8)
            final_image = np.where(base_image == 0, noise, base_image)
            images.append(final_image)

        data = torch.tensor(np.array(images)).float()
        targets = torch.tensor(np.array(labels)).float()
        return data, targets


DATASET_REGISTRY: Dict[str, Callable[..., BaseDataset]] = {
    "mnist": MNISTDataset,
    "fashionmnist": FashionMNISTDataset,
    "cifar10": CIFAR10Dataset,
    "synthetic": SyntheticStripesDataset,
}


def resolve_dataset(name: str, **kwargs) -> BaseDataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name](**kwargs)
