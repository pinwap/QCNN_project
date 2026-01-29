from typing import Callable, Dict

from .base import BaseDataset
from .cifar10_dataset import CIFAR10Dataset
from .fashion_mnist_dataset import FashionMNISTDataset
from .mnist_dataset import MNISTDataset
from .synthetic_stripes_dataset import SyntheticStripesDataset

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
