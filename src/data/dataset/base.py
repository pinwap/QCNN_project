from abc import ABC, abstractmethod
from typing import Tuple

import torch
from sklearn.model_selection import train_test_split


class BaseDataset(ABC):
    def __init__(
        self,
        data_dir: str,
        n_train: int,
        n_test: int,
        target_labels: tuple[int, int] | None = None,
        random_seed: int = 42,
    ):
        self.data_dir = data_dir
        self.n_train = n_train
        self.n_test = n_test
        self.target_labels = target_labels
        self.random_seed = random_seed

    @abstractmethod
    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return full dataset tensors.

        Returns:
            data (torch.Tensor): The feature vectors.
            labels (torch.Tensor): The target labels.
        """
        raise NotImplementedError

    def split(
        self, data: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the dataset into training and testing sets using scikit-learn.

        Args:
            data (torch.Tensor): The feature vectors.
            labels (torch.Tensor): The target labels.

        Returns:
            x_train (torch.Tensor): The training feature vectors.
            x_test (torch.Tensor): The testing feature vectors.
            y_train (torch.Tensor): The training target labels.
            y_test (torch.Tensor): The testing target labels.
        """
        return tuple(
            train_test_split(
                data,
                labels,
                train_size=self.n_train,
                test_size=self.n_test,
                random_state=self.random_seed,
                shuffle=True,
            )
        )
