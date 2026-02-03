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
        n_val: int = 0,
        target_labels: tuple[int, int] | None = None,
        random_seed: int = 42,
    ):
        self.data_dir = data_dir
        self.n_train = n_train
        self.n_test = n_test
        self.n_val = n_val
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
    ) -> Tuple[torch.Tensor, ...]:
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
            (Optional) x_val, y_val if n_val > 0
        """
        # First split: Train+Val / Test
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            data,
            labels,
            test_size=self.n_test,
            random_state=self.random_seed,
            shuffle=True,
            stratify=labels,
        )

        if self.n_val > 0:
            # Second split: Train / Val
            # n_val is absolute number
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val,
                y_train_val,
                test_size=self.n_val,
                random_state=self.random_seed,
                shuffle=True,
                stratify=y_train_val,
            )
            # Ensure n_train matches if provided, otherwise it's the remainder
            # Ideally n_train + n_val + n_test <= total_data
            # But here we just split n_val out of the non-test set.
            # If n_train was strict, we might want to slice x_train[:self.n_train]
            if self.n_train < len(x_train):
                x_train = x_train[: self.n_train]
                y_train = y_train[: self.n_train]

            return x_train, x_val, x_test, y_train, y_val, y_test

        # No validation split
        x_train = x_train_val
        y_train = y_train_val
        if self.n_train < len(x_train):
             x_train = x_train[:self.n_train]
             y_train = y_train[:self.n_train]

        return x_train, x_test, y_train, y_test
