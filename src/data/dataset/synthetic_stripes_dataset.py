from typing import Tuple

import numpy as np
import torch

from .base import BaseDataset


class SyntheticStripesDataset(BaseDataset):
    """Binary synthetic stripes dataset (horizontal vs vertical) used by qiskit demos."""

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

        rng = np.random.default_rng(self.random_seed)
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
