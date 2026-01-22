from abc import ABC, abstractmethod
from typing import Optional, Tuple
import logging

import torch
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# 1. สร้าง Abstract Base Class
class BaseDataManager(ABC):
    @abstractmethod
    def get_data(
        self,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """ทุก DataManager ต้องมีฟังก์ชันนี้"""
        pass


# 2. สร้าง MNIST Manager ที่สืบทอดมาจาก Abstract Base Class
class MNISTDataManager(BaseDataManager):
    def __init__(self, data_path: str = "../data", n_train: int = 400, n_test: int = 100):
        self.data_path = data_path
        self.n_train = n_train
        self.n_test = n_test

    def get_data(
        self,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        logger.info(f"Loading MNIST data from {self.data_path}")
        # 1. Transform ขั้นตอนการแปลงข้อมูล
        transform = transforms.Compose(
            [
                transforms.Resize((4, 4)),
                transforms.ToTensor(),
            ]
        )

        # 2. Load MNIST Dataset
        try:
            dataset = datasets.MNIST(
                root=self.data_path, train=True, download=True, transform=transform
            )
        except Exception as e:
            logger.error(f"Error loading MNIST: {e}")
            return None, None, None, None

        # 3. Filter 3 & 6
        idx = (dataset.targets == 3) | (dataset.targets == 6)
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

        # 4. Remap Labels (-1, 1)
        targets = dataset.targets.clone().detach()
        new_targets = torch.where(targets == 3, torch.tensor(-1.0), torch.tensor(1.0))

        # 5. Preprocessing (Resize & Normalize)
        data = dataset.data.float().unsqueeze(1) / 255.0
        data = torch.nn.functional.interpolate(data, size=(4, 4), mode="bilinear")
        data = data.view(-1, 16)

        # 6. Split
        x_train, y_train = data[: self.n_train], new_targets[: self.n_train]
        x_test, y_test = (
            data[self.n_train : self.n_train + self.n_test],
            new_targets[self.n_train : self.n_train + self.n_test],
        )

        logger.info(f"Data loaded successfully. Train size: {x_train.shape[0]}, Test size: {x_test.shape[0]}")
        return x_train, y_train, x_test, y_test

        print(f"✅ Data Ready: Train {x_train.shape}, Test {x_test.shape}")
        return x_train, y_train, x_test, y_test
