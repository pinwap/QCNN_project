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
        binary_groups: list[list[int]] | None = None,
    ):
        super().__init__(data_dir, n_train, n_test, n_val, target_labels, random_seed)
        self.binary_groups = binary_groups

    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Loading FashionMNIST data from {self.data_dir}")
        dataset = datasets.FashionMNIST( #dataset เก็บข้อมูล fashionmnist [60000=label,280,28] 28*28 เก็บ intensity ของpixel
            root=self.data_dir, train=True, download=True, transform=transforms.ToTensor()
        )

        if self.binary_groups:
            logger.info(f"Using binary groups: {self.binary_groups}")
            if len(self.binary_groups) != 2: 
                raise ValueError("binary_groups must contain exactly 2 lists of labels")
                # binary_gruops = จะเป็น list ที่มี 2 มิติ [[],[]] ที่อันแรกเก็บ labels ของtargetที่เราจะเลือกเป็น group0 และอันที่สองเก็บ labels ของอันอื่นที่ไม่ใช่ target เป็น group1 เช่น label 2กับไม่ใช่2
                
            group0_labels = self.binary_groups[0]
            group1_labels = self.binary_groups[1]

            # หา index ของข้อมูลที่ตรงกับ group0 และ group1
            def get_indices(labels_list):
                mask = torch.zeros_like(dataset.targets, dtype=torch.bool) #สร้าง tensor ที่มีขนาดเท่ากับ dataset.targets = จำนวน label แล้วเก็บค่า false ทั้งหมด
                for l in labels_list:
                    mask |= (dataset.targets == l) #ถ้า targets ตรงกับ label ที่เราสนใจ ให้เปลี่ยนค่าใน mask ตำแหน่งนั้นเป็น true
                return mask.nonzero(as_tuple=True)[0] #ส่งคืนตำแหน่งที่เป็น true ใน mask. as_tuple ได้ (([index ที่ true]),) เลยต้องใส่ [0] เพื่อเอาแค่ ()แรกที่เก็บindex

            # ได้ index ของลาเบลที่เราต้องการว่าอยู่ที่ตำแหน่งไหนบน dataset ใหญ่
            idx0 = get_indices(group0_labels)
            idx1 = get_indices(group1_labels)
            
            # ทำให้จำนวนตัวอย่างใน group0=group1 เพื่อไม่ให้เกิด bias โมเดลเดาสุ่ม โดยการสุ่มลดขนาดกลุ่มที่มีจำนวนมากกว่า
            n_samples = len(idx0)
            generator = torch.Generator().manual_seed(self.random_seed) #สร้าง generator สำหรับการสุ่มที่มี seed คงที่เพื่อให้ผลลัพธ์การสุ่มเหมือนเดิมทุกครั้ง
            if len(idx1) > n_samples:
                perm = torch.randperm(len(idx1), generator=generator)
                idx1 = idx1[perm[:n_samples]]
                logger.info(f"Subsampled group 1 from {len(perm)} to {len(idx1)} to match group 0")
            
            # รวม index ของกลุ่ม 0 และ 1
            all_idx = torch.cat([idx0, idx1])
            
            # สุ่มข้อมูลรวมเพื่อหลีกเลี่ยงการเรียงลำดับบล็อก
            shuffle_perm = torch.randperm(len(all_idx), generator=generator)
            all_idx = all_idx[shuffle_perm]

            dataset.data = dataset.data[all_idx]
            original_targets = dataset.targets[all_idx]
            
            # Identify which targets belong to group 0
            mask0 = torch.zeros_like(original_targets, dtype=torch.bool)
            for l in group0_labels:
                mask0 |= (original_targets == l)
            
            # Map group 0 to -1.0 and group 1 to 1.0
            new_targets = torch.where(mask0, torch.tensor(-1.0), torch.tensor(1.0))
            data = dataset.data.float().unsqueeze(1) / 255.0
            
            logger.info(f"Loaded {len(data)} samples with binary groups.")
            return data, new_targets

        # Use provided target_labels or default to (0, 1)
        labels = self.target_labels or (0, 1)
        # idx เป็นได้เป็น tensor true/false ที่เก็บตำแหน่งของข้อมูลที่ตรงกับเงื่อนไข
        idx = (dataset.targets == labels[0]) | (dataset.targets == labels[1])
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

        targets = dataset.targets.clone().detach()
        # Map classes to -1.0 and 1.0
        new_targets = torch.where(targets == labels[0], torch.tensor(-1.0), torch.tensor(1.0))
        data = dataset.data.float().unsqueeze(1) / 255.0

        return data, new_targets
