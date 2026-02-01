import numpy as np
import torch
from scipy.fftpack import dct
from torch import Tensor

from .base import BasePreprocessor
from .min_max_scale import MinMaxScale


class DCTPreprocessor(BasePreprocessor):
    def __init__(self, target_dim: int | None = None):
        self.target_dim = target_dim
        # Calculate crop size (ex. if target_dim=16, we take 4x4 top-left)
        self.keep_size = int(np.sqrt(target_dim)) if target_dim is not None else None

    def _dct2d(self, a):
        # 2D DCT using Scipy (Type II, Orthogonal)
        return dct(dct(a.T, norm="ortho").T, norm="ortho")

    def __call__(self, data: Tensor) -> Tensor:
        # Convert to Numpy for Scipy
        device = data.device
        # Ensure input is (N, H, W) removing channel dim if exists
        if data.dim() == 4:
            imgs = data.squeeze(1).cpu().numpy()
        else:
            imgs = data.cpu().numpy()

        processed = []
        for img in imgs:
            # 1. Apply 2D DCT
            dct_img = self._dct2d(img)

            # 2. Zig-zag Crop / Top-Left Crop (Low Frequencies)
            crop = dct_img[: self.keep_size, : self.keep_size]

            # 3. Flatten
            flat = crop.flatten()
            processed.append(flat)

        # Convert back to Tensor
        result = torch.tensor(np.array(processed), dtype=torch.float32).to(device)

        # 4. Normalize to [0, 1]
        scaler = MinMaxScale()
        return scaler(result)
