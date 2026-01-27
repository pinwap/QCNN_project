from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Callable, Sequence
import numpy as np
import torch
from torchvision import transforms
from scipy.fftpack import dct


class Preprocessor(ABC):
    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Transform input tensor and return it."""


class Identity(Preprocessor):
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data


class BilinearResize(Preprocessor):
    # Resize input images to specific dimensions  
    def __init__(self, size: tuple[int, int] = (4, 4)):
        self.size = size

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (N, 1, H, W) or (N, H, W)
        if data.dim() == 3:
            data = data.unsqueeze(1)  # add channel dimension for interpolation
        
        resized = torch.nn.functional.interpolate(data, size=self.size, mode="bilinear", align_corners=False)
        # Output shape: (N, 1, H_new, W_new) -> Flatten to (N, H*W)
        return resized.view(data.shape[0], -1)


class Flatten(Preprocessor):
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data.view(data.shape[0], -1)

class MinMaxScale(Preprocessor):
    """Normalize data to [0, 1] range per sample."""
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (N, Features)
        min_val = data.min(dim=1, keepdim=True)[0]
        max_val = data.max(dim=1, keepdim=True)[0]
        range_val = max_val - min_val
        
        # Avoid division by zero
        range_val[range_val == 0] = 1.0
        
        return (data - min_val) / range_val

class PCAReducer(Preprocessor):
    def __init__(self, target_dim: int):
        self.target_dim = target_dim

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        flat = data.view(data.shape[0], -1)
        u, s, vh = torch.pca_lowrank(flat, q=self.target_dim)
            # u: left singular vectors (shape: num_samples, q)
            # s: singular values (shape: q)
            # vh: right singular vectors (principal components/directions, shape: num_features, q)
        # Transform the data to the new principal components space
        projected = torch.matmul(flat - flat.mean(dim=0), vh[:, : self.target_dim])
        scaler = MinMaxScale()
        return scaler(projected)
class EnsureFeatureDimension(Preprocessor):
    """Trim or zero-pad features to a fixed length (e.g., match qubit count)."""

    def __init__(self, target_dim: int):
        self.target_dim = target_dim

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        flat = data.view(data.shape[0], -1)
        current = flat.shape[1]
        if current == self.target_dim:
            return flat
        if current > self.target_dim:
            return flat[:, : self.target_dim]
        pad = torch.zeros((flat.shape[0], self.target_dim - current), device=flat.device, dtype=flat.dtype)
        return torch.cat([flat, pad], dim=1)


class DCTPreprocessor(Preprocessor):
    def __init__(self, target_dim: int | None = None):
        self.target_dim = target_dim
        # Calculate crop size (ex. if target_dim=16, we take 4x4 top-left)
        self.keep_size = int(np.sqrt(target_dim))

    def _dct2d(self, a):
        # 2D DCT using Scipy (Type II, Orthogonal)
        return dct(dct(a.T, norm='ortho').T, norm='ortho')
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
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
            crop = dct_img[:self.keep_size, :self.keep_size]
            
            # 3. Flatten
            flat = crop.flatten()
            processed.append(flat)
            
        # Convert back to Tensor
        result = torch.tensor(np.array(processed), dtype=torch.float32).to(device)
        
        # 4. Normalize to [0, 1]
        scaler = MinMaxScale()
        return scaler(result)

PREPROCESSOR_REGISTRY: Dict[str, Callable[..., Preprocessor]] = {
    "linear": Identity,
    "bilinear_resize_4x4": lambda: BilinearResize((4, 4)),
    "flatten": Flatten,
    "pca_16": lambda: PCAReducer(16),
    "pca_32": lambda: PCAReducer(32),
    "dct_16": lambda: DCTPreprocessor(16),
    "dct_64": lambda: DCTPreprocessor(64),
}


def resolve_preprocessors(steps: Sequence[Preprocessor | str] | None):
    resolved: list[Preprocessor] = []
    if not steps:
        return resolved
    for step in steps:
        if isinstance(step, Preprocessor):
            resolved.append(step)
        elif isinstance(step, str):
            if step not in PREPROCESSOR_REGISTRY:
                raise ValueError(f"Unknown preprocessor key: {step}")
            resolved.append(PREPROCESSOR_REGISTRY[step]())
        else:
            raise TypeError(f"Unsupported preprocessor type: {type(step)}")
    return resolved
