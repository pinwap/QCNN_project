from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Callable, Sequence
import torch


class Preprocessor(ABC):
    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Transform input tensor and return it."""


class Identity(Preprocessor):
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data


class BilinearResize(Preprocessor):
    def __init__(self, size: tuple[int, int] = (4, 4)):
        self.size = size

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(data, size=self.size, mode="bilinear")


class Flatten(Preprocessor):
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data.view(data.shape[0], -1)


class PCAReducer(Preprocessor):
    def __init__(self, target_dim: int):
        self.target_dim = target_dim

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        flat = data.view(data.shape[0], -1)
        u, s, vh = torch.pca_lowrank(flat, q=self.target_dim)
        return torch.matmul(flat - flat.mean(dim=0), vh[:, : self.target_dim])


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
    def __init__(self, keep: int | None = None):
        self.keep = keep

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if not hasattr(torch.fft, "dct"):
            raise RuntimeError("torch.fft.dct is unavailable; upgrade PyTorch to >= 2.1")
        flat = data.view(data.shape[0], -1)
        coeffs = torch.fft.dct(flat, dim=1, norm="ortho")
        if self.keep:
            coeffs = coeffs[:, : self.keep]
        return coeffs


PREPROCESSOR_REGISTRY: Dict[str, Callable[..., Preprocessor]] = {
    "linear": Identity,
    "bilinear_resize_4x4": lambda: BilinearResize((4, 4)),
    "flatten": Flatten,
    "pca_16": lambda: PCAReducer(16),
    "pca_32": lambda: PCAReducer(32),
    "dct": DCTPreprocessor,
    "dct_keep_64": lambda: DCTPreprocessor(64),
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
