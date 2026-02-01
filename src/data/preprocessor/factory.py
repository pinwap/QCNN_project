from typing import Callable, Dict, Sequence

from .base import BasePreprocessor
# from .bilinear_resize import BilinearResize
# from .dct_preprocessor import DCTPreprocessor
from .flatten import Flatten
# from .identity import Identity
from .pca_reducer import PCAReducer

PREPROCESSOR_REGISTRY: Dict[str, Callable[..., BasePreprocessor]] = {
    # "linear": Identity,
    # "bilinear_resize_4x4": lambda: BilinearResize((4, 4)),
    "flatten": Flatten,
    "pca_4": lambda: PCAReducer(4),
    "pca_8": lambda: PCAReducer(8),
    "pca_16": lambda: PCAReducer(16),
    # "dct_16": lambda: DCTPreprocessor(16),
}


def resolve_preprocessors(steps: Sequence[BasePreprocessor | str] | None):
    resolved: list[BasePreprocessor] = []
    if not steps:
        return resolved
    for step in steps:
        if isinstance(step, BasePreprocessor):
            resolved.append(step)
        elif isinstance(step, str):
            if step not in PREPROCESSOR_REGISTRY:
                raise ValueError(f"Unknown preprocessor key: {step}")
            resolved.append(PREPROCESSOR_REGISTRY[step]())
        else:
            raise TypeError(f"Unsupported preprocessor type: {type(step)}")
    return resolved
