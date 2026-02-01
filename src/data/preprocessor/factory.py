from typing import Callable, Dict, Sequence

from .base import BasePreprocessor
from .autoencoder import AutoencoderReducer
# from .bilinear_resize import BilinearResize
# from .dct_preprocessor import DCTPreprocessor
from .flatten import Flatten
# from .identity import Identity
from .pca_reducer import PCAReducer

PREPROCESSOR_REGISTRY: Dict[str, Callable[..., BasePreprocessor]] = {
    # "linear": Identity,
    # "bilinear_resize_4x4": lambda: BilinearResize((4, 4)),
    "flatten": lambda **kwargs: Flatten(),
    "pca_16": lambda **kwargs: PCAReducer(16),
    "autoencoder_16": lambda **kwargs: AutoencoderReducer(16, dataset_name=kwargs.get("dataset_name", "unknown")),
    # "dct_16": lambda: DCTPreprocessor(16),
}


def resolve_preprocessors(steps: Sequence[BasePreprocessor | str] | None, **kwargs):
    resolved: list[BasePreprocessor] = []
    if not steps:
        return resolved
    for step in steps:
        if isinstance(step, BasePreprocessor):
            resolved.append(step)
        elif isinstance(step, str):
            if step not in PREPROCESSOR_REGISTRY:
                raise ValueError(f"Unknown preprocessor key: {step}")
            resolved.append(PREPROCESSOR_REGISTRY[step](**kwargs))
        else:
            raise TypeError(f"Unsupported preprocessor type: {type(step)}")
    return resolved
