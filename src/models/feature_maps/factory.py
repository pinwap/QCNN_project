from typing import Dict, Type, Union

from .angle import AngleEncodingMap
from .base import FeatureMapBuilder
from .pge import PGEEncodingMap
from .zz import ZZFeatureMapBuilder

FEATURE_MAP_REGISTRY: Dict[str, Type[FeatureMapBuilder]] = {
    "angle": AngleEncodingMap,
    "pge": PGEEncodingMap,
    "zz": ZZFeatureMapBuilder,
}


def resolve_feature_map(feature_map: Union[str, FeatureMapBuilder]) -> FeatureMapBuilder:
    """
    Resolve a feature map from a string or return the builder if already one.
    """
    if isinstance(feature_map, FeatureMapBuilder):
        return feature_map
    if isinstance(feature_map, str):
        key = feature_map.lower()
        if key not in FEATURE_MAP_REGISTRY:
            raise ValueError(f"Unknown feature map type: {feature_map}")
        return FEATURE_MAP_REGISTRY[key]()
    raise TypeError(f"Unsupported feature map type: {type(feature_map)}")
