from typing import Dict, Type, Union

from .angle import AngleEncodingMap
from .base import FeatureMapBuilder
from .pge import PGEEncodingMap
from .zz import ZZFeatureMapBuilder

# dict เก็บว่า string ไหนแมปกับ feature map builder ไหน เพื่อให้เรียกใช้ด้วย string ได้
FEATURE_MAP_REGISTRY: Dict[str, Type[FeatureMapBuilder]] = {
    "angle": AngleEncodingMap,
    "pge": PGEEncodingMap,
    "zz": ZZFeatureMapBuilder,
}


def resolve_feature_map(feature_map: Union[str, FeatureMapBuilder]) -> FeatureMapBuilder:
    """
    Resolve a feature map from a string or return the builder if already one.
    """
    if isinstance(feature_map, FeatureMapBuilder): #ถ้ารับมาเป็น object อยู่แล้ว ก็เอาไปใช้เลย
        return feature_map
    if isinstance(feature_map, str): #แต่ถ้ารับมาเป็น string ก็เช็คใน dict ว่าเป็นตัวไหน ส่งคืน object ของตัวนั้น
        key = feature_map.lower()
        if key not in FEATURE_MAP_REGISTRY:
            raise ValueError(f"Unknown feature map type: {feature_map}")
        return FEATURE_MAP_REGISTRY[key]()
    # ถ้ารับมาไม่ใช่ทั้ง string และ object ก็แจ้ง error
    raise TypeError(f"Unsupported feature map type: {type(feature_map)}")
