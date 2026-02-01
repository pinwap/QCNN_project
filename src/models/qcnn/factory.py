from typing import Dict, Type

from .base import BaseQCNN
from .evolutionary_qcnn import EvolutionaryQCNN
from .standard_qcnn import StandardQCNN

QCNN_REGISTRY: Dict[str, Type[BaseQCNN]] = {
    "standard": StandardQCNN,
    "evolutionary": EvolutionaryQCNN,
}

# บอกโรงงานว่าอยากสร้าง QCNN แบบไหน เดะไปแกะใน dict ให้ว่ามันคืออันไหน จะได้ไปบอก constructor ถูก 

def resolve_qcnn(name: str, **kwargs) -> BaseQCNN:
    """
    Instantiate a QCNN model by name.

    Args:
        name: The key from QCNN_REGISTRY (e.g., 'standard' or 'evolutionary')
        **kwargs: Arguments for the specific model's constructor
                  (e.g., num_qubits, chromosome)
    """
    if name not in QCNN_REGISTRY:
        raise ValueError(f"Unknown QCNN type: {name}. Available: {list(QCNN_REGISTRY.keys())}")

    return QCNN_REGISTRY[name](**kwargs)
