import logging
from typing import Optional

from models.qcnn.base import BaseQCNN
from training.engines.base import BaseEngine
from training.engines.hybrid import HybridEngine

logger = logging.getLogger(__name__)


class ProductionPipeline:
    """
    Standard Pipeline for production-grade training of a fixed QCNN model.
    A Pipeline wraps an Engine for deep training and result management.
    """

    def __init__(
        self,
        model: BaseQCNN,
        engine: Optional[BaseEngine] = None,
        # Default config if engine not provided
        epochs: int = 50,
        lr: float = 0.001,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model = model

        if engine:
            self.engine = engine
        else:
            # Default to Hybrid Engine for production training
            self.engine = HybridEngine(epochs=epochs, lr=lr, device=device, verbose=verbose)

    def run(self, x_train, y_train, x_test=None, y_test=None):
        """
        Execute the training pipeline.
        """
        logger.info(f"Executing Production Pipeline for model: {self.model.__class__.__name__}")

        # 1. Build the circuit structure
        qc, params, last_qubit = self.model.build_with_metadata()

        logger.info(f"Circuit built. Num Qubits: {qc.num_qubits}, Num Params: {len(params)}")

        # 2. Delegate to Engine
        score, history, trained_obj = self.engine.fit(
            circuit=qc,
            params=params,
            last_qubit=last_qubit,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

        logger.info(f"Pipeline Execution Complete. Final Score: {score:.4f}")
        return score, history, trained_obj
