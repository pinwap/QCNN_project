import logging
from typing import Any, Optional, Tuple, Union

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimatorV1
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_algorithms.gradients import SPSAEstimatorGradient
from qiskit_algorithms.optimizers import ADAM
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN

from models.feature_maps import FeatureMapBuilder, resolve_feature_map

from .base import BaseEngine

logger = logging.getLogger(__name__)


class QiskitEngine(BaseEngine):
    """
    Optimization Engine using Pure Qiskit (NeuralNetworkClassifier).
    """

    def __init__(
        self,
        feature_map: Union[str, FeatureMapBuilder] = "angle",
        max_iter: int = 50,
        verbose: bool = True,
        use_v2_primitives: bool = True,
    ):
        self.feature_map_builder = resolve_feature_map(feature_map)
        self.max_iter = max_iter
        self.verbose = verbose
        self.use_v2 = use_v2_primitives

        self._setup_estimator()

    def _setup_estimator(self):
        if self.use_v2:
            self.estimator = AerEstimatorV2()
            try:
                self.estimator.options.update(device="GPU")  # type: ignore
            except Exception:
                pass
        else:
            self.estimator = AerEstimatorV1()  # type: ignore

        self.grad_method = SPSAEstimatorGradient(self.estimator, epsilon=0.01)  # type: ignore

    def _create_observable(self, num_qubits: int, last_qubit: int) -> SparsePauliOp:
        return SparsePauliOp.from_sparse_list([("Z", [last_qubit], 1.0)], num_qubits=num_qubits)

    def fit(
        self,
        circuit: QuantumCircuit,
        params: ParameterVector,
        last_qubit: int,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
    ) -> Tuple[float, dict, Any]:
        num_qubits = circuit.num_qubits
        fm, input_params = self.feature_map_builder.build(num_qubits)

        composed = fm.compose(circuit)
        assert isinstance(composed, QuantumCircuit)
        full_circuit: QuantumCircuit = composed
        observable = self._create_observable(num_qubits, last_qubit)

        # Filter params
        circuit_params = set(full_circuit.parameters)
        filtered_params = [p for p in params if p in circuit_params]

        qnn = EstimatorQNN(
            circuit=full_circuit,
            input_params=list(input_params),
            weight_params=filtered_params,
            observables=observable,
            estimator=self.estimator,
            gradient=self.grad_method,  # type: ignore
            input_gradients=True,
        )

        # Prepare history tracking
        history = {"loss": []}

        def callback(weights, loss):
            history["loss"].append(float(loss))
            if self.verbose:
                # Log every 10 iterations to avoid spamming, or every 1 if max_iter is small
                if len(history["loss"]) % 10 == 0 or len(history["loss"]) == 1:
                    logger.info(
                        f"Qiskit Optimization Iteration {len(history['loss'])}/{self.max_iter} "
                        f"- Loss: {loss:.4f}"
                    )

        classifier = NeuralNetworkClassifier(
            qnn,
            optimizer=ADAM(maxiter=self.max_iter),  # type: ignore
            callback=callback,
        )

        X_train_np = x_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy()

        try:
            if self.verbose:
                logger.info(f"Starting Qiskit training for {self.max_iter} iterations...")

            classifier.fit(X_train_np, y_train_np)

            score = 0.0
            if x_test is not None and y_test is not None:
                X_test_np = x_test.cpu().numpy()
                y_test_np = y_test.cpu().numpy()
                score = classifier.score(X_test_np, y_test_np)

            # Now we return the REAL history captured by the callback!
            return float(score), history, classifier
        except Exception as e:
            logger.error(f"QiskitEngine failed: {e}")
            return 0.0, history, None
