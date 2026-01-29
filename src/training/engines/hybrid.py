import logging
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimatorV1
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from models.feature_maps import FeatureMapBuilder, resolve_feature_map

from .base import BaseEngine

logger = logging.getLogger(__name__)


class HybridEngine(BaseEngine):
    """
    Optimization Engine using PyTorch + Qiskit.
    """

    def __init__(
        self,
        feature_map: Union[str, FeatureMapBuilder] = "angle",
        epochs: int = 5,
        lr: float = 0.01,
        gradient_method: str = "param_shift",
        device: Optional[str] = None,
        use_v2_primitives: bool = True,
        verbose: bool = True,
    ):
        self.feature_map_builder = resolve_feature_map(feature_map)
        self.epochs = epochs
        self.lr = lr
        self.gradient_method = gradient_method.lower()
        self.verbose = verbose
        self.use_v2 = use_v2_primitives
        self.loss_fn = nn.MSELoss()

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if self.verbose:
            logger.info(f"HybridEngine using device: {self.device}")

        self._setup_estimator()

    def _setup_estimator(self):
        if self.use_v2:
            self.estimator = AerEstimatorV2()
            if self.device.type == "cuda":
                # Attempt to set GPU; using type ignore as Aer V2 options are dynamic
                try:
                    self.estimator.options.update(device="GPU")  # type: ignore
                except Exception:
                    pass
                if self.verbose:
                    logger.info("HybridEngine: Attempting GPU enable.")
        else:
            options = {}
            if self.device.type == "cuda":
                options["device"] = "GPU"
            self.estimator = AerEstimatorV1(backend_options=options)  # type: ignore

        if self.gradient_method == "spsa":
            self.grad_method = SPSAEstimatorGradient(self.estimator, epsilon=0.01)  # type: ignore
        else:
            self.grad_method = ParamShiftEstimatorGradient(self.estimator)  # type: ignore

        if self.verbose:
            logger.info(f"HybridEngine: Using gradient method: {self.gradient_method}")

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

        if self.verbose:
            logger.info(f"HybridEngine: Optimizing {len(filtered_params)} parameters.")

        qnn = EstimatorQNN(
            circuit=full_circuit,
            input_params=list(input_params),
            weight_params=filtered_params,
            observables=observable,
            estimator=self.estimator,
            gradient=self.grad_method,  # type: ignore
            input_gradients=True,
        )

        model = TorchConnector(qnn).to(self.device)

        # Explicit initialization to avoid Barren Plateaus/Zero-gradient starts
        with torch.no_grad():
            for p in model.parameters():
                p.uniform_(-0.5, 0.5)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        if self.verbose:
            logger.info("HybridEngine: Initialized weights with uniform distribution [-0.5, 0.5]")
            logger.debug(
                f"HybridEngine: Circuit parameters: {[p.name for p in full_circuit.parameters]}"
            )
            logger.debug(f"HybridEngine: Training params: {[p.name for p in filtered_params]}")
            logger.debug(f"HybridEngine: Data Magnitude: {x_train.abs().mean().item():.3f}")
            if self.device.type == "mps":
                logger.warning(
                    "Using MPS (Metal Performance Shaders). "
                    "If gradients are zero, try 'engine.device=cpu'."
                )

        x_train_dev = x_train.to(self.device)
        y_train_dev = y_train.to(self.device).unsqueeze(1)

        history = {"loss": [], "train_acc": []}
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = model(x_train_dev)
            loss = self.loss_fn(output, y_train_dev)
            loss.backward()

            # Diagnostic: Check gradient magnitude
            total_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad += p.grad.norm().item()

            optimizer.step()

            history["loss"].append(float(loss.item()))

            # Calculate Epoch Training Accuracy
            with torch.no_grad():
                preds = torch.sign(output)
                train_acc = (preds == y_train_dev).float().mean().item()
                history["train_acc"].append(float(train_acc))

            if self.verbose:
                out_mean = output.mean().item()
                out_std = output.std().item()
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f} | "
                    f"Acc: {train_acc:.4f} | "
                    f"Grad: {total_grad:.2e} | Mean: {out_mean:.3f}, Std: {out_std:.3f}"
                )

        if x_test is not None and y_test is not None:
            x_test_dev = x_test.to(self.device)
            y_test_dev = y_test.to(self.device).unsqueeze(1)
            model.eval()
            with torch.no_grad():
                preds = torch.sign(model(x_test_dev))
                acc = (preds == y_test_dev).float().mean().item()
            return float(acc), history, model.state_dict()

        return 0.0, history, model.state_dict()
