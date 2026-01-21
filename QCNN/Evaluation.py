import copy
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from QCNN.DataManager import BaseDataManager
from QCNN.QCNN_structure import QCNNBuilder
from QCNN.QEA_core import QuantumChromosome

# Suppress Qiskit Machine Learning logging warnings about gradients
logging.getLogger("qiskit_machine_learning").setLevel(logging.ERROR)


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        structure_code: List[int],
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> float:
        pass


class HybridEvaluator(Evaluator):
    def __init__(
        self,
        builder: QCNNBuilder,
        epochs: int = 5,
        lr: float = 0.01,
        verbose: bool = True,
    ):
        """
        builder: QCNNBuilder
        epochs: จำนวนรอบการฝึก
        lr: learning rate
        """
        self.builder = builder
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.loss_fn = nn.MSELoss()

    def _create_feature_map(self, n_qubits: int = 16) -> Tuple[QuantumCircuit, ParameterVector]:
        # สร้างวงจร Encode ข้อมูล ด้วย Angle Encoding
        fm = QuantumCircuit(n_qubits)
        inputs = ParameterVector("input", n_qubits)
        for i in range(n_qubits):
            fm.rx(inputs[i] * np.pi, i)
        return fm, inputs

    def _crate_observable(self, last_qubit: int, n_qubits: int = 16) -> SparsePauliOp:
        # สร้างตัววัดค่า Z ที่ Qubit สุดท้าย
        # qiskit เรียง qubit จากขวาไปซ้าย !!!
        return SparsePauliOp.from_sparse_list([("Z", [last_qubit], 1)], num_qubits=n_qubits)

    def evaluate(
        self,
        structure_code: List[int],
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> float:
        # Main Pipeline: Build -> Train -> Test -> Return Accuracy

        # 1. สร้างโมเดล QCNN จากโครงสร้าง
        qc, last_qubit = self.builder.assemble(structure_code)

        # 2. feature map & observe
        fm, input_params = self._create_feature_map(self.builder.n_qubits)
        observable = self._crate_observable(last_qubit, self.builder.n_qubits)

        # 3. รวม feature map กับ QCNN
        full_circuit = QuantumCircuit(self.builder.n_qubits)
        full_circuit.append(fm, range(self.builder.n_qubits))
        full_circuit.append(qc, range(self.builder.n_qubits))

        # 4. define QNN
        qnn = EstimatorQNN(
            circuit=full_circuit,
            input_params=list(input_params),
            weight_params=list(qc.parameters),
            observables=observable,
        )

        # 5. Train Hybrid(Torch+Qiskit)
        model = TorchConnector(qnn)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = model(x_train)
            loss = self.loss_fn(output, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
            if self.verbose:
                print(f"    Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        # 6. Test Accuracy
        model.eval()
        with torch.no_grad():
            preds = torch.sign(model(x_test))
            acc = (preds == y_test.unsqueeze(1)).float().mean().item()

        return acc


class Experiment:
    def __init__(
        self,
        data_mgr: BaseDataManager,
        evaluator: Evaluator,
        n_pop: int = 10,
        n_gen: int = 5,
        n_gates: int = 180,
    ):
        self.data_mgr = data_mgr  # Data Manager
        self.evaluator = evaluator
        self.n_gen = n_gen  # จำนวนรุ่น

        # สร้างจำนวนวงจรเริ่มต้น n_pop ตัวใน 1 รุ่น
        self.population = [QuantumChromosome(n_gates) for _ in range(n_pop)]
        self.global_best: Optional[QuantumChromosome] = None
        self.history: List[float] = []

    def _apply_crossover(self) -> None:
        print("Triggering Crossover")
        n_pop = len(self.population)
        n_genes = len(self.population[0].genes)

        # สร้าง population ใหม่ มารอรับค่า
        new_population = [p.copy() for p in self.population]

        for i in range(n_pop):
            for j in range(n_genes):
                # สูตรตาม Table 2: ตัวที่ i รับยีนตำแหน่ง j มาจากตัวที่ (i+j)%N
                source_idx = (i + j) % n_pop
                # ก๊อปปี้ gene มาใส่ (ต้องระวังเรื่อง pass by reference)
                new_population[i].genes[j] = copy.deepcopy(self.population[source_idx].genes[j])

        self.population = new_population

    def run(self) -> Tuple[Optional[QuantumChromosome], List[float]]:
        # 1. Prepair Data
        x_train, y_train, x_test, y_test = self.data_mgr.get_data()
        if x_train is None or y_train is None or x_test is None or y_test is None:
            print("Data loading failed. Experiment cannot proceed.")
            return None, []

        print(f"\n🚀 Start Experiment: {self.n_gen} Generations x {len(self.population)} Pop")

        stagnation_counter = 0

        for gen in range(self.n_gen):
            print(f"\n--- Generation {gen + 1}/{self.n_gen} ---")

            # 2. Loop Population
            for i, chromo in enumerate(self.population):
                print(f"  Evaluating Population {i + 1}/{len(self.population)}...")
                # A. Collapse -> Structure
                struct_code = chromo.collapse()

                # B. Train - Evaluate 5 Epochs
                acc = self.evaluator.evaluate(struct_code, x_train, y_train, x_test, y_test)
                chromo.fitness = acc
                print(f"  Pop {i + 1} Result: Acc = {acc:.4f}")

            # 3. Find Update Global Best
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.global_best is None or current_best.fitness > self.global_best.fitness:
                self.global_best = current_best.copy()
                stagnation_counter = 0  # รีเซ็ตนับใหม่ เพราะคะแนนขยับ
                print(f"  🏆 New Global Best: {self.global_best.fitness:.4f}")
            else:
                stagnation_counter += 1  # คะแนนไม่ขยับ
                print(f" Stagnation: {stagnation_counter}/10")

            # 4. Evolution (Update Genes)
            if stagnation_counter >= 10:
                self._apply_crossover()
                stagnation_counter = 0  # รีเซ็ตนับใหม่หลังครอสโอเวอร์
            else:
                for chromo in self.population:
                    chromo.update_genes(self.global_best)

            self.history.append(self.global_best.fitness)

        return self.global_best, self.history
