from qiskitQCNN.DataManager import generate_dataset
from qiskitQCNN.qiskitQCNN_structure import crate_qcnn_ansatz

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNTrainer:
    def __init__(self, circuit, input_params, weight_params, initial_point_path=None):
        self.circuit = circuit
        self.objective_func_vals = []
        self.initial_point_path = initial_point_path or "initial_point.json"
        
        # Setup QNN
        estimator = Estimator()
        # วัดค่า Z ที่ Qubit ตัวแรก (เพราะเรา pool จนเหลือตัวเดียวที่ index 0)
        observable = SparsePauliOp.from_list([("Z" + "I" * (circuit.num_qubits - 1), 1)])
        
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )
        
        self.classifier = None
    
    def callback_graph(self, weights, obj_func_eval):
        clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals)
        plt.show()
    
    def load_or_initialize_point(self):
        """ถ้ามีไฟล์ json ให้โหลด ถ้าไม่มีให้สุ่มใหม่"""
        if os.path.exists(self.initial_point_path):
            print(f"Loading initial point from {self.initial_point_path}")
            with open(self.initial_point_path, "r") as f:
                return json.load(f)
        else:
            print("No initial point found. Initializing random weights.")
            # สุ่มตามจำนวนพารามิเตอร์ของวงจร
            return np.random.random(self.qnn.num_weights)    

    def train(self, X, y, max_iter=200):
        initial_point = self.load_or_initialize_point()
        
        self.classifier = NeuralNetworkClassifier(
            self.qnn,
            optimizer=COBYLA(maxiter=max_iter),
            callback=self._callback_graph,
            initial_point=initial_point,
        )
        
        self.classifier.fit(X, y)
        
    def evaluate(self, X, y):
        score = self.classifier.score(X, y)
        print(f"Accuracy: {np.round(100 * score, 2)}%")
        return score

    def predict(self, X):
        return self.classifier.predict(X)
