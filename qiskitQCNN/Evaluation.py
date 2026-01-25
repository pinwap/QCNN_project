from qiskitQCNN.DataManager import generate_dataset
from qiskitQCNN.qiskitQCNN_structure import crate_qcnn_ansatz

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNTrainer:
    def __init__(self, circuit, input_params, weight_params, initial_point_path=None):
        self.circuit = circuit
        self.objective_func_vals = []
        self.accuracy_history = []
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
        self._X_train = None
        self._y_train = None
    
    def callback_graph(self, weights, obj_func_eval):
        clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        
        # Calculate Accuracy (Manual Forward Pass)
        if self._X_train is not None and self._y_train is not None:
            # Forward pass to get expectation values
            # Output shape: (N_samples, 1)
            forward_output = self.qnn.forward(self._X_train, weights)
            
            # Map expectation values to labels {0, 1}
            # Assumption: Output > 0 -> Class 0 ("Horizontal" approx |0>)
            #             Output < 0 -> Class 1 ("Vertical" approx |1>)
            # NeuralNetworkClassifier default mapping might differ based on first label encountered
            # We assume standard mapping here for visualization
            
            # Note: NeuralNetworkClassifier usually does:
            # -1 -> Class 0, +1 -> Class 1 OR based on label encoding.
            # Let's try matching the most likely mapping.
            # If QNN uses Z operator: |0> -> +1, |1> -> -1.
            # If Class 0 is Horizontal (Simple), Class 1 is Vertical.
            # Let's assume +1 -> Class 0, -1 -> Class 1.
            preds = np.where(forward_output > 0, 0, 1)
            
            # If labels are not 0/1, this will fail. Assuming 0/1.
            accuracy = np.mean(preds.flatten() == self._y_train)
            self.accuracy_history.append(accuracy)
            
        # Optional: Internal Plotting (Can be removed if relying on main script)
        # plt.title("Objective function value against iteration")
        # plt.xlabel("Iteration")
        # plt.ylabel("Objective function value")
        # plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals)
        # plt.show()
    
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

    def train(self, X, y, max_iter=40):
        initial_point = self.load_or_initialize_point()
        
        # Reset histories
        self.objective_func_vals = []
        self.accuracy_history = []
        self._X_train = X
        self._y_train = y
        
        self.classifier = NeuralNetworkClassifier(
            self.qnn,
            optimizer=SPSA(maxiter=max_iter, learning_rate=0.01, perturbation=0.1),
            callback=self.callback_graph,
            initial_point=initial_point,
        )
        
        self.classifier.fit(X, y)
        
    def evaluate(self, X, y):
        score = self.classifier.score(X, y)
        print(f"Accuracy: {np.round(100 * score, 2)}%")
        return score

    def predict(self, X):
        return self.classifier.predict(X)
