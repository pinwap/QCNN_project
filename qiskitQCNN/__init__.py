import json
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import z_feature_map
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN

from sklearn.model_selection import train_test_split

# from qiskitQCNN.DataManager import generate_dataset
# from qiskitQCNN.qiskitQCNN_structure import crate_qcnn_ansatz
