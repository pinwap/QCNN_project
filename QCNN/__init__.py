import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from abc import ABC, abstractmethod

from qiskit.circuit.library import RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate
from qiskit import QuantumCircuit 
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp

from QCNN.QEA_core import QuantumChromosome  
from QCNN.DataManager import MNISTDataManager
from QCNN.QCNN_structure import QCNNBuilder
from QCNN.Evaluation import HybridEvaluator, Experiment

