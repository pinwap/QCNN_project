import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector