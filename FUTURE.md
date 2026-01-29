# Future Development & Model Usage Guide

## 📦 Loading Saved Models

This project saves different types of model artifacts depending on the engine used. This guide explains how to load them correctly.

### 1. Hybrid Engine Models (PyTorch)
Files saved from runs using `engine=hybrid` are standard PyTorch state dictionaries.

**File Extension:** `.pth`
**Content:** Dictionary of weights (`OrderedDict`).

```python
import torch
from models.qcnn import StandardQCNN

# 1. Initialize the architecture (must match training)
model = StandardQCNN(num_qubits=16)

# 2. Load weights
state_dict = torch.load("path/to/model.pth")
model.load_state_dict(state_dict)

# 3. Inference
model.eval()
# y_pred = model(x_input)
```

### 2. Qiskit Engine Models (Pure Qiskit)
Files saved from runs using `engine=qiskit` are pickled `NeuralNetworkClassifier` objects.

**File Extension:** `.pth`
**Content:** Full `qiskit_machine_learning` object instance.

```python
import torch
# No need to import model class, the object is self-contained

# 1. Load the full object
# NOTE: weights_only=False is required for custom class loading
classifier = torch.load("path/to/model.pth", weights_only=False)

# 2. Inference
# y_pred = classifier.predict(x_input_numpy)
```

### 3. Evolutionary Search Results
Evolutionary runs save the **Best Chromosome** (a custom object).

**File Extension:** `.pth`
**Content:** `QuantumChromosome` object (Structure Only).

> **Note:** The chromosome defines the **architecture** (gates and connections), not the trained weights. You must retrain this structure to get a working model.

```python
import torch
from models.qcnn import EvolutionaryQCNN

# 1. Load the chromosome
chromosome = torch.load("path/to/best_chromosome.pth", weights_only=False)

# 2. Extract structure code
structure_code = chromosome.collapse()

# 3. Build the circuit (Unbound parameters)
# Note: You must match the num_qubits used during search (e.g., 16)
model_builder = EvolutionaryQCNN(num_qubits=16, chromosome=structure_code)
qc, params, last_qubit = model_builder.build_with_metadata()

# 4. To use this model:
# You would now pass 'qc' into the HybridEngine/QiskitEngine
# to TRAIN it from scratch on your full dataset.
```

## ⚠️ Important Note on Security
When loading Qiskit/Evolution models (`weights_only=False`), Python's `pickle` module executes arbitrary code during loading. **Only load models from sources you trust.**
