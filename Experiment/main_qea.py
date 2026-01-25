
import sys
import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from QCNN.DataManager import MNISTDataManager
from QCNN.QCNN_structure import QCNNBuilder
from QCNN.Evaluation import HybridEvaluator
from QCNN.utils import initialize_output_dir, graph_history, save_model
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

logger = logging.getLogger(__name__)

def main():
    # 0. Initialize Output Directory & Logging
    save_dir, file_id = initialize_output_dir(script_name="main_qea_retrain")
    
    # 1. Load Data
    # Path to the data file with the best structure
    json_path = os.path.join("output", "main_full_20260122_121801", "main_full_20260122_121801_data.json")
    
    if not os.path.exists(json_path):
        logger.error(f"Data file not found: {json_path}")
        return

    logger.info(f"Loading structure from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract configuration
    best_structure_code = data["results"]["best_structure_code"]
    config = data["configuration"]
    
    # Extract data parameters from config (or default if missing)
    n_train = config.get("data", {}).get("n_train", 400)
    n_test = config.get("data", {}).get("n_test", 100)
    data_path = config.get("data", {}).get("data_path", "../data")
    n_qubits = config.get("evaluator", {}).get("n_qubits", 16)
    
    # Override training epochs
    epochs = 40
    lr = 0.01

    logger.info(f"Configuration: n_qubits={n_qubits}, n_train={n_train}, epochs={epochs}")

    # 2. Prepare Data Manager
    data_manager = MNISTDataManager(
        data_path=data_path, n_train=n_train, n_test=n_test
    )
    # Generate/Load datasets
    experiment_wrapper = type('obj', (object,), {'data_mgr': data_manager}) # Mock object to access get_data if needed directly or just trust the property access
    
    # Access tensors directly
    x_train = data_manager.train_images
    y_train = data_manager.train_labels
    x_test = data_manager.test_images
    y_test = data_manager.test_labels

    # 3. Rebuild the Model using the best structure
    logger.info("Rebuilding best QCNN model...")
    builder = QCNNBuilder(n_qubits=n_qubits)
    evaluator = HybridEvaluator(builder, epochs=epochs, lr=lr)

    # We need to manually duplicate the evaluator logic here to get the model object + training loop with history
    # Step 3a: Assemble Circuit
    qc, last_qubit = builder.assemble(best_structure_code)
    
    # Step 3b: Feature Map & Observable
    fm, input_params = evaluator._create_feature_map(n_qubits)
    observable = evaluator._crate_observable(last_qubit, n_qubits)
    full_circuit = fm.compose(qc)

    # Step 3c: QNN
    qnn = EstimatorQNN(
        circuit=full_circuit,
        input_params=list(input_params),
        weight_params=list(qc.parameters),
        observables=observable,
        estimator=evaluator.estimator,
        gradient=evaluator.grad_method,
        input_gradients=True,
    )

    # Step 3d: Torch Model
    model = TorchConnector(qnn).to(evaluator.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Move data to device
    x_train_dev = x_train.to(evaluator.device)
    y_train_dev = y_train.to(evaluator.device).unsqueeze(1)
    x_test_dev = x_test.to(evaluator.device)
    y_test_dev = y_test.to(evaluator.device).unsqueeze(1)

    # 4. Training Loop
    logger.info(f"Starting training for {epochs} epochs...")
    loss_history = []
    accuracy_history = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train_dev)
        loss = loss_fn(output, y_train_dev)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # Calculate Accuracy for this epoch
        with torch.no_grad():
            preds = (output >= 0).float() # Assuming label mapping -1/1 or 0/1. 
            # Note: Input labels are likely 0 and 1. QNN with Z observable usually outputs [-1, 1].
            # If QNN outputs in [-1, 1], we map to [0, 1] for accuracy check.
            # Usually: exp val > 0 -> Class 0? Wait, let's check HybridEvaluator assumption.
            # HybridEvaluator uses MSELoss against y_train_dev.
            # If y is 0/1, model tries to output 0/1.
            # Predictions: round to nearest integer [0, 1]
             # For MSELoss with 0/1 targets, threshold usually 0.5
            preds = (output >= 0.5).float()
            correct = (preds == y_train_dev).sum().item()
            acc = correct / len(y_train_dev)
            accuracy_history.append(acc)


        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_val:.4f}, Acc: {acc:.4f}")

    # 5. Final Evaluation
    model.eval()
    with torch.no_grad():
        test_output = model(x_test_dev)
        test_preds = (test_output >= 0.5).float()
        test_correct = (test_preds == y_test_dev).sum().item()
        test_acc = test_correct / len(y_test_dev)
    
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")

    # 6. Save Results using the shared util
    # We construct a mock experiment object to satisfy the graph_history signature requirements 
    # for the Qiskit-style plotting (Loss + Acc) we just implemented.
    class MockExperiment:
        def __init__(self):
            self.objective_func_vals = loss_history # Triggers 'is_qiskit' logic in utils.py
            self.accuracy_history = accuracy_history
            # Add other attrs if needed by utils.py

    mock_exp = MockExperiment()
    
    logger.info("Saving results...")
    graph_history(
        best_model=test_acc,
        history=loss_history,
        experiment=mock_exp,
        save_dir=save_dir,
        file_id=file_id
    )
    
    # Save the PyTorch Model
    save_model(model, save_dir=save_dir, file_id=file_id)

    # Visual check (Optional - console only since we save graphs)
    print(f"✅ Training Complete. Saved to {save_dir}")

if __name__ == "__main__":
    main()
