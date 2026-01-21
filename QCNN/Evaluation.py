import numpy as np
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister 
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import torch
from QCNN.QEA_core import QuantumChromosome

class HybridEvaluator:
    def __init__(self, builder, epochs = 5):
        self.builder = builder
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()
    
    # 
    def evaluate(self, structure_code, x_train, y_train, x_test, y_test):
        # 1. สร้างโมเดล QCNN จากโครงสร้าง
        qc, last_qubit = self.builder.assemble(structure_code)
        
        # 2. feature map (angle encode)
        fm = QuantumCircuit(16)
        inputs = ParameterVector('input', 16)
        for i in range(16):
            fm.rx(inputs[i]*np.pi, i)
        
        # 3. รวม feature map กับ QCNN
        full_circuit = QuantumCircuit(16)
        full_circuit.append(fm, range(16))
        full_circuit.append(qc, range(16))
        
    # 4. QNN Definition
        # Create observable circuit properly
        obs_circuit = QuantumCircuit(16)
        if isinstance(last_qubit, int):
            obs_circuit.z(last_qubit)
        else:
            obs_circuit.append(last_qubit, range(16))
        
        qnn = EstimatorQNN(
            circuit=full_circuit,
            input_params=inputs,
            weight_params=qc.parameters,
            observables=obs_circuit
        )    
        model = TorchConnector(qnn)
        opt = optim.Adam(model.parameters(), lr=0.01)
        
        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = self.loss_func(model(x_train), y_train.unsqueeze(1))
            loss.backward()
            opt.step()
            
        # 6. Test
        model.eval()
        with torch.no_grad():
            preds = torch.sign(model(x_test))
            acc = (preds == y_test.unsqueeze(1)).float().mean().item()
            
        return acc
    
class Experiment:
    def __init__(self, data_mgr, builder, evaluator, n_pop=10, n_gen=5, n_gates=180):
        self.data_mgr = data_mgr # Data Manager
        self.builder = builder 
        self.evaluator = evaluator
        
        # สร้างจำนวนวงจรเริ่มต้น n_pop ตัวใน 1 รุ่น
        self.population = [QuantumChromosome(n_gates) for _ in range(n_pop)]
        self.n_gen = n_gen # จำนวนรุ่น
        self.global_best = None
        self.history = []

    def run(self):
        # 1. Get Data
        x_train, y_train, x_test, y_test = self.data_mgr.get_data()
        
        print(f"\n🚀 Start Experiment: {self.n_gen} Generations x {len(self.population)} Pop")
        
        for gen in range(self.n_gen):
            print(f"\n--- Generation {gen+1} ---")
            fitnesses = []
            
            # 2. Loop Population
            for i, chromo in enumerate(self.population):
                # A. Collapse -> Structure
                struct_code = chromo.collapse()
                
                # B. Evaluate (Train 5 Epochs)
                acc = self.evaluator.evaluate(struct_code, x_train, y_train, x_test, y_test)
                chromo.fitness = acc
                fitnesses.append(acc)
                print(f"  Pop {i+1}: Acc = {acc:.4f}")

            # 3. Find Best
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.global_best is None or current_best.fitness > self.global_best.fitness:
                self.global_best = current_best.copy()
                print(f"  🏆 New Global Best: {self.global_best.fitness:.4f}")
            
            self.history.append(self.global_best.fitness)

            # 4. Evolution (Update Genes)
            for chromo in self.population:
                chromo.update(self.global_best)

        return self.global_best, self.history