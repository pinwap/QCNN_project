import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit 
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp
from QCNN.QEA_core import QuantumChromosome  

class HybridEvaluator:
    def __init__(self, builder, epochs = 5, lr = 0.01):
        '''
        builder: QCNNBuilder
        epochs: จำนวนรอบการฝึก
        lr: learning rate
        '''
        self.builder = builder
        self.epochs = epochs
        self.lr = lr
        self.loss_fn = nn.MSELoss()
    
    def _create_feature_map(self, n_qubits=16):
        # สร้างวงจร Encode ข้อมูล ด้วย Angle Encoding
        fm = QuantumCircuit(n_qubits)
        inputs = ParameterVector('input', n_qubits)
        for i in range(n_qubits):
            fm.rx(inputs[i]*np.pi, i)
        return fm, inputs
    
    def _crate_observable(self, last_qubit, n_qubits=16):
        # สร้างตัววัดค่า Z ที่ Qubit สุดท้าย
        # qiskit เรียง qubit จากขวาไปซ้าย !!!
        return SparsePauliOp.from_list([("Z", [last_qubit], 1)])

    def evaluate(self, structure_code, x_train, y_train, x_test, y_test):
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
            input_params=input_params,
            weight_params=qc.parameters,
            observables=observable
        )    
        
        # 5. Train Hybrid(Torch+Qiskit)
        model = TorchConnector(qnn)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            output = model(x_train)
            loss = self.loss_fn(output, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
        # 6. Test Accuracy
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
        self.n_gen = n_gen # จำนวนรุ่น
        
        # สร้างจำนวนวงจรเริ่มต้น n_pop ตัวใน 1 รุ่น
        self.population = [QuantumChromosome(n_gates) for _ in range(n_pop)]
        self.global_best = None
        self.history = []

    def run(self):
        # 1. Prepair Data
        x_train, y_train, x_test, y_test = self.data_mgr.get_data()
        if x_train is None:
            print("Data loading failed. Experiment cannot proceed.")
            return None, []
        
        print(f"\n🚀 Start Experiment: {self.n_gen} Generations x {len(self.population)} Pop")
        
        for gen in range(self.n_gen):
            print(f"\n--- Generation {gen+1} ---")
            
            # 2. Loop Population
            for i, chromo in enumerate(self.population):
                # A. Collapse -> Structure
                struct_code = chromo.collapse()
                
                # B. Train - Evaluate 5 Epochs
                acc = self.evaluator.evaluate(struct_code, x_train, y_train, x_test, y_test)
                chromo.fitness = acc
                print(f"  Pop {i+1}: Acc = {acc:.4f}")
                
            # 3. Find Update Global Best
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.global_best is None or current_best.fitness > self.global_best.fitness:
                self.global_best = current_best.copy()
                print(f"  🏆 New Global Best: {self.global_best.fitness:.4f}")
            
            self.history.append(self.global_best.fitness)

            # 4. Evolution (Update Genes)
            for chromo in self.population:
                chromo.update_genes(self.global_best)

        return self.global_best, self.history