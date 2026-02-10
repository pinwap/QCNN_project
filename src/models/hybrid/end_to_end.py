import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Any
import logging
from torch.autograd import Function
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit import ParameterVector
import numpy as np

from ..qcnn.standard_qcnn import StandardQCNN
from ..feature_maps import FeatureMapBuilder

logger = logging.getLogger(__name__)

class HybridAutoencoderQCNN(nn.Module):
    """
    Hybrid End-to-End Model: 
    CNN Encoder (Classical) -> Latent Vector -> Quantum Circuit (QCNN) -> Measurement -> Classification
    """
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        num_qubits: int, 
        qcnn_structure: List[int] = None,
        feature_map_type: str = "angle",
        pretrained_encoder_path: Optional[str] = None,
        freeze_encoder: bool = False
    ):
        super(HybridAutoencoderQCNN, self).__init__()
        
        self.num_qubits = num_qubits
        if qcnn_structure is None:
             # Default structure if not provided (placeholder)
             # In a real scenario, this should come from evolutionary search or config
             qcnn_structure = [0] * num_qubits # Example
        
        # 1. Classical Encoder Part (Reusing ConvAutoencoderModel's encoder)
        # We need to construct the exact same structure or load it
        # Ideally, we should import the class definition to avoid duplication
        # But for now, let's redefine the encoder part here to be self-contained or import it if possible.
        # Let's import the one from data.preprocessor.autoencoder to ensure consistency.
        from data.preprocessor.autoencoder import ConvAutoencoderModel
        
        dummy_ae = ConvAutoencoderModel(input_dim, encoding_dim) #สร้างวงจร autoencoderจาก class ConvAutoencoderModel ที่เราเคยสร้างไว้เพื่อให้ได้ encoder ที่มีโครงสร้างเดียวกันกับที่เราใช้ฝึกก่อนหน้านี้ 
        self.encoder = dummy_ae.encoder #เอาแค่ส่วน encoder มาใช้ในโมเดลนี้
        
        if pretrained_encoder_path:
            logger.info(f"Loading pretrained encoder from {pretrained_encoder_path}")
            checkpoint = torch.load(pretrained_encoder_path) 
            #โหลด state dict ของโมเดลที่เราเคยฝึกไว้ก่อนหน้านี้มาใช้ใน encoder ตัวนี้
            encoder_state_dict = {
                k.replace('encoder.', ''): v 
                for k, v in checkpoint.items() 
                if k.startswith('encoder.')
            }
            # กันเหนียวเผื่อว่า checkpoint ที่โหลดมาอาจจะเป็น state dict ของโมเดลทั้งตัวเลยก็ได้ ถ้าไม่มี prefix 'encoder.' ก็ลองดูว่ามี key ที่ตรงกับ encoder เลยไหม
            if not encoder_state_dict:
                 # Maybe the checkpoint IS the state dict of the model directly
                 # If the keys match exactly
                 encoder_state_dict = {
                    k.replace('encoder.', ''): v 
                    for k, v in checkpoint.get('state_dict', checkpoint).items() 
                    if k.startswith('encoder.')
                 }
            # โหลดน้ำหนักลงใน encoder ของโมเดลนี้
            self.encoder.load_state_dict(encoder_state_dict, strict=False)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False #ให้ encoder ไม่ถูกอัพเดตน้ำหนักระหว่างการฝึก
            logger.info("Classical Encoder Frozen.")
        else:
            # End-to-End
            for param in self.encoder.parameters():
                param.requires_grad = True
            logger.info("Classical Encoder Unfrozen (Trainable).")

        # 2. Quantum Component
        # Latent vector from Encoder (size = encoding_dim) -> Quantum Feature Map -> QCNN
        if encoding_dim != num_qubits:
            # ถ้า dimension ของ latent vector ไม่เท่ากับจำนวน qubits ที่จะใช้ใน QCNN จะ projection = linear layer เพื่อแปลงขนาด
             logger.warning(f"Encoding dim ({encoding_dim}) != num_qubits ({num_qubits}). Adding a projection layer.")
             self.projection = nn.Linear(encoding_dim, num_qubits)
             self.use_projection = True
        else:
             self.use_projection = False

        # สร้าง QCNN Circuit เปล่า
        self.qcnn_wrapper = StandardQCNN(num_qubits=num_qubits)
        self.qc = QuantumCircuit(num_qubits)
        
        # Feature Map
        from models.feature_maps.factory import resolve_feature_map
        # resolve_feature_map returns an INSTANCE of the Builder (initialized with no args usually)
        fm_builder_instance = resolve_feature_map(feature_map_type)
        # Then call build(num_qubits)
        # Returns (QuantumCircuit, ParameterVector/List)
        self.feature_map_circuit, self.feature_map_params = fm_builder_instance.build(num_qubits)
        self.qc.compose(self.feature_map_circuit, inplace=True)
        
        # QCNN Ansatz
        self.ansatz, self.ansatz_params = self.qcnn_wrapper.build()
        self.qc.compose(self.ansatz, inplace=True)
        
        # Measurement: Measure the last remaining qubit (usually qubit 0 or defined by pooling)
        _, _, last_qubit_idx = self.qcnn_wrapper.build_with_metadata()
        
        # Fix observable construction:
        from qiskit.quantum_info import SparsePauliOp
        # สร้าง Observable "Z" เฉพาะที่ตัวสุดท้าย (ตัวอื่นเป็น "I" คือ Identity/ไม่ทำอะไร)
        # ลำดับใน Qiskit เรียงกลับหลัง (q_n ... q_0) เลยต้องใช้ num_qubits - 1 - last_qubit_idx
        op_list = ["I"] * num_qubits
        op_list[num_qubits - 1 - last_qubit_idx] = "Z"
        op_str = "".join(op_list)
        # สร้างตัว Observable ที่คิวบิตสุดท้าย
        observable = SparsePauliOp.from_list([(op_str, 1)])
        
        # ใช้ torch connector เพื่อเชื่อมต่อ QNN กับ PyTorch ให้รับ input เป็น tensor ได้ และวัดผลลัพธ์เป็น tensor
        self.qnn = EstimatorQNN(
            circuit=self.qc,
            input_params=self.feature_map_params,
            weight_params=self.ansatz_params,
            observables=observable
        )
        self.quantum_layer = TorchConnector(self.qnn)

    def forward(self, x):
        # x is [Batch, 1, 28, 28] image
        
        # 1. Classical Encoding : รับภาพเข้าไปใน encoder เพื่อย่อเป็น latent vector
        # The ConvAutoencoderModel's encoder ends with Sigmoid, so output is [0, 1]
        latent = self.encoder(x) # [Batch, encoding_dim]

        # 2. ทำให้ค่า 0-1 เป็น 0-2pi 
        # Projection ถ้า encoding_dim ไม่เท่ากับ num_qubits ให้แปลงขนาดด้วย linear layer
        if self.use_projection:
             latent = self.projection(latent)
             # Apply Sigmoid -> Scale to 2pi
             latent = torch.sigmoid(latent) * (2 * np.pi)
        else:
             # Latent is already sigmoid [0, 1] from encoder
             # Scale to 2*PI for full rotation coverage in Angle Embedding
             # This effectively maps [0, 1] -> [0, 2pi]
             latent = latent * (2 * np.pi)

        # 3. Quantum Layer 
        # เอา latent vector (Batch, num_qubits) มาใส่ใน feature map แล้วส่งผ่าน ansatz ของ QCNN 
        # เพื่อวัดผลลัพธ์เป็นค่า expectation ของ observable <Z> ซึ่งจะอยู่ในช่วง [-1, 1] 
        output = self.quantum_layer(latent) 
        
        # 4. ส่งออกค่าทำนายผล -1 , 1
        return output