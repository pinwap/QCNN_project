import logging
import time
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.hybrid.end_to_end import HybridAutoencoderQCNN
from training.strategies.base import EvaluationStrategy

logger = logging.getLogger(__name__)

class HybridEndToEndStrategy(EvaluationStrategy):
    """
    Evaluation Strategy for Hybrid End-to-End training (CNN Encoder + QCNN).
    """
    def __init__(
        self,
        num_qubits: int,
        epochs: int = 10,
        lr: float = 0.001,
        input_dim: int = 1, # Channel dim (1 for MNIST/FashionMNIST)
        encoding_dim: int = 16, # Should match num_qubits ideally or be projected
        device: Optional[str] = None,
        pretrained_encoder_path: Optional[str] = None
    ):
        self.num_qubits = num_qubits
        self.epochs = epochs
        self.lr = lr
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_encoder_path = pretrained_encoder_path

    '''
    เป็นตัวเรียกใช้โมเดล HybridAutoencoderQCNN ที่เราได้สร้างไว้ในไฟล์ end_to_end.py เพื่อฝึกและประเมินผลโมเดลนี้บนชุดข้อมูลที่ให้มา โดยจะมีขั้นตอนหลักๆ ดังนี้:
    1. เตรียมโมเดล: สร้าง instance ของ HybridAutoencoderQCNN
    2. เตรียม DataLoader สำหรับชุดข้อมูลฝึกและทดสอบ
    3. วนลูปฝึกโมเดลตามจำนวน epoch ที่กำหนด โดยคำนวณ loss และความแม่นยำในแต่ละ epoch
    4. ประเมินผลบนชุดทดสอบและคืนค่าความแม่นยำพร้อมกับ state dict ของโมเดลที่ดีที่สุด
    '''
    def evaluate(
        self,
        structure_code: list[int], # Unused for standard QCNN
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        save_dir: str = "./output" # Added save_dir
    ) -> Tuple[float, Any]: # คืนค่าความแม่นยำและ state dict ของโมเดลที่ดีที่สุด
        
        # 1. Prepare Model
        model = HybridAutoencoderQCNN(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            num_qubits=self.num_qubits,
            pretrained_encoder_path=self.pretrained_encoder_path,
            freeze_encoder=False # Always Unfreeze for End-to-End training
        ).to(self.device)
        
        # 2. Prepare DataLoaders
        # Ensure correct shape for CNN: [N, C, H, W]
        if x_train.dim() == 2: # [N, Features] e.g. [N, 784]
             # Reshape to [N, 1, 28, 28] assuming 28x28
             size = int(x_train.shape[1]**0.5)
             x_train = x_train.view(-1, 1, size, size) #-1 คือ จำนวนรูปทั้งหมด 1 คือ channel (ขาวดำ) size*size = 28*28 pixel
             x_test = x_test.view(-1, 1, size, size)
             if x_val is not None:
                 x_val = x_val.view(-1, 1, size, size)
        
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        if x_val is not None:
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=32)
        else:
            val_loader = None
            
        # 3. Training Loop
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss_func = nn.MSELoss() # Common for QNN outputting expectation values [-1, 1]
        
        best_val_acc = 0.0 #เก็บคะแนนสูงสุดของ validation ที่เคยทำได้
        best_model_state = None #เก็บ state dict =ร่างทองของโมเดลที่ให้ผลลัพธ์ validation ดีที่สุด
        
        history = {
            "loss": [],
            "train_acc": [],
            "val_acc": []
        }

        model.train() #บอก torch ให้เปิดการฝึกโมเดล (เปิด gradient, dropout,ฯลฯ)
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = model(batch_x) # เรียกใช้ farward [Batch, 1]
                
                loss = loss_func(output, batch_y.view(-1, 1)) # reshape batch_y เป็น [Batch, 1] แนวตั้งให้ตรงกับ output
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * batch_x.size(0) # สะสม loss ของ batch นี้ คูณด้วยขนาด batch เพื่อให้ได้ loss รวม
                
                # คำนวณ accuracy
                preds = torch.sign(output).flatten()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
            
            # สรุปผลของ epoch นี้  
            avg_loss = total_loss / total
            train_acc = correct / total
            
            # Validation
            val_acc = 0.0
            if val_loader:
                model.eval() # เปลี่ยนโมเดลเป็นโหมดประเมินผล (ปิด gradient, dropout,ฯลฯ)
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        output = model(batch_x)
                        # คำนวณ val accuracy
                        preds = torch.sign(output).flatten()
                        val_correct += (preds == batch_y).sum().item()
                        val_total += batch_y.size(0)
                val_acc = val_correct / val_total
                model.train() # สอบเสร็จแล้ว บอก torch ให้เปิดการฝึกโมเดล (เปิด gradient, dropout,ฯลฯ)
                
                # model checkpointing กัน overfit : เก็บร่างที่valดีที่สุด
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()
            
            history["loss"].append(avg_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        # 4. Final Evaluation on Test Set : ใช้ร่างที่ดีที่สุดที่ได้จาก val มาทดสอบบน test set ไม่ได้ใช้ร่าง epoch สุดท้ายที่ฝึก
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        model.eval()
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32)
        # คำนวณ test accuracy
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = model(batch_x)
                preds = torch.sign(output).flatten()
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)
                
        test_acc = test_correct / test_total
        logger.info(f"Final Test Accuracy: {test_acc:.4f}")
        
        # --- Visualization Start ---
        try:
            self._visualize_latent(model, x_test, y_test, save_dir)
        except Exception as e:
            logger.error(f"Failed to visualize latent space: {e}")
        # --- Visualization End ---
        
        return test_acc, history, model.state_dict()

    def _visualize_latent(self, model, x_test, y_test, save_dir):
        """
        Visualize latent vectors for a few samples from each class.
        Includes original images and their 16-dim latent representations.
        """
        logger.info("Generating Latent Space Visualization...")
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # pick first few samples.
        num_samples = 4
        indices = range(min(num_samples, len(x_test)))
        
        fig, axes = plt.subplots(len(indices), 2, figsize=(10, 4 * len(indices)))
        if len(indices) == 1: axes = axes.reshape(1, -1)
        
        model.eval()
        with torch.no_grad():
            for i, idx in enumerate(indices):
                img = x_test[idx].unsqueeze(0).to(self.device) # [1, 1, 28, 28]
                target = y_test[idx].item()
                
                # Encode
                latent = model.encoder(img)
                latent_vec = latent.cpu().numpy().flatten()
                
                # Predict
                output = model(img).item()
                pred_label = 1 if output > 0 else -1
                
                # Plot Image
                ax_img = axes[i, 0]
                img_np = img.cpu().squeeze().numpy()
                ax_img.imshow(img_np, cmap='gray')
                ax_img.set_title(f"Sample {idx}\nTarget: {target}")
                ax_img.axis('off')
                
                # Plot Latent
                ax_lat = axes[i, 1]
                ax_lat.bar(range(len(latent_vec)), latent_vec, color='teal')
                ax_lat.set_ylim(0, 1)
                ax_lat.set_title(f"Latent Rep. (Pred: {output:.2f})")
                ax_lat.set_xlabel("Dim")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(plots_dir, f"latent_vis_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Latent visualization saved to {save_path}")

