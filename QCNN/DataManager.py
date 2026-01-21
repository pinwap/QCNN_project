# เตรียมวัตถุดิบ (โหลดรูป MNIST ย่อรูป)
from torchvision import datasets, transforms
import torch

class MNISTDataManager:
    def __init__(self, batch_size=16, n_train=400, n_test=100):
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_test = n_test

    def get_data(self):
        """
        โหลดและเตรียมข้อมูล MNIST สำหรับ QCNN
        """
        # 1. Transform: ย่อรูป 28x28 เหลือ 4x4 และแปลงเป็น Tensor
        transform = transforms.Compose([
                        transforms.Resize((4, 4)),
                        transforms.ToTensor(),
                    ])
        # โหลด MNIST 
        dataset = datasets.MNIST(root='D:/Pin/STUDY/PROJECT/senior project/QCNN_project/data',
                                 train=True, download=True, transform=transform)
        
        # 2. Filter: เอาแค่เลข 3 และ 6
        # Label: 3 , 6  (ตาม experiment 2)
        idx = (dataset.targets == 3) | (dataset.targets == 6)
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
        
        # แปลง Label เป็น 3 -> -1, 6 -> 1
        targets = dataset.targets.clone().detach()
        new_targets = torch.where(targets == 3, torch.tensor(-1.0), torch.tensor(1.0))
        
        # 3. prepocessing
        # เลือกข้อมูลมาแค่บางส่วน (Subsampling)
        data = dataset.data.float().unsqueeze(1) / 255.0 # Normalize 0-1
        # Resize จริงๆ (เพราะ dataset.data ยังเป็น 28x28)
        data = torch.nn.functional.interpolate(data, size=(4, 4), mode='bilinear')
        data = data.view(-1, 16) # Flatten เป็น 16 features
        
        # แบ่ง Train/Test
        x_train, y_train = data[:self.n_train], new_targets[:self.n_train]
        x_test, y_test = data[self.n_train:self.n_train + self.n_test], new_targets[self.n_train:self.n_train + self.n_test]
        
        print(f"✅ Data Ready: Train {x_train.shape}, Test {x_test.shape}")
        return x_train, y_train, x_test, y_test
    
