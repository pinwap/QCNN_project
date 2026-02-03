import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.plotter import plot_loss_history
from .base import BasePreprocessor

logger = logging.getLogger(__name__)

class ConvAutoencoderModel(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super(ConvAutoencoderModel, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: 28x28 -> 14x14
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            
            # Layer 2: 14x14 -> 7x7
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Flatten: เตรียมย่อเป็น Vector เพื่อเข้า Quantum
            nn.Flatten(),
            
            # Bottleneck: ย่อเหลือ 16 ตัวเลข (หรือตาม encoding_dim)
            # ต้องมี Sigmoid เพื่อ Angle Encoding (0...1)
            nn.Linear(32 * 7 * 7, encoding_dim),  # Bottleneck
            nn.Sigmoid() # Force latent variables to [0, 1] for Angle Encoding compatibility
        )
        # Decoder
        # ขยายกลับจาก Vector -> ภาพ จะได้เข้า ConvTranspose2d ที่รับเป็นภาพได้
        self.decoder_input = nn.Linear(encoding_dim, 32 * 7 * 7)
        
        self.decoder = nn.Sequential(
            # Reshape กลับเป็น 3D: (32, 7, 7) ต้องทำใน forward
            
            # Layer 1: 7x7 -> 14x14
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Layer 2: 14x14 -> 28x28
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output range [0, 1] assuming input images are normalized
        )

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.decoder_input(encoded)
        x = x.view(-1, 32, 7, 7) # Reshape กลับเป็นทรงกล่องสี่เหลี่ยม
        decoded = self.decoder(x)
        return encoded, decoded

class AutoencoderReducer(BasePreprocessor):
    """
    Dimensionality reduction using a simple fully-connected Autoencoder.
    Trains on the input data effectively acting as a feature extractor.
    """
    def __init__(self, target_dim: int, dataset_name: str = "unknown", epochs: int = 20, learning_rate: float = 1e-3, batch_size: int = 64):
        self.target_dim = target_dim
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model = None

        # Define path for saving/loading model weights
        # We put them in a local 'autoencoders' directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, "autoencoders")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, f"conv_ae_{dataset_name}_{target_dim}.pth")
        self.plot_dir = os.path.join(self.model_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def _train(self, flat_data: torch.Tensor):
        input_dim = flat_data.shape[1]
        
        # Initialize model
        self.model = ConvAutoencoderModel(input_dim, self.target_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Check if saved model exists
        if os.path.exists(self.model_path):
            logger.info(f"Loading pre-trained Autoencoder from: {self.model_path}")
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=device))
                # Set to eval mode is not strictly necessary here as we set it before usage,
                # but good practice.
                return
            except Exception as e:
                logger.warning(f"Failed to load model from {self.model_path}: {e}. Retraining...")

        # If not loaded, train fresh
        flat_data = flat_data.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = TensorDataset(flat_data, flat_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger.info(f"Training ConvAutoencoder (Input: {input_dim} -> Latent: {self.target_dim}) for {self.epochs} epochs on {device}...")
        
        self.model.train()
        loss_history = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, _ in dataloader:
                # แปลงร่างข้อมูลก่อนเข้า Model จาก [Batch, 784] -> [Batch, 1, 28, 28]
                img_batch = batch_x.view(-1, 1, 28, 28)
                optimizer.zero_grad()
                _, decoded = self.model(img_batch) #forward pass
                loss = criterion(decoded, img_batch) #เทียบกับ input image
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            loss_history.append(avg_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  ConvAE Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        # Plot Loss History
        plot_path = os.path.join(self.plot_dir, f"conv_ae_{self.dataset_name}_{self.target_dim}_loss.png")
        plot_loss_history(
            history=loss_history,
            save_path=plot_path,
            title=f"ConvAutoencoder Training Loss ({self.dataset_name}, Dim={self.target_dim})"
        )
        logger.info(f"Training plot saved to: {plot_path}")

        # Save trained model
        try:
            torch.save(self.model.state_dict(), self.model_path)
            logger.info(f"ConvAutoencoder model saved to: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save Autoencoder model: {e}")

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Trains the AE (if not trained) and returns the encoded features (latent space).
        """
        # Flatten input: (N, H, W) or (N, C, H, W) -> (N, Features)
        flat_data = data.view(data.shape[0], -1)
        
        # If model is not trained yet, train it on this data
        if self.model is None:
            self._train(flat_data)
        
        # Transform data using Encoder
        device = next(self.model.parameters()).device
        flat_data_dev = flat_data.to(device)
        
        img_data = flat_data_dev.view(-1, 1, 28, 28)
        
        self.model.eval()
        with torch.no_grad():
            encoded, _ = self.model(img_data)
        
        encoded = encoded.cpu()
        
        # Sigmoid activation in Encoder already ensures [0, 1] range.
        return encoded