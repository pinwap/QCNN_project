import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.plotter import plot_loss_history
from .base import BasePreprocessor

logger = logging.getLogger(__name__)

class AutoencoderModel(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super(AutoencoderModel, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, encoding_dim),  # Bottleneck
            nn.Sigmoid() # Force latent variables to [0, 1] for Angle Encoding compatibility
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Output range [0, 1] assuming input images are normalized
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
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
        self.model_path = os.path.join(self.model_dir, f"ae_{dataset_name}_{target_dim}.pth")
        self.plot_dir = os.path.join(self.model_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def _train(self, flat_data: torch.Tensor):
        input_dim = flat_data.shape[1]
        
        # Initialize model
        self.model = AutoencoderModel(input_dim, self.target_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Check if saved model exists
        if os.path.exists(self.model_path):
            logger.info(f"Loading pre-trained Autoencoder from: {self.model_path}")
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
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

        logger.info(f"Training Autoencoder (Input: {input_dim} -> Latent: {self.target_dim}) for {self.epochs} epochs on {device}...")
        
        self.model.train()
        loss_history = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                _, decoded = self.model(batch_x)
                loss = criterion(decoded, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            loss_history.append(avg_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  AE Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        # Plot Loss History
        plot_path = os.path.join(self.plot_dir, f"ae_{self.dataset_name}_{self.target_dim}_loss.png")
        plot_loss_history(
            history=loss_history,
            save_path=plot_path,
            title=f"Autoencoder Training Loss ({self.dataset_name}, Dim={self.target_dim})"
        )
        logger.info(f"Training plot saved to: {plot_path}")

        # Save trained model
        try:
            torch.save(self.model.state_dict(), self.model_path)
            logger.info(f"Autoencoder model saved to: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save Autoencoder model: {e}")

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Trains the AE (if not trained) and returns the encoded features (latent space).
        """
        # Flatten input: (N, H, W) or (N, C, H, W) -> (N, Features)
        flat_data = data_in = data.view(data.shape[0], -1)
        
        # If model is not trained yet, train it on this data
        if self.model is None:
            self._train(flat_data)
        
        # Transform data using Encoder
        device = next(self.model.parameters()).device
        flat_data_dev = flat_data.to(device)
        
        self.model.eval()
        with torch.no_grad():
            encoded, _ = self.model(flat_data_dev)
        
        encoded = encoded.cpu()
        
        # Sigmoid activation in Encoder already ensures [0, 1] range.
        return encoded