import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils.plotter import plot_train_val_loss
from .base import BasePreprocessor

logger = logging.getLogger(__name__)

class ConvAutoencoderModel(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super(ConvAutoencoderModel, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: 28x28 -> 14x14
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            # Layer 2: 14x14 -> 7x7
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # Flatten: 32*7*7 = 1568
            nn.Flatten(),

            # Partial reduction before bottleneck
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),

            # Bottleneck
            nn.Linear(128, encoding_dim),
            nn.Sigmoid() # Force latent variables to [0, 1] for Angle Encoding
        )

        # Decoder Input
        self.decoder_input = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 7 * 7),
            nn.ReLU()
        )

        # Decoder CNN
        self.decoder = nn.Sequential(
            # Layer 1: 7x7 -> 14x14
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 2: 14x14 -> 28x28
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.decoder_input(encoded)
        x = x.view(-1, 32, 7, 7) # Reshape back to feature map
        decoded = self.decoder(x)
        return encoded, decoded

class AutoencoderReducer(BasePreprocessor):
    def __init__(self, target_dim: int, dataset_name: str = "unknown", epochs: int = 20, learning_rate: float = 1e-3, batch_size: int = 64):
        self.target_dim = target_dim
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model = None

        # Define path for saving/loading model weights
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

        # Determine device: MPS (Apple Silicon) -> CUDA -> CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

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

        # Split dataset into train and val (80-20 split)
        val_size = max(int(len(dataset) * 0.2), 1)
        if val_size >= len(dataset):
            val_size = max(len(dataset) - 1, 0)
        train_size = len(dataset) - val_size
        if train_size <= 0:
            train_dataset = dataset
            val_dataset = None
        else:
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=generator
            )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_dataset, batch_size=self.batch_size)
            if val_dataset is not None and val_size > 0
            else None
        )

        logger.info(f"Training ConvAutoencoder (Input: {input_dim} -> Latent: {self.target_dim}) for {self.epochs} epochs on {device}...")

        self.model.train()
        train_history = []
        val_history = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, _ in train_loader:
                # แปลงร่างข้อมูลก่อนเข้า Model จาก [Batch, 784] -> [Batch, 1, 28, 28]
                img_batch = batch_x.view(-1, 1, 28, 28)
                optimizer.zero_grad()
                _, decoded = self.model(img_batch) #forward pass
                loss = criterion(decoded, img_batch) #เทียบกับ input image
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_history.append(avg_loss)

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_total = 0.0
                    for batch_x, _ in val_loader:
                        img_batch = batch_x.view(-1, 1, 28, 28)
                        _, decoded = self.model(img_batch)
                        val_total += criterion(decoded, img_batch).item()
                val_avg = val_total / len(val_loader)
                val_history.append(val_avg)
                self.model.train()
            else:
                val_history.append(None)

            # Logging every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                msg = f"  ConvAE Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}"
                if val_history[-1] is not None:
                    msg += f", Val Loss: {val_history[-1]:.4f}"
                logger.info(msg)

        # Plot Loss History
        plot_path = os.path.join(self.plot_dir, f"conv_ae_{self.dataset_name}_{self.target_dim}_loss.png")
        plot_train_val_loss(
            train_history=train_history,
            val_history=val_history if any(v is not None for v in val_history) else None,
            save_path=plot_path,
            title=f"Autoencoder Train vs Val Loss ({self.dataset_name}, Dim={self.target_dim})"
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
            encoded, decoded = self.model(img_data)

        encoded = encoded.cpu()
        # decoded = decoded.cpu() # unused for downstream tasks
        
        return encoded