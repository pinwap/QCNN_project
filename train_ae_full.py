import sys
import os
import torch
import logging

# Add 'src' to python path because 'autoencoder.py' imports 'utils' as a top-level module
sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.getcwd())

from src.data.dataset.fashion_mnist_dataset import FashionMNISTDataset
from src.data.preprocessor.autoencoder import AutoencoderReducer

# Configure logging to see progress
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
logger = logging.getLogger("train_ae_full")

def main():
    logger.info("Starting Full FashionMNIST Autoencoder Training (Classes 0-9)")

    # 1. Load Full Dataset
    # target_labels=None loads all classes (thanks to our fix in fashion_mnist_dataset.py)
    logger.info("Loading FashionMNIST Dataset (All Classes)...")
    dataset = FashionMNISTDataset(
        data_dir="./data", 
        n_train=60000, 
        n_test=10000, 
        target_labels=None 
    )
    
    # Load raw data (normalized to [0,1])
    data, labels = dataset.load()
    logger.info(f"Loaded Data Shape: {data.shape}")
    logger.info(f"Loaded Labels Shape: {labels.shape}")
    
    # 2. Initialize Autoencoder Reducer
    # This will set up the model path. Since we renamed the old one, it should train fresh.
    # We increase epochs to 50 for better convergence as requested.
    reducer = AutoencoderReducer(
        target_dim=16, 
        dataset_name="fashionmnist", 
        epochs=50, 
        batch_size=64, 
        learning_rate=1e-3
    )
    
    logger.info(f"Autoencoder Model Path: {reducer.model_path}")
    if os.path.exists(reducer.model_path):
        logger.warning(f"Warning: Model file already exists at {reducer.model_path}. It might be loaded instead of retraining!")
        # We expect it NOT to exist because we renamed it. 
        # If it exists, we force removal to ensure fresh training.
        try:
            os.remove(reducer.model_path)
            logger.info("Removed existing model file to force retraining.")
        except Exception as e:
            logger.error(f"Could not remove existing model: {e}")

    # 3. Train
    # calling reducer(data) triggers _train checks
    logger.info("Triggering Autoencoder Training...")
    encoded_features = reducer(data)
    
    logger.info("Training Complete.")
    logger.info(f"Encoded Features Shape: {encoded_features.shape}")
    logger.info(f"New model saved to: {reducer.model_path}")

if __name__ == "__main__":
    main()
