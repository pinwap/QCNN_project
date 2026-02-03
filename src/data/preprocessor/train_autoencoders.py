import os
import sys
import logging

# Setup path
sys.path.append(os.path.join(os.getcwd(), "src"))

from data.manager import DataManager
from data.preprocessor.autoencoder import AutoencoderReducer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainAE")

def train_autoencoders():
    datasets = ["mnist", "fashionmnist"]
    dims = [16]

    # Configuration for high accuracy training
    EPOCHS = 100
    BATCH_SIZE = 128
    LR = 0.001

    for ds_name in datasets:
        logger.info(f"========================================")
        logger.info(f"Processing Dataset: {ds_name.upper()}")
        logger.info(f"========================================")

        # Load raw data (no preprocessors yet)
        try:
            # We use a safe subset size (total is ~12000 for 2 classes)
            # n_train=10000 leaves ~2000 for test, which is fine.
            dm = DataManager(
                dataset_name=ds_name,
                data_path="./data",
                n_train=10000,
                n_test=2000,
                preprocessors=["flatten"] # Flatten first, AE expects flat vectors
            )
            # We only need x_train for training the unsupervised AE
            x_train, y_train, x_test, y_test = dm.get_data()

            # Convert to torch tensor if it's numpy (DataManager usually returns tensors unless as_numpy=True)
            # Actually DataManager.get_data returns Tensors by default.

            logger.info(f"Data Loaded. Shape: {x_train.shape}")

            for dim in dims:
                logger.info(f"--- Training Autoencoder for Target Dim: {dim} ---")

                # Instantiate AutoencoderReducer
                # This will initialize the model logic
                ae = AutoencoderReducer(
                    target_dim=dim,
                    dataset_name=ds_name,
                    epochs=EPOCHS,
                    learning_rate=LR,
                    batch_size=BATCH_SIZE
                )

                # The __call__ method triggers training if model doesn't exist.
                # It returns the reduced features, but we just want to trigger the training & saving.
                _ = ae(x_train)

                logger.info(f"--- Finished AE-{dim} for {ds_name} ---")

        except Exception as e:
            logger.error(f"Failed to process dataset {ds_name}: {e}")

if __name__ == "__main__":
    train_autoencoders()