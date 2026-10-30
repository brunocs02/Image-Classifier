from pathlib import Path
from loguru import logger
import numpy as np
from tensorflow import keras
import tensorflow as tf

from image_classifier.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def load_data():
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # y vem como coluna.
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    class_names = ["avião", "automóvel", "pássaro", "gato", "cervo",
                "cachorro", "sapo", "cavalo", "navio", "caminhão"]
    
    logger.success(f"Dataset loaded.")
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return (X_train, y_train), (X_test, y_test), class_names

def save_data(X_train, y_train, X_test, y_test, processed_data_dir: Path):

    processed_data_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_data_dir / "X_train.npy", X_train)
    np.save(processed_data_dir / "y_train.npy", y_train)
    np.save(processed_data_dir / "X_test.npy", X_test)
    np.save(processed_data_dir / "y_test.npy", y_test)

    logger.success(f"Dataset saved to {processed_data_dir}.")
