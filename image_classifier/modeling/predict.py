from pathlib import Path
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt

from image_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR

def predict(model, X_test):

    pred_probs = model.predict(X_test)
    pred_labels = np.argmax(pred_probs, axis=1)

    return pred_labels

