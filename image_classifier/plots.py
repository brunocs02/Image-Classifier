from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

from image_classifier.config import FIGURES_DIR, PROCESSED_DATA_DIR

def example_plot(X_train, y_train, class_names):

    plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_train[i])
        plt.title(class_names[y_train[i]])
        plt.axis('off')

    plt.suptitle("Exemplos da base CIFAR-10.")
    plt.savefig(FIGURES_DIR / "cifar10_example.png")
    plt.close()

    logger.success(f"Exemplos salvos em {FIGURES_DIR / 'cifar10_example.png'}.")

def plot_training_history(history):

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label="Treino")
    plt.plot(history.history['val_accuracy'], label="Validação")
    plt.title("Acurácia por época")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.savefig(FIGURES_DIR / "training_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label="Treino")
    plt.plot(history.history['val_loss'], label="Validação")
    plt.title("Loss por época")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(FIGURES_DIR / "training_loss.png")
    plt.close()

def plot_predictions(X_test, y_test, pred_labels, class_names):

    plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_test[i])
        real = class_names[y_test[i]]
        pred = class_names[pred_labels[i]]
        plt.title(f"Real: {real}\nPrediction: {pred}")
        plt.axis('off')

    plt.suptitle("Predictions on Test Set")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predictions.png")
    plt.close()

    logger.success(f"Predictions saved to {FIGURES_DIR / 'predictions.png'}.")

def plot_confusion_matrix(confusion_matrix, class_names):

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - CIFAR-10')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png")
    plt.close()

    logger.success(f"Confusion matrix saved to {FIGURES_DIR / 'confusion_matrix.png'}.")