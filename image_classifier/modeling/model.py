from loguru import logger
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)

def create_model():
    """
    Cria e retorna um modelo de rede neural convolucional (CNN) para classificação de imagens.
    O modelo é projetado para trabalhar com imagens de entrada de tamanho 32x32x3 (como as do CIFAR-10).
    """
    model = keras.Sequential()

    # Entrada
    model.add(Input(shape=(32, 32, 3)))

    # -------------------------
    # Bloco 1
    # -------------------------
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(BatchNormalization())

    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    # -------------------------
    # Bloco 2
    # -------------------------
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(BatchNormalization())

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    # -------------------------
    # Bloco 3
    # -------------------------
    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(BatchNormalization())

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.30))


    # -------------------------
    # CNN → Rede totalmente conectada
    # -------------------------
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # CIFAR-10 possui 10 classes
    model.add(Dense(10, activation='softmax'))


    # -------------------------
    # Compilação
    # -------------------------
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.success("Successfully created and compiled the CNN model.")
    model.summary()