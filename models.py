import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2  # Usado para manipulação de imagem
import time
import csv

# Definindo a arquitetura do modelo
def model_cartpole(input_shape, output_shape):
    model = models.Sequential()

    # Camadas densas para processamento de vetores de estado
    model.add(layers.Dense(24, activation='relu', input_shape=input_shape))  # 24 neurônios na primeira camada
    model.add(layers.Dense(24, activation='relu'))  # Segunda camada densa
    model.add(layers.Dense(output_shape, activation='softmax'))  # Número de ações depende do ambiente

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    model.summary()

    return model