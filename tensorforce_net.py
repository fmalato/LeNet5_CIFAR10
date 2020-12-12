import tensorflow as tf

from tensorflow.keras import models, layers
from tensorforce.agents import Agent


class DyadicConvNet(models.Sequential):

    def __init__(self, num_channels=64, input_shape=(1, 32, 32, 3)):
        super().__init__(layers=[
            layers.Conv2D(filters=num_channels, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(strides=(2, 2)),
            layers.Conv2D(filters=num_channels, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(strides=(2, 2)),
            layers.Conv2D(filters=num_channels, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(strides=(2, 2)),
            layers.Conv2D(filters=num_channels, kernel_size=(1, 1), activation='relu', padding='same'),
            layers.MaxPooling2D(strides=(2, 2)),
            layers.Conv2D(filters=num_channels, kernel_size=(1, 1), activation='relu', padding='same'),
            layers.MaxPooling2D(strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])
        self.build(input_shape=input_shape)


class DyadicBaseAgent(Agent):
    def __init__(self):
        super().__init__()

