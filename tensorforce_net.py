import tensorflow as tf

from tensorflow.keras import models, layers


class DyadicConvNet(models.Sequential):

    def __init__(self, num_channels=64, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()

        self._layers = [
            ConvBlock(num_channels, kernel_size, stride),
            ConvBlock(num_channels, kernel_size, stride),
            ConvBlock(num_channels, kernel_size, stride),
            ConvBlock(num_channels, kernel_size, stride),
            ConvBlock(num_channels, kernel_size, stride),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ]


class ConvBlock(layers.Layer):

    def __init__(self, num_channels=64, kernel_size=(3, 3), stride=(1, 1), name='ConvBlock'):
        super().__init__()

        self.conv = layers.Conv2D(filters=num_channels, kernel_size=kernel_size, strides=stride, activation='relu')
        self.pool = layers.MaxPool2D(pool_size=(2, 2))
        self._name = name


