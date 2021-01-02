import numpy as np

from tensorflow.keras import models, layers


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

    def extract_features(self, image):
        features = {}
        output = image
        layer_index = 0
        for layer in self.layers:
            output = layer(output)
            # TODO: MaxPool features?
            if 'conv2d' in layer.name:
                features[layer_index] = np.reshape(output.numpy(), (output.shape[1], output.shape[2], output.shape[3]))
                layer_index += 1

        return features
