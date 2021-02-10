from sklearn.linear_model import LinearRegression

from tensorforce_net import DyadicConvNet
from utils import one_image_per_class
from tensorflow.keras import datasets

import numpy as np

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# Network initialization
net = DyadicConvNet(num_channels=64, input_shape=(1, 32, 32, 3))
net.load_weights('models/model_CIFAR10/20210204-122725.h5')
# Dataset initialization
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
# Extracting one image per class
indexes, labels = one_image_per_class(train_labels, len(class_names))
train_images = np.array([train_images[idx] for idx in indexes])
train_labels = np.array(labels)
data = []
for el in train_images:
    train_image_4dim = np.reshape(el, (1, 32, 32, 3))
    # Convolutional features extraction
    fv = np.reshape(net.extract_features(train_image_4dim)[4], (64,))
    data.append(fv)

lrc = LinearRegression()

lrc.fit(np.array(data), train_labels)
p = lrc.predict(np.array(data))
for i in range(len(p)):
    print('predicted: {p}    label: {l}'.format(p=p[i], l=train_labels[i]))
