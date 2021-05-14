import os, sys
import json
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import datasets
from tensorforce_net import DyadicConvNet, NewOutputLayer

from utils import split_dataset_idxs


if __name__ == '__main__':
    # Parameters initialization
    num_epochs = 30
    batch_size = 1
    dataset_name = 'CIFAR100'
    # Network initialization
    net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
    transfer_mlp = NewOutputLayer(input_shape=(batch_size, 64))
    transfer_mlp.summary()
    # Dataset initialization
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
    with open('training_idxs_cifar100.json', 'r') as f:
        idxs = json.load(f)
        f.close()
    train_images, val_images, train_labels, val_labels = split_dataset_idxs(dataset=train_images, labels=train_labels,
                                                                            train_idxs=idxs['train'], valid_idxs=idxs['valid'])
    train_images = np.array(train_images)
    val_images = np.array(val_images)
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    train_images, test_images = train_images / 255.0, test_images / 255.0
    val_images = val_images / 255.0
    net.load_weights('models/model_CIFAR10/20210421-123951.h5')
    tensor_train_set = []
    tensor_valid_set = []
    print("Extracting training set tensors.")
    idx = 1
    for el in train_images:
        img = np.reshape(el, (1, 32, 32, 3))
        tensor_train_set.append(net.extract_dense_output(img, num_dense=2).numpy())
        sys.stdout.write('\rComputing image {current}/{num_img}'.format(current=idx, num_img=train_images.shape[0]))
        idx += 1
    print("\nExtracting validation set tensors.")
    idx = 1
    for el in val_images:
        img = np.reshape(el, (1, 32, 32, 3))
        tensor_valid_set.append(net.extract_dense_output(img, num_dense=2).numpy())
        sys.stdout.write('\rComputing image {current}/{num_img}'.format(current=idx, num_img=val_images.shape[0]))
        idx += 1
    print("\nDone. Starting training.")
    tensor_train_set = np.array(tensor_train_set)
    tensor_valid_set = np.array(tensor_valid_set)
    # Training
    with tf.device('/device:GPU:0'):
        transfer_mlp.compile(optimizer='adam',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                             metrics=['accuracy'])

        history = transfer_mlp.fit(tensor_train_set, train_labels, epochs=num_epochs,
                                   validation_data=(tensor_valid_set, val_labels))
        # Plotting accuracy
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1])
        plt.legend(loc='lower right')
        plt.show()
        # Testing
        tensor_test_set = []
        print("Extracting test set tensors.")
        idx = 1
        for el in test_images:
            img = np.reshape(el, (1, 32, 32, 3))
            tensor_test_set.append(net.extract_dense_output(img, num_dense=2).numpy())
            sys.stdout.write('\rComputing image {current}/{num_img}'.format(current=idx, num_img=train_images.shape[0]))
            idx += 1
        tensor_test_set = np.array(tensor_test_set)
        test_loss, test_acc = transfer_mlp.evaluate(test_images, test_labels, verbose=2)
    # Saving model
    print('Saving model in ' + 'models/model_' + dataset_name + '/')
    os.makedirs(os.path.dirname('models/model_' + dataset_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                + '.h5'), exist_ok=True)
    net.save('models/model_' + dataset_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
