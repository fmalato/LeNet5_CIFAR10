import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets
from tensorforce_net import DyadicConvNet


if __name__ == '__main__':
    # Parameters initialization
    num_epochs = 10
    batch_size = 128
    dataset_name = 'CIFAR10'
    # Network initialization
    net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
    net.summary()
    # Dataset initialization
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    # Training
    with tf.device('/device:GPU:0'):
        net.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        history = net.fit(train_images, train_labels, epochs=10,
                          validation_data=(test_images, test_labels))
        # Plotting accuracy
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1])
        plt.legend(loc='lower right')
        plt.show()
        # Testing
        test_loss, test_acc = net.evaluate(test_images, test_labels, verbose=2)
    # Saving model
    print('Saving model in ' + 'models/model_' + dataset_name + '/')
    os.makedirs(os.path.dirname('models/model_' + dataset_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                + '.h5'), exist_ok=True)
    net.save('models/model_' + dataset_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
    # TODO: either make it useful or delete it
    with open('models/model_' + dataset_name + '/' + 'parameters.txt', 'w+') as f:
        f.write('num epochs: ' + str(num_epochs) + '\n')
        f.write('batch size: ' + str(batch_size) + '\n')
        f.write('dataset: ' + dataset_name)



