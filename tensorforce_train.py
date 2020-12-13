import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorforce.agents import Agent
from tensorflow.keras import datasets

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicImageEnvironment


if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        # Parameters initialization
        batch_size = 1
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20201212-125436.h5')
        net.summary()
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        train_image = train_images[random.randint(0, len(train_images) - 1), :, :, :]
        train_image_4dim = np.reshape(train_image, (batch_size, 32, 32, 3))
        del (train_images, train_labels)
        del (test_images, test_labels)
        environment = DyadicImageEnvironment(image=train_image, net=net)
        """agent = Agent.create(agent='tensorforce', environment=environment, update=64,
                             optimizer=dict(optimizer='adam', learning_rate=1e-3),
                             objective='policy_gradient', reward_estimation=dict(horizon=20)
                             )"""
        agent = Agent.create(environment=environment,
                             policy=[
                                 [
                                     dict(type='retrieve', tensors=['observation']),
                                     # size 16x16
                                     dict(type='conv2d', size=64, window=3, padding='same'),
                                     dict(type='pool2d', reduction='max', stride=2),
                                     # size 8x8
                                     dict(type='conv2d', size=64, window=3, padding='same'),
                                     dict(type='pool2d', reduction='max', stride=2),
                                     # size 4x4
                                     dict(type='conv2d', size=64, window=3, padding='same'),
                                     dict(type='pool2d', reduction='max', stride=2),
                                     # size 2x2
                                     dict(type='conv2d', size=64, window=3, padding='same'),
                                     dict(type='pool2d', reduction='max', stride=2),
                                     # size 1x1
                                     dict(type='flatten'),
                                     dict(type='dense', size=64),
                                     dict(type='dense', size=64),
                                     dict(type='dense', size=10)
                                 ],
                                 [
                                     dict(type='retrieve', tensors=['distribution']),
                                     dict(type='flatten'),
                                     dict(type='dense', size=64),
                                     dict(type='dense', size=4)
                                 ]
                             ],
                             optimizer=dict(optimizer='adam', learning_rate=1e-3), update=64,
                             objective='policy_gradient', reward_estimation=dict(horizon=20),
                             states=dict(
                                 observation=dict(type='float', shape=(16, 16, 3)),
                                 distribution=dict(type='float', shape=10)
                             ),
                             actions=dict(type='float', shape=4)
                             )
        states = environment.reset()
        plt.imshow(np.reshape(train_image, (32, 32, 3)))
        plt.show()
        scores = net.predict(x=train_image_4dim, batch_size=batch_size)
        print('Classifier output: {label} ({n} - {c})'.format(label=scores,
                                                              n=np.argmax(scores),
                                                              c=class_names[int(np.argmax(scores))]))
        loop_number = 0
        avg_reward = 0.0
        while True:
            loop_number += 1
            actions = agent.act(states=states)
            states, reward = environment.execute(actions=actions, output=states['distribution'])
            agent.observe(reward=reward)
            avg_reward += reward
            if loop_number % 1000 == 0:
                print('Agent action: {a}    Average reward: {r}'.format(a=np.argmax(actions), r=avg_reward / 1000))
                avg_reward = 0.0
