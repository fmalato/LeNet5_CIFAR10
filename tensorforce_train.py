import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorflow.keras import datasets

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicImageEnvironment


if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        # Parameters initialization
        batch_size = 1
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20201212-125436.h5')
        net.summary()
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        train_image = train_images[random.randint(0, len(train_images) - 1), :, :, :]
        train_image = np.reshape(train_image, (batch_size, 32, 32, 3))
        environment = DyadicImageEnvironment(image=train_image, net=net)
        agent = Agent.create(agent='tensorforce', environment=environment, update=64,
                             optimizer=dict(optimizer='adam', learning_rate=1e-3),
                             objective='policy_gradient', reward_estimation=dict(horizon=20)
                             )

        states = environment.reset()
        plt.imshow(np.reshape(train_image, (32, 32, 3)))
        plt.show()
        print('Classifier output: {label}'.format(label=net.predict(x=train_image, batch_size=batch_size)))
        while True:
            # TODO: states has len == 0 ?!
            actions, output = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions, output=output)
            agent.observe(terminal=terminal, reward=reward)
