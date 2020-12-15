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
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        visualization = False
        # Image is 32x32, make sure grid_rows is a divisor
        grid_row = 2
        num_actions = pow(grid_row, 2)
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20201212-125436.h5')
        net.summary()
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        # Extraction of a random image
        train_image = train_images[random.randint(0, len(train_images) - 1), :, :, :]
        train_image_4dim = np.reshape(train_image, (batch_size, 32, 32, 3))
        # TODO: Not sure about that, but if we need just a single image there's no need for these
        del (train_images, train_labels)
        del (test_images, test_labels)
        # Environment initialization
        environment = DyadicImageEnvironment(image=train_image, net=net, grid_scale=grid_row)
        # Agent initialization
        agent = Agent.create(environment=environment,
                             policy=[
                                 # First module: from observation to distribution
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
                                     dict(type='dense', size=10),
                                     dict(type='register', tensor='obs-output')
                                 ],
                                 # Second module: From distribution to actions
                                 [
                                     dict(type='retrieve', tensors=['obs-output']),
                                     dict(type='flatten'),
                                     dict(type='dense', size=64),
                                     dict(type='dense', size=num_actions),
                                     dict(type='register', tensor='distr-output')
                                 ],
                                 # Third module: Concatenates outputs from previous modules
                                 [
                                     dict(type='retrieve', aggregation='concat', tensors=['obs-output', 'distr-output'])
                                 ]
                             ],
                             optimizer=dict(optimizer='adam', learning_rate=1e-3), update=1,
                             objective='policy_gradient', reward_estimation=dict(horizon=20),
                             states=dict(
                                 observation=dict(type='float', shape=(int(32 / grid_row), int(32 / grid_row), 3)),
                             ),
                             # WORKAROUND: Technically it's a (distribution, actions) tuple
                             actions=dict(type='float', shape=10+num_actions)
                             )
        states = environment.reset()
        # Fancy (but not needed) visualization of the training image
        if visualization:
            scores = environment.prediction
            plt.imshow(train_image)
            plt.show()
            print('Classifier output: {label} ({n} - {c})'.format(label=scores,
                                                                  n=np.argmax(scores),
                                                                  c=class_names[int(np.argmax(scores))]))
        # Stats initialization
        loop_number = 0
        avg_reward = 0.0
        rewards = []
        # Infinite loop: at each timestep, we reward the agent based on how well it performed wrt the classifier
        while True:
            loop_number += 1
            output = agent.act(states=states)
            # As previously said, the policy outputs are concatenated. Here we separate them
            distribution = output[0:10]
            actions = output[10:]
            states, reward = environment.execute(actions=actions, output=distribution)
            agent.observe(reward=reward)
            # Stats update
            avg_reward += reward
            rewards.append(reward)
            if loop_number % 1000 == 0:
                hist = np.histogram(rewards, [-1.0, 0.0, 1.0, 1.8])
                print('Step: {ln}    Average reward: {r}'.format(ln=loop_number, r=avg_reward / 1000))
                print('Failures: {f}   Worse than classifier: {w}   Better than classifier: {g}'.format(
                    f=hist[0][0], w=hist[0][1], g=hist[0][2]
                ))
                print('---------------------------------------------------------------------------------')
                # Stats reset
                avg_reward = 0.0
                rewards = []
