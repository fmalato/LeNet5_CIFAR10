import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import Environment
from tensorflow.keras import datasets

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicImageEnvironment, DyadicConvnetGymEnv


if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        # Parameters initialization
        batch_size = 1
        steps_per_episode = 50
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

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
        # Convolutional features extraction
        net_features = net.extract_features(train_image_4dim)
        net_distribution = np.reshape(net(train_image_4dim).numpy(), (10,))
        # Environment initialization
        environment = DyadicConvnetGymEnv(features=net_features,
                                          distribution=net_distribution,
                                          max_steps=50
                                          )
        num_actions = environment.action_space.n
        environment = Environment.create(environment=environment,
                                         states=dict(
                                             features=dict(type=float, shape=(64,)),
                                             distribution=dict(type=float, shape=(10,))
                                         ),
                                         actions=dict(type=int, num_values=num_actions)
                                         )
        # Agent initialization
        agent = Agent.create(environment=environment,
                             policy=[
                                 # First module: from observation to distribution
                                 [
                                     dict(type='flatten'),
                                     dict(type='dense', size=64),
                                     dict(type='dense', size=64),
                                     dict(type='dense', size=10),
                                     dict(type='register', tensor='obs-output')
                                 ],
                                 # Second module: From distribution to actions
                                 [
                                     dict(type='retrieve', tensors=['obs-output']),
                                     dict(type='dense', size=64),
                                     dict(type='dense', size=64),
                                     dict(type='dense', size=num_actions),
                                     dict(type='register', tensor='distr-output')
                                 ]
                             ],
                             optimizer=dict(optimizer='adam', learning_rate=1e-3),
                             update=steps_per_episode,
                             objective='policy_gradient',
                             reward_estimation=dict(horizon=steps_per_episode),
                             states=dict(
                                 features=dict(type=float, shape=(64,)),
                                 distribution=dict(type=float, shape=(10,))
                             ),
                             actions=dict(type=int, num_values=num_actions)
                             )
        first_time = True
        episode = 0
        while True:
            if not first_time:
                # Extraction of a random image for next episode
                train_image = train_images[random.randint(0, len(train_images) - 1), :, :, :]
                train_image_4dim = np.reshape(train_image, (batch_size, 32, 32, 3))
                # Convolutional features extraction
                net_features = net.extract_features(train_image_4dim)
                net_distribution = np.reshape(net(train_image_4dim).numpy(), (10,))
            # Environment reset with new features and distribution
            environment.__setattr__('features', net_features)
            environment.__setattr__('distribution', net_distribution)
            states = environment.reset()
            cum_reward = 0.0
            for step in range(steps_per_episode):
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
                cum_reward += reward
            print('Episode {ep} - Cumulative Reward: {cr}'.format(ep=episode, cr=cum_reward))
            episode += 1
