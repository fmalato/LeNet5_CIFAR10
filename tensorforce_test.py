import os
import sys
import datetime

import numpy as np
import tensorflow as tf
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorflow.keras import datasets

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicConvnetGymEnv
from utils import split_dataset, n_images_per_class


if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        # Network hyperparameters
        batch_size = 5
        discount = 0.99
        num_classes = 10
        lstm_horizon = 5
        steps_per_episode = 15
        policy_lr = 1e-3
        baseline_lr = 1e-2
        e_r = 0.05
        # Control parameters
        visualize = True
        # Train/test parameters
        num_epochs = 1
        ########################### PREPROCESSING ##############################
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(1, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20210204-122725.h5')
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        del train_labels, train_images
        test_images = test_images / 255.0
        num_episodes = len(test_labels) * num_epochs
        num_images = len(test_labels)
        #########################################################################
        # Training environment initialization
        environment = DyadicConvnetGymEnv(network=net,
                                          dataset=test_images,
                                          labels=test_labels,
                                          max_steps=steps_per_episode,
                                          visualize=visualize,
                                          num_layers=5,
                                          class_penalty=0.01
                                          )
        num_actions = len(environment.actions)
        environment = Environment.create(environment=environment,
                                         states=dict(
                                             features=dict(type=float, shape=(67,)),
                                         ),
                                         actions=dict(type=int, num_values=num_actions+num_classes),
                                         max_episode_timesteps=steps_per_episode
                                         )
        dirs = ['models/RL/20210302-114403']
        for directory in dirs:
            print('Testing {dir}'.format(dir=directory))
            old_episodes = 200000
            print('Loading checkpoint. Last episode: %d' % old_episodes)
            agent = Agent.load(directory=directory,
                               filename='agent-{x}'.format(x=old_episodes),
                               format='hdf5',
                               environment=environment,
                               agent='ppo',
                               network=[
                                       dict(type='lstm', size=64, horizon=lstm_horizon, activation='relu'),
                               ],
                               baseline=[
                                   dict(type='lstm', size=64, horizon=lstm_horizon, activation='relu')
                               ],
                               baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                               learning_rate=policy_lr,
                               batch_size=batch_size,
                               tracking=['distribution'],
                               discount=discount,
                               states=dict(
                                   features=dict(type=float, shape=(67,)),
                               ),
                               actions=dict(type=int, num_values=num_actions+num_classes),
                               entropy_regularization=e_r
                               )
            # Parameters for test loop
            episode = 0
            correct = 0
            rewards = []
            # Test loop
            for i in range(1, num_episodes + 1):
                terminal = False
                ep_reward = 0
                obs = environment.reset()
                internals_valid = agent.initial_internals()
                while not terminal:
                    action, internals = agent.act(states=dict(features=obs['features']), internals=internals_valid,
                                                  independent=True, deterministic=True)
                    environment.environment.set_agent_classification(agent.tracked_tensors()['agent/policy/action_distribution/probabilities'])
                    state, terminal, reward = environment.execute(actions=action)
                    if terminal:
                        if action == test_labels[i - 1]:
                            correct += 1
                    ep_reward += reward
                rewards.append(ep_reward)
                avg_reward = np.sum(rewards) / len(rewards)
                sys.stdout.write('\rValidation: Episode {ep} - Average reward: {cr} - Correct: {ok}%'.format(ep=i,
                                                                                                             cr=round(avg_reward, 3),
                                                                                                             ok=round((correct / i) * 100, 2)))
                sys.stdout.flush()
            print('\n')
