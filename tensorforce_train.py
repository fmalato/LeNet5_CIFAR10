import random
import sys
import datetime

import numpy as np
import tensorflow as tf
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorflow.keras import datasets

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicConvnetGymEnv
from utils import one_image_per_class, n_images_per_class


if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        # Network hyperparameters
        batch_size = 1
        steps_per_episode = 10
        policy_lr = 1e-3
        baseline_lr = 1e-2
        e_r = 0.05
        # Control parameters
        visualize = False
        load_checkpoint = False
        train = True
        starting_index = 0
        images_per_class = 1
        ########################### PREPROCESSING ##############################
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20210204-122725.h5')
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        # Extracting one image per class - comment for whole dataset
        indexes, labels = n_images_per_class(n=images_per_class, labels=train_labels, num_classes=len(class_names),
                                             starting_index=starting_index)
        train_images = np.array([train_images[idx] for idx in indexes])
        train_labels = np.array(labels)
        #########################################################################
        # Environment initialization
        environment = DyadicConvnetGymEnv(network=net,
                                          dataset=train_images,
                                          labels=train_labels,
                                          max_steps=steps_per_episode,
                                          visualize=visualize
                                          )
        num_actions = len(environment.actions)
        environment = Environment.create(environment=environment,
                                         states=dict(
                                             features=dict(type=float, shape=(67,)),
                                         ),
                                         actions=dict(type=int, num_values=16),
                                         max_episode_timesteps=steps_per_episode
                                         )
        # Agent initialization
        if load_checkpoint:
            directory = 'models/RL/20210216-101807/'
            old_episodes = 18000
            print('Loading checkpoint. Last episode: %d' % old_episodes)
            agent = Agent.load(directory=directory,
                               filename='agent-{x}'.format(x=old_episodes),
                               format='hdf5',
                               environment=environment,
                               agent='ppo',
                               network=[
                                       dict(type='lstm', size=64, horizon=5, activation='relu'),
                               ],
                               baseline=[
                                   dict(type='lstm', size=64, horizon=5, activation='relu')
                               ],
                               baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                               # Tensorboard initialized only if training
                               summarizer=dict(
                                   directory='data/summaries',
                                   summaries='all'
                               ) if train else None,
                               learning_rate=policy_lr,
                               batch_size=5,
                               tracking=['distribution'],
                               discount=0.99,
                               states=dict(
                                   features=dict(type=float, shape=(67,)),
                               ),
                               actions=dict(type=int, num_values=16),
                               entropy_regularization=e_r
                               )
        else:
            old_episodes = 0
            agent = Agent.create(environment=environment,
                                 agent='ppo',
                                 network=[
                                     dict(type='lstm', size=64, horizon=5, activation='relu'),
                                 ],
                                 baseline=[
                                     dict(type='lstm', size=64, horizon=5, activation='relu')
                                 ],
                                 baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                                 # Tensorboard initialized only if training
                                 summarizer=dict(
                                       directory='data/summaries',
                                       summaries='all'
                                   ) if train else None,
                                 learning_rate=policy_lr,
                                 batch_size=5,
                                 tracking=['distribution'],
                                 discount=0.99,
                                 states=dict(
                                     features=dict(type=float, shape=(67,)),
                                 ),
                                 actions=dict(type=int, num_values=16),
                                 entropy_regularization=e_r
                                 )
        # Parameters for training loop
        first_time = True
        episode = 0
        # Where to store checkpoints
        if load_checkpoint:
            save_dir = directory
        else:
            save_dir = 'models/RL/{x}/'.format(x=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # Train/test loop
        while episode <= 6000:
            state = environment.reset()
            cum_reward = 0.0
            terminal = False
            first_step = True
            if not train:
                internals = agent.initial_internals()
            # Episode loop
            while not terminal:
                if train:
                    action = agent.act(states=dict(features=state['features']))
                else:
                    action, internals = agent.act(states=dict(features=state['features']), internals=internals,
                                                  independent=True, deterministic=True)
                state, terminal, reward = environment.execute(actions=action)
                if train:
                    agent.observe(terminal=terminal, reward=reward)
                cum_reward += reward
                first_step = False
            # Stats for current episode
            sys.stdout.write('\rEpisode {ep} - Cumulative Reward: {cr}'.format(ep=episode+old_episodes, cr=cum_reward))
            sys.stdout.flush()
            episode += 1
            # Saving model every 1000 episodes
            if episode % 1000 == 0:
                agent.save(directory=save_dir,
                           filename='agent-{ep}'.format(ep=episode+old_episodes),
                           format='hdf5')
                with open(save_dir + '/parameters.txt', 'w+') as f:
                    f.write('entropy regularization: %d \n' % e_r)
                    f.write('policy learning rate: %f \n' % policy_lr)
                    f.write('baseline learning rate: %f \n' % baseline_lr)
                    f.write('episode length: %d \n' % steps_per_episode)
