import random
import sys
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorflow.keras import datasets

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicConvnetGymEnv
from tracked_dense import TrackedDense
from grid_drawer import AgentSprite, Drawer
from utils import one_image_per_class


if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        # Parameters initialization
        batch_size = 1
        steps_per_episode = 1
        policy_lr = 1e-3
        baseline_lr = 1e-2
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        e_r = 0.05
        visualize = False
        load_checkpoint = False
        train = True
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20210204-122725.h5')
        #net.summary()
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        # Extracting one image per class
        indexes, labels = one_image_per_class(train_labels, len(class_names))
        train_images = np.array([train_images[idx] for idx in indexes])
        train_labels = np.array(labels)
        # Extraction of a random image - For now extraction of first image
        image_index = 0
        train_image = train_images[image_index, :, :, :]
        train_label = int(train_labels[image_index])
        train_image_4dim = np.reshape(train_image, (batch_size, 32, 32, 3))
        # Convolutional features extraction
        net_features = net.extract_features(train_image_4dim)
        net_distribution = np.reshape(net(train_image_4dim).numpy(), (10,))
        # Environment initialization
        environment = DyadicConvnetGymEnv(features=net_features,
                                          image_class=train_label,
                                          distribution=net_distribution,
                                          max_steps=steps_per_episode
                                          )
        num_actions = len(environment.actions)
        environment = Environment.create(environment=environment,
                                         states=dict(
                                             # 64 features + 3 positional coding
                                             features=dict(type=float, shape=(64,)),
                                         ),
                                         actions=dict(
                                             classification=dict(type=int, num_values=len(class_names))
                                         ),
                                         max_episode_timesteps=steps_per_episode
                                         )
        # Agent initialization
        if load_checkpoint:
            directory = 'models/RL/20210209-111540/'
            old_episodes = 75000
            print('Loading checkpoint. Last episode: %d' % old_episodes)
            agent = Agent.load(directory=directory,
                               filename='agent-{x}'.format(x=old_episodes),
                               format='hdf5',
                               environment=environment,
                               agent='ppo',
                               network=[
                                       dict(type='dense', size=64, activation='relu'),
                               ],
                               baseline=[
                                   dict(type='dense', size=64, activation='relu')
                               ],
                               baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                               summarizer=dict(
                                   directory='data/summaries',
                                   summaries='all'
                               ) if train else None,
                               learning_rate=policy_lr,
                               batch_size=10,
                               tracking=['distribution'],
                               discount=0.99,
                               states=dict(
                                   # 64 features + 3 positional coding
                                   features=dict(type=float, shape=(64,)),
                               ),
                               actions=dict(
                                   classification=dict(type=int, num_values=len(class_names)),
                               ),
                               entropy_regularization=e_r
                               )
        else:
            old_episodes = 0
            agent = Agent.create(environment=environment,
                                 agent='ppo',
                                 network=[
                                     dict(type='dense', size=64, activation='relu'),
                                 ],
                                 baseline=[
                                     dict(type='dense', size=64, activation='relu')
                                 ],
                                 baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                                 summarizer=dict(
                                       directory='data/summaries',
                                       summaries='all'
                                   ) if train else None,
                                 learning_rate=policy_lr,
                                 batch_size=10,
                                 tracking=['distribution'],
                                 discount=0.99,
                                 states=dict(
                                       # 64 features + 3 positional coding
                                       features=dict(type=float, shape=(64,)),
                                   ),
                                 actions=dict(
                                     classification=dict(type=int, num_values=len(class_names)),
                                 ),
                                 entropy_regularization=e_r
                                 )
        first_time = True
        episode = 0
        if load_checkpoint:
            save_dir = directory
        else:
            save_dir = 'models/RL/{x}/'.format(x=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if visualize:
            # Visualization objects
            tile_width = 10
            num_layers = 5
            agent_sprite = AgentSprite(rect_width=tile_width, num_layers=num_layers)
            drawer = Drawer(agent_sprite, num_layers=num_layers, tile_width=tile_width)
        while True:
            if not first_time:
                # Extraction of a random image for next episode
                image_index = episode % len(class_names)
                train_image = train_images[image_index, :, :, :]
                train_label = int(train_labels[image_index])
                train_image_4dim = np.reshape(train_image, (batch_size, 32, 32, 3))
                # Convolutional features extraction
                net_features = net.extract_features(train_image_4dim)
                net_distribution = np.reshape(net(train_image_4dim).numpy(), (10,))
                # Environment reset with new features and distribution
                environment.environment.features = net_features
                environment.environment.distribution = net_distribution
                environment.environment.image_class = train_label
            else:
                first_time = False
            state = environment.reset()
            cum_reward = 0.0
            terminal = False
            if not train:
                internals = agent.initial_internals()
            while not terminal:
                if train:
                    action = agent.act(states=dict(features=state['features']))
                else:
                    action, internals = agent.act(states=dict(features=state['features']), internals=internals,
                                                  independent=True, deterministic=False)
                distrib = agent.tracked_tensors()['agent/policy/classification_distribution/probabilities']
                environment.environment.agent_classification = distrib
                state, terminal, reward = environment.execute(actions=action)
                if train:
                    agent.observe(terminal=terminal, reward=reward)
                cum_reward += reward
                if visualize:
                    print('Correct label: {l} - Predicted label: {p} - Action: {a}'.format(l=class_names[train_label],
                                                                             p=class_names[int(np.argmax(distrib))],
                                                                                           a=action['movement']))
                    drawer.render(agent=agent_sprite)
                    agent_sprite.move(environment.environment.agent_pos)
            sys.stdout.write('\rEpisode {ep} - Cumulative Reward: {cr}'.format(ep=episode+old_episodes, cr=cum_reward))
            sys.stdout.flush()
            episode += 1
            # Saving model every 1000 episodes
            if episode % 1000 == 0:
                agent.save(directory=save_dir,
                           filename='agent-{ep}'.format(ep=episode+old_episodes),
                           format='hdf5')
                with open(save_dir + '/parameters.txt', 'w+') as f:
                    f.write('image index: %d \n' % image_index)
                    f.write('policy learning rate: %f \n' % policy_lr)
                    f.write('baseline learning rate: %f \n' % baseline_lr)
                    f.write('episode length: %d \n' % steps_per_episode)
