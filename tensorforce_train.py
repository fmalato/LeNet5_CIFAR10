import random
import sys
import datetime
import operator
import numpy as np
import tensorflow as tf
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorflow.keras import datasets
from tensorflow import TensorSpec

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicConvnetGymEnv
from tracked_dense import TrackedDense
from grid_drawer import AgentSprite, Drawer


if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        # Parameters initialization
        batch_size = 1
        steps_per_episode = 30
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        visualize = False
        load_checkpoint = False
        train = True
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20210112-134853.h5')
        #net.summary()
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        # Extraction of a random image
        image_index = random.randint(0, len(train_images) - 1)
        train_image = train_images[image_index, :, :, :]
        train_label = train_labels[image_index]
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
        num_actions = environment.action_space.n
        environment = Environment.create(environment=environment,
                                         states=dict(
                                             features=dict(type=float, shape=(67,)),
                                             distribution=dict(type=float, shape=(10,))
                                         ),
                                         actions=dict(type=int, num_values=num_actions),
                                         max_episode_timesteps=steps_per_episode
                                         )
        # Agent initialization
        if load_checkpoint:
            old_episodes = 2000
            print('Loading checkpoint. Last episode: %d' % old_episodes)
            agent = Agent.load(directory='models/RL/20210113-143724',
                               filename='agent-12.data-00000-of-00001',
                               format='checkpoint',
                               environment=environment,
                               agent='ppo',
                               network=[
                                   # First module: shared dense block
                                   [
                                       dict(type='dense', size=64, activation='relu'),
                                       dict(type='lstm', size=64, horizon=steps_per_episode, activation='relu'),
                                       dict(type='register', tensor='lstm_output')
                                   ],
                                   # Second module: from lstm output to categorical distribution
                                   [
                                       dict(type='retrieve', tensors=['lstm_output']),
                                       dict(type='dense', size=64, activation='relu'),
                                       dict(type=TrackedDense, size=10, activation='softmax')
                                   ],
                                   # Third module: from lstm output to action
                                   [
                                       dict(type='retrieve', tensors=['lstm_output']),
                                       dict(type='dense', size=64, activation='relu')
                                   ]
                               ],
                               summarizer=dict(
                                   directory='data/summaries',
                                   summaries='all'
                               ),
                               learning_rate=1e-3,
                               batch_size=10,
                               tracking=['tracked_dense'],
                               discount=0.99,
                               states=dict(
                                   # 64 features + 10 distribution + 3 positional coding
                                   features=dict(type=float, shape=(77,)),
                               ),
                               actions=dict(
                                   action=dict(type=int, num_values=num_actions),
                                   distribution=dict(type=int, num_values=len(class_names))
                               ),
                               entropy_regularization=0.01
                               )
        else:
            old_episodes = 0
            agent = Agent.create(agent='ppo',
                                 environment=environment,
                                 network=[
                                     # First module: shared dense block
                                     [
                                         dict(type='dense', size=64, activation='relu'),
                                         dict(type='lstm', size=64, horizon=steps_per_episode, activation='relu'),
                                         dict(type='register', tensor='lstm_output')
                                     ],
                                     # Second module: from lstm output to categorical distribution
                                     [
                                         dict(type='retrieve', tensors=['lstm_output']),
                                         dict(type='dense', size=64, activation='relu'),
                                         dict(type=TrackedDense, size=10, activation='softmax')
                                     ],
                                     # Third module: from lstm output to action
                                     [
                                         dict(type='retrieve', tensors=['lstm_output']),
                                         dict(type='dense', size=64, activation='relu')
                                     ]
                                 ],
                                 summarizer=dict(
                                     directory='data/summaries',
                                     summaries='all'
                                 ),
                                 learning_rate=1e-3,
                                 batch_size=10,
                                 tracking=['tracked_dense'],
                                 discount=0.99,
                                 states=dict(
                                     # 64 features + 10 distribution + 3 positional coding
                                     features=dict(type=float, shape=(77,)),
                                 ),
                                 actions=dict(
                                                action=dict(type=int, num_values=num_actions),
                                                distribution=dict(type=int, num_values=len(class_names))
                                              ),
                                 entropy_regularization=0.01
                                 )
        first_time = True
        episode = 0
        if visualize:
            # Visualization objects
            tile_width = 10
            num_layers = 5
            agent_sprite = AgentSprite(rect_width=tile_width, num_layers=num_layers)
            drawer = Drawer(agent_sprite, num_layers=num_layers, tile_width=tile_width)
        while True:
            if not first_time:
                # Extraction of a random image for next episode
                image_index = random.randint(0, len(train_images) - 1)
                train_image = train_images[image_index, :, :, :]
                train_label = train_labels[image_index]
                train_image_4dim = np.reshape(train_image, (batch_size, 32, 32, 3))
                # Convolutional features extraction
                net_features = net.extract_features(train_image_4dim)
                net_distribution = np.reshape(net(train_image_4dim).numpy(), (10,))
                # Environment reset with new features and distribution
                environment.environment.features = net_features
                environment.environment.distribution = net_distribution
                environment.environment.image_label = train_label
            else:
                first_time = False
            state = environment.reset()
            cum_reward = 0.0
            terminal = False
            while not terminal:
                action = agent.act(states=dict(features=state['features']), independent=operator.not_(train))
                distrib = agent.tracked_tensors()['agent/policy/network/layer0/tracked_dense']
                environment.environment.agent_classification = distrib
                state, terminal, reward = environment.execute(actions=action)
                if train:
                    agent.observe(terminal=terminal, reward=reward)
                cum_reward += reward
                if visualize:
                    drawer.render(agent=agent_sprite)
                    agent_sprite.move(environment.environment.agent_pos)
            sys.stdout.write('\rEpisode {ep} - Cumulative Reward: {cr}'.format(ep=episode+old_episodes, cr=cum_reward))
            sys.stdout.flush()
            episode += 1
            # Saving model every 1000 episodes
            if episode % 1000 == 0:
                agent.save(directory='models/RL/{x}/'.format(x=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
                           filename='agent',
                           format='checkpoint')
