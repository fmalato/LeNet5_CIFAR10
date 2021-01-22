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


if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        # Parameters initialization
        batch_size = 1
        steps_per_episode = 30
        policy_lr = 1e-3
        baseline_lr = 1e-2
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
        #image_index = 1614
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
        num_actions = environment.action_space.n
        environment = Environment.create(environment=environment,
                                         states=dict(
                                             # 64 features + 3 positional coding
                                             features=dict(type=float, shape=(67,)),
                                         ),
                                         actions=dict(type=int, num_values=num_actions),
                                         max_episode_timesteps=steps_per_episode
                                         )
        # Agent initialization
        if load_checkpoint:
            directory = 'models/RL/20210120-201518/'
            old_episodes = (len(os.listdir(directory)) - 1) * 1000
            print('Loading checkpoint. Last episode: %d' % old_episodes)
            agent = Agent.load(directory=directory,
                               filename='agent-19000',
                               format='hdf5',
                               environment=environment,
                               agent='ppo',
                               network=[
                                   # First module: shared dense block
                                   [
                                       dict(type='dense', size=64, activation='relu'),
                                       dict(type='dense', size=64, activation='relu'),
                                       dict(type='lstm', size=64, horizon=steps_per_episode, activation='relu'),
                                       dict(type=TrackedDense, size=10, activation='softmax'),
                                       dict(type='dense', size=64, activation='relu')
                                   ],

                               ],
                               baseline=[
                                   dict(type='dense', size=64, activation='relu'),
                                   dict(type='dense', size=64, activation='relu')
                               ],
                               baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                               summarizer=dict(
                                   directory='data/summaries',
                                   summaries='all'
                               ),
                               learning_rate=policy_lr,
                               batch_size=10,
                               tracking=['tracked_dense'],
                               discount=0.99,
                               states=dict(
                                   # 64 features + 3 positional coding
                                   features=dict(type=float, shape=(67,)),
                               ),
                               actions=dict(type=int, num_values=num_actions),
                               entropy_regularization=0.01
                               )
        else:
            old_episodes = 0
            agent = Agent.create(environment=environment,
                                 agent='ppo',
                                 network=[
                                       # First module: shared dense block
                                       [
                                           dict(type='dense', size=64, activation='relu'),
                                           dict(type='dense', size=64, activation='relu'),
                                           dict(type='lstm', size=64, horizon=steps_per_episode, activation='relu'),
                                           dict(type=TrackedDense, size=10, activation='softmax'),
                                           dict(type='dense', size=64, activation='relu')
                                       ],

                                   ],
                                 baseline=[
                                     dict(type='dense', size=64, activation='relu'),
                                     dict(type='dense', size=64, activation='relu')
                                 ],
                                 baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3),
                                 summarizer=dict(
                                       directory='data/summaries',
                                       summaries='all'
                                   ),
                                 learning_rate=policy_lr,
                                 batch_size=10,
                                 tracking=['tracked_dense'],
                                 discount=0.99,
                                 states=dict(
                                       # 64 features + 3 positional coding
                                       features=dict(type=float, shape=(67,)),
                                   ),
                                 actions=dict(type=int, num_values=num_actions),
                                 entropy_regularization=0.01
                                 )
        first_time = True
        episode = 0
        save_dir = 'models/RL/{x}/'.format(x=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if visualize:
            # Visualization objects
            tile_width = 10
            num_layers = 5
            agent_sprite = AgentSprite(rect_width=tile_width, num_layers=num_layers)
            drawer = Drawer(agent_sprite, num_layers=num_layers, tile_width=tile_width)
        while True:
            # Only one image for now
            """if not first_time:
                # Extraction of a random image for next episode
                image_index = random.randint(0, len(train_images) - 1)
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
                first_time = False"""
            # new image after 500 episodes
            if episode == 500:
                prev_index = image_index
                prev_image = train_image
                prev_label = train_label
                prev_features = net_features
                prev_distrib = net_distribution
                while True:
                    image_index = random.randint(0, len(train_images) - 1)
                    if int(train_labels[image_index]) != train_label:
                        break
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
            # switching image every 500 episodes
            if episode % 500 == 0 and episode > 500:
                tmp_img = prev_image
                tmp_label = prev_label
                tmp_index = prev_index
                environment.environment.features = prev_features
                environment.environment.distribution = prev_distrib
                environment.environment.image_class = prev_label
                prev_index = image_index
                prev_image = train_image
                prev_label = train_label
                image_index = tmp_index
                train_image = tmp_img
                train_label = tmp_label
                prev_features = net_features
                prev_distrib = net_distribution
            state = environment.reset()
            cum_reward = 0.0
            terminal = False
            while not terminal:
                action = agent.act(states=dict(features=state['features']))
                distrib = agent.tracked_tensors()['agent/policy/network/layer0/tracked_dense']
                environment.environment.agent_classification = distrib
                state, terminal, reward = environment.execute(actions=action)
                agent.observe(terminal=terminal, reward=reward)
                cum_reward += reward
                if visualize:
                    print('Correct label: {l} - Predicted label: {p}'.format(l=class_names[train_label],
                                                                             p=class_names[int(np.argmax(distrib))]))
                    drawer.render(agent=agent_sprite)
                    agent_sprite.move(environment.environment.agent_pos)
            sys.stdout.write('\rEpisode {ep} - Cumulative Reward: {cr}'.format(ep=episode+old_episodes, cr=cum_reward))
            sys.stdout.flush()
            episode += 1
            # Saving model every 1000 episodes
            if episode % 1000 == 0:
                agent.save(directory=save_dir,
                           filename='agent-{ep}'.format(ep=episode),
                           format='hdf5')
                with open(save_dir + '/parameters.txt', 'w+') as f:
                    f.write('image index: %d, %d \n' % (image_index, prev_index))
                    f.write('policy learning rate: %f \n' % policy_lr)
                    f.write('baseline learning rate: %f \n' % baseline_lr)
                    f.write('episode length: %d \n' % steps_per_episode)
