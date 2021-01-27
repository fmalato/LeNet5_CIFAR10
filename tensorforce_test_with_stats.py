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
        directory = 'models/RL/20210120-201518/'
        old_episodes = 19000
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        visualize = False
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20210112-134853.h5')
        #net.summary()
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        # Extraction of a random image
        #image_index = random.randint(0, len(train_images) - 1)
        image_index = 1614
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
        print('Loading checkpoint. Last episode: %d' % old_episodes)
        agent = Agent.load(directory=directory,
                           filename='agent-{x}'.format(x=old_episodes),
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
        if agent is None:
            print("Couldn't load agent.")
        first_time = True
        episode = 0
        if visualize:
            # Visualization objects
            tile_width = 10
            num_layers = 5
            agent_sprite = AgentSprite(rect_width=tile_width, num_layers=num_layers)
            drawer = Drawer(agent_sprite, num_layers=num_layers, tile_width=tile_width)
        with open(directory + 'stats_agent_{x}.txt'.format(x=old_episodes), 'w+') as file:
            file.write('action, max_prob, class_label, reward\n')
            for i in range(50):
                file.write('Episode %d\n' % i)
                state = environment.reset()
                cum_reward = 0.0
                terminal = False
                internals = agent.initial_internals()
                while not terminal:
                    action, internals = agent.act(states=dict(features=state['features']), internals=internals,
                                                  independent=True, deterministic=False)
                    distrib = agent.tracked_tensors()['agent/policy/network/layer0/tracked_dense']
                    environment.environment.agent_classification = distrib
                    state, terminal, reward = environment.execute(actions=action)
                    file.write(
                        '{a},{p},{l},{r}\n'.format(a=action, p=distrib[int(np.argmax(distrib))], l=np.argmax(distrib),
                                                   r=reward))
                    cum_reward += reward
                    if visualize:
                        print('Correct label: {l} - Predicted label: {p}'.format(l=class_names[train_label],
                                                                                 p=class_names[int(np.argmax(distrib))]))
                        drawer.render(agent=agent_sprite)
                        agent_sprite.move(environment.environment.agent_pos)
                sys.stdout.write('\rEpisode {ep} - Cumulative Reward: {cr}'.format(ep=episode+old_episodes, cr=cum_reward))
                sys.stdout.flush()
                episode += 1
