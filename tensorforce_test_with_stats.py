import sys
import os
import random
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
        steps_per_episode = 30
        policy_lr = 1e-3
        baseline_lr = 1e-2
        directory = 'models/RL/20210203-134302/'
        last_episode = 26000
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        visualize = False
        # 'stats' directory creation
        if 'stats' not in os.listdir(directory):
            os.mkdir(directory + 'stats')
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20210112-134853.h5')
        #net.summary()
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        # Extracting one image per class
        indexes, labels = one_image_per_class(test_labels, len(class_names))
        train_images = np.array([train_images[idx] for idx in indexes])
        train_labels = np.array(labels)
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
        num_actions = len(environment.actions)
        environment = Environment.create(environment=environment,
                                         states=dict(
                                             # 64 features + 3 positional coding
                                             features=dict(type=float, shape=(67,)),
                                         ),
                                         actions=dict(
                                             movement=dict(type=int, num_values=num_actions),
                                             classification=dict(type=int, num_values=len(class_names))
                                         ),
                                         max_episode_timesteps=steps_per_episode
                                         )
        # Agent initialization
        for i in range(1, int(last_episode / 1000) + 1):
            old_episodes = i*1000
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
                                       dict(type='dense', size=64, activation='relu'),
                                       dict(type='lstm', size=64, horizon=steps_per_episode, activation='relu'),
                                   ],

                               ],
                               baseline=[
                                   dict(type='dense', size=64, activation='relu'),
                                   dict(type='dense', size=64, activation='relu')
                               ],
                               baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                               learning_rate=policy_lr,
                               batch_size=10,
                               tracking=['distribution'],
                               discount=0.99,
                               states=dict(
                                   # 64 features + 3 positional coding
                                   features=dict(type=float, shape=(67,)),
                               ),
                               actions=dict(
                                   movement=dict(type=int, num_values=num_actions),
                                   classification=dict(type=int, num_values=len(class_names))
                               ),
                               entropy_regularization=0.01,
                               # exploration=0.1
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
            with open(directory + 'stats/' + 'stats_agent_{x}.txt'.format(x=old_episodes), 'w+') as file:
                file.write('action; max_prob; class_label; reward; [class_reward, mov_reward]\n')
                for i in range(50):
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
                    file.write('Episode %d\n' % i)
                    state = environment.reset()
                    cum_reward = 0.0
                    terminal = False
                    internals = agent.initial_internals()
                    while not terminal:
                        action, internals = agent.act(states=dict(features=state['features']), internals=internals,
                                                      independent=True, deterministic=False)
                        distrib = agent.tracked_tensors()['agent/policy/classification_distribution/probabilities']
                        environment.environment.agent_classification = distrib
                        state, terminal, reward = environment.execute(actions=action)
                        reward_components = [environment.environment.class_reward,
                                             environment.environment.mov_reward]
                        file.write(
                            '{a};{p};{l};{r};{rl}\n'.format(a=list(action.items()), p=distrib[int(np.argmax(distrib))], l=np.argmax(distrib),
                                                            r=reward, rl=reward_components))
                        cum_reward += reward
                        if visualize:
                            print('Correct label: {l} - Predicted label: {p}'.format(l=class_names[train_label],
                                                                                     p=class_names[int(np.argmax(distrib))]))
                            drawer.render(agent=agent_sprite)
                            agent_sprite.move(environment.environment.agent_pos)
                    sys.stdout.write('\rEpisode {ep} - Cumulative Reward: {cr}'.format(ep=episode+old_episodes, cr=cum_reward))
                    sys.stdout.flush()
                    episode += 1
