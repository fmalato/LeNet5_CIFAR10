import random
import sys
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import Environment
from tensorflow.keras import datasets

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicImageEnvironment, DyadicConvnetGymEnv
from tracked_dense import TrackedDense
from grid_drawer import AgentSprite, Drawer


if __name__ == '__main__':
    with tf.device('/device:GPU:0'):
        # Parameters initialization
        batch_size = 1
        steps_per_episode = 200
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        visualize = False
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(batch_size, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20201212-125436.h5')
        #net.summary()
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
                                          max_steps=steps_per_episode
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
        agent = Agent.create(agent='tensorforce',
                             environment=environment,
                             policy=[
                                 # First module: from observation to distribution
                                 [
                                     dict(type='flatten'),
                                     dict(type='dense', size=256),
                                     dict(type='dense', size=256),
                                     dict(type=TrackedDense, size=10),
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
                             tracking=['tracked_dense'],
                             reward_estimation=dict(horizon=50),
                             states=dict(
                                 features=dict(type=float, shape=(64,)),
                                 distribution=dict(type=float, shape=(10,))
                             ),
                             actions=dict(type=int, num_values=num_actions),
                             exploration=dict(type='linear', unit='timesteps',
                                              num_steps=5000 * steps_per_episode,
                                              initial_value=0.99, final_value=0.2),
                             )
        first_time = True
        episode = 0
        if visualize:
            agent_sprite = AgentSprite(rect_width=10, num_layers=5)
            drawer = Drawer(agent_sprite, num_layers=5, tile_width=10)
            num_layers = 5
            max_tiles = pow(2, num_layers - 1)
        while True:
            if not first_time:
                # Extraction of a random image for next episode
                train_image = train_images[random.randint(0, len(train_images) - 1), :, :, :]
                train_image_4dim = np.reshape(train_image, (batch_size, 32, 32, 3))
                # Convolutional features extraction
                net_features = net.extract_features(train_image_4dim)
                net_distribution = np.reshape(net(train_image_4dim).numpy(), (10,))
                # Environment reset with new features and distribution
                environment.environment.features = net_features
                environment.environment.distribution = net_distribution
            else:
                first_time = False
            state = environment.reset()
            cum_reward = 0.0
            terminal = False
            for step in range(steps_per_episode):
                action = agent.act(states=state)
                # TODO: is there a better solution to extract the distribution?
                distrib = agent.tracked_tensors()['agent/policy/network/layer0/tracked_dense']
                environment.environment.agent_distribution = distrib
                state, terminal, reward = environment.execute(actions=action)
                agent.observe(terminal=terminal, reward=reward)
                cum_reward += reward
                if visualize:
                    drawer.render(agent=agent_sprite)
                    agent_sprite.move(environment.environment.agent_pos)
            sys.stdout.write('\rEpisode {ep} - Cumulative Reward: {cr}'.format(ep=episode, cr=cum_reward))
            sys.stdout.flush()
            episode += 1
            # Saving model every 1000 episodes
            if episode % 1000 == 0:
                agent.save(directory='models/RL/{x}/'.format(x=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
                           filename='agent',
                           format='checkpoint')
