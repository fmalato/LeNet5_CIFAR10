import copy
import sys


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
        batch_size = 50
        sampling_ratio = 0.5
        discount = 0.999
        num_classes = 10
        lstm_horizon = 5
        steps_per_episode = 15
        policy_lr = 1e-3
        baseline_lr = 1e-2
        e_r = 0.15
        split_ratio = 0.8
        # Reward parameters
        class_penalty = 0.15
        correct_class = 2.0
        illegal_mov = 0.25
        same_position = 0.1
        timestep_passed = 0.01
        fast_class_bonus = 0.02
        # Control parameters
        visualize = False
        load_checkpoint = False
        # Train/test parameters
        num_epochs = 1
        images_per_class = 50
        ########################### PREPROCESSING ##############################
        # Network initialization
        with tf.device('/device:CPU:0'):
            net = DyadicConvNet(num_channels=64, input_shape=(1, 32, 32, 3))
            net.load_weights('models/model_CIFAR10/20210204-122725.h5')
            # Dataset initialization
            (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
            img_idxs, labels = n_images_per_class(n=images_per_class, labels=test_labels, num_classes=len(class_names))
            test_images = np.array([test_images[idx] for idx in img_idxs])
            test_labels = np.array(labels)
            test_images = test_images / 255.0
            # Initializing everything that the env requires to work properly
            RGB_images = copy.deepcopy(test_images)
            tmp = []
            distributions = []
            # We extract EVERY single representation to avoid doing it at every episode (MEMORY INTENSIVE)
            for img in test_images:
                image = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
                tmp.append(net.extract_features(image))
                # Distribution are computed in the exact same order as training images
                distributions.append(np.reshape(net(image).numpy(), (10,)))
            test_images = copy.deepcopy(tmp)
            del train_images, train_labels
            del net, tmp
        #########################################################################
        # Environment initialization
        environment = DyadicConvnetGymEnv(dataset=test_images,
                                          labels=test_labels,
                                          images=RGB_images,
                                          distributions=distributions,
                                          max_steps=steps_per_episode,
                                          visualize=visualize,
                                          testing=True,
                                          num_layers=4,
                                          class_penalty=class_penalty,
                                          correct_class=correct_class,
                                          illegal_mov=illegal_mov,
                                          same_position=same_position,
                                          timestep_passed=timestep_passed,
                                          fast_class_bonus=fast_class_bonus
                                          )
        num_actions = len(environment.actions)
        environment = Environment.create(environment=environment,
                                         states=dict(
                                             features=dict(type=float, shape=(147,)),
                                         ),
                                         actions=dict(type=int, num_values=num_actions+num_classes),
                                         max_episode_timesteps=steps_per_episode
                                         )
        dirs = ['models/RL/20210315-092741']
        for directory in dirs:
            print('Testing {dir}'.format(dir=directory))
            old_episodes = 200000
            print('Loading checkpoint. Last episode: %d' % old_episodes)
            agent = Agent.load(directory=directory,
                               filename='agent',
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
                                   features=dict(type=float, shape=(147,)),
                               ),
                               actions=dict(type=int, num_values=num_actions+num_classes),
                               entropy_regularization=e_r
                               )
            # Parameters for test loop
            episode = 0
            correct = 0
            rewards = []
            num_images = len(test_labels)
            # Test loop
            for i in range(1, len(test_labels) + 1):
                terminal = False
                ep_reward = 0
                state = environment.reset()
                internals = agent.initial_internals()
                while not terminal:
                    action, internals = agent.act(states=dict(features=state['features']), internals=internals,
                                                  independent=True, deterministic=True)
                    environment.environment.set_agent_classification(agent.tracked_tensors()['agent/policy/action_distribution/probabilities'])
                    state, terminal, reward = environment.execute(actions=action)
                    if terminal:
                        if action == test_labels[i - 1]:
                            correct += 1
                    ep_reward += reward
                rewards.append(ep_reward)
                avg_reward = np.sum(rewards) / len(rewards)
                sys.stdout.write('\rTest: Episode {ep} - Average reward: {cr} - Correct: {ok}%'.format(ep=i,
                                                                                                       cr=round(avg_reward, 3),
                                                                                                       ok=round((correct / i) * 100, 2)))
                sys.stdout.flush()
            print('\n')
