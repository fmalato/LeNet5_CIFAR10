import copy
import random
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
    with tf.device('/device:GPU:0'):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        # Network hyperparameters
        num_classes = 10
        steps_per_episode = 15
        # Reward parameters
        class_penalty = 3.0
        correct_class = 2.0
        illegal_mov = 1.0
        same_position = 0.05
        non_classified = 3.0
        step_reward_multiplier = 0.01
        # Control parameters
        visualize = False
        # Train/test parameters
        images_per_class = 10
        layers = [0, 1, 2, 3]
        ########################### PREPROCESSING ##############################
        # Network initialization
        with tf.device('/device:CPU:0'):
            net = DyadicConvNet(num_channels=64, input_shape=(1, 32, 32, 3))
            net.load_weights('models/model_CIFAR10/20210303-125114.h5')
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
                tmp.append(net.extract_features(image, active_layers=layers))
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
                                          layers=layers,
                                          max_steps=steps_per_episode,
                                          visualize=visualize,
                                          training=False,
                                          class_penalty=class_penalty,
                                          correct_class=correct_class,
                                          illegal_mov=illegal_mov,
                                          same_position=same_position,
                                          non_classified=non_classified,
                                          step_reward_multiplier=step_reward_multiplier
                                          )
        num_actions = len(environment.actions)
        environment = Environment.create(environment=environment,
                                         states=dict(
                                             features=dict(type=float, shape=(147,)),
                                         ),
                                         actions=dict(type=int, num_values=num_actions+num_classes),
                                         max_episode_timesteps=steps_per_episode
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
            step_count = 0
            print("Env reset. Starting position: {pos}".format(pos=environment.environment.agent_pos))
            while not terminal:
                action = int(input("(Correct class: {cc} - Step {s}) Select action: ".format(cc=test_labels[i - 1],
                                                                                             s=step_count)))
                environment.environment.set_agent_classification([0.07 * step_count if x == test_labels[i - 1] else (1.0 - 0.07 * step_count) / 9 for x in range(10)])
                state, terminal, reward = environment.execute(actions=action)
                print([0.07 * step_count if x == test_labels[i - 1] else (1.0 - 0.07 * step_count) / 9 for x in range(10)])
                step_count += 1
                print("Step reward: {rew} - Current position: {pos}".format(rew=round(reward, 3),
                                                                            pos=environment.environment.agent_pos))
                if terminal:
                    if action == test_labels[i - 1]:
                        correct += 1
                ep_reward += reward
            sys.stdout.write('\rEpisode reward: {ep_rew}\n'.format(ep_rew=ep_reward))
            sys.stdout.flush()
        print('\n')
