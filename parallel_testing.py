import copy
import sys
import json
from threading import Thread
from statistics import mode

import numpy as np
import tensorflow as tf
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorflow.keras import datasets

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicConvnetGymEnv
from utils import split_dataset, n_images_per_class_new, build_heatmap


class Consensus:
    def __init__(self):
        self.votes = np.zeros((10,))
        self.positions = []

    def reset(self):
        self.num_votes = 0
        self.votes = np.zeros((10,))
        self.positions = []

    def update(self, vote, pos):
        self.num_votes += 1
        self.votes += vote
        self.votes = self.votes / self.num_votes
        self.positions.append(pos)


def run_episode(agent, environment, consensus, worker_id, num_episodes):
    print("Worker {id} starting test".format(id=worker_id))
    for i in range(1, num_episodes + 1):
        state = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            action, internals = agent.act(states=dict(features=state['features']), internals=internals,
                                          independent=True, deterministic=True)
            distrib = agent.tracked_tensors()['agent/policy/action_distribution/probabilities']
            environment.environment.set_agent_classification(distrib)
            state, terminal, reward = environment.execute(actions=action)
            if terminal:
                consensus.update(action)
    print("Worker {id} finishing test".format(id=worker_id))


if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        # Network hyperparameters
        batch_size = 50
        sampling_ratio = 0.99
        discount = 0.999
        num_classes = 10
        num_features = 147
        lstm_units = 128
        lstm_horizon = 5
        steps_per_episode = 15
        policy_lr = 1e-5
        baseline_lr = 1e-3
        e_r = 0.2
        split_ratio = 0.8
        # Reward parameters
        class_penalty = 1.0
        correct_class = 2.0
        illegal_mov = 0.25
        same_position = 0.05
        non_classified = 3.0
        step_reward_multiplier = 0.01
        # Control parameters
        visualize = False
        # Test parameters
        layers = [0, 1, 2, 3]
        num_epochs = 1
        num_agents = [25]
        partial_dataset = False
        if partial_dataset:
            images_per_class = 1
        else:
            images_per_class = 1000
        heatmap_needed = True
        histogram_needed = True
        ########################### PREPROCESSING ##############################
        # Network initialization
        with tf.device('/device:CPU:0'):
            net = DyadicConvNet(num_channels=64, input_shape=(1, 32, 32, 3))
            net.load_weights('models/model_CIFAR10/20210421-123951.h5')
            # Dataset initialization
            (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
            test_images = np.array(test_images, dtype=np.float32)
            if partial_dataset:
                img_idxs = n_images_per_class_new(n=images_per_class, labels=test_labels, num_classes=len(class_names))
                test_images = np.array([test_images[idx] for idx in img_idxs])
                test_labels = np.array([test_labels[idx] for idx in img_idxs])
            test_images = test_images / 255.0
            # Initializing everything that the env requires to work properly
            RGB_images = copy.deepcopy(test_images)
            tmp = []
            # We extract EVERY single representation to avoid doing it at every episode (MEMORY INTENSIVE)
            idx = 1
            for img in test_images:
                sys.stdout.write(
                    '\rComputing image {current}/{num_img}'.format(current=idx, num_img=test_images.shape[0]))
                image = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
                tmp.append(net.extract_features(image, active_layers=layers, last_layer=4))
                idx += 1
            test_images = copy.deepcopy(tmp)
            del train_images, train_labels
            del tmp
            num_episodes = len(test_labels)
        #########################################################################
        directory = 'models/RL/20210421-130304'
        check_dir = directory + '/checkpoints/'
        print('\nTesting {dir}'.format(dir=directory))
        old_epochs = 15
        #threads = []
        for size in num_agents:
            environments = []
            jedi_council = []
            consensus = Consensus()
            print('Testing {s} agents'.format(s=size))
            for i in range(size):
                environment = DyadicConvnetGymEnv(dataset=test_images,
                                                  labels=test_labels,
                                                  images=RGB_images,
                                                  layers=layers,
                                                  num_classes=10,
                                                  num_features=num_features,
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
                                                     features=dict(type=float, shape=(num_features,)),
                                                 ),
                                                 actions=dict(type=int, num_values=num_actions+num_classes),
                                                 max_episode_timesteps=steps_per_episode
                                                 )
                environments.append(environment)
                agent = Agent.load(directory=check_dir,
                                   filename='agent-{oe}'.format(oe=old_epochs - 1),
                                   format='hdf5',
                                   environment=environments[i],
                                   agent='ppo',
                                   network=[
                                       dict(type='lstm', size=lstm_units, horizon=lstm_horizon, activation='relu'),
                                   ],
                                   baseline=[
                                       dict(type='lstm', size=lstm_units, horizon=lstm_horizon, activation='relu')
                                   ],
                                   baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                                   learning_rate=policy_lr,
                                   batch_size=batch_size,
                                   tracking=['distribution'],
                                   discount=discount,
                                   states=dict(
                                       features=dict(type=float, shape=(num_features,)),
                                   ),
                                   actions=dict(type=int, num_values=num_actions + num_classes)
                                   )
                jedi_council.append(agent)
                #threads.append(Thread(target=run_episode, args=(agent, environment, consensus, i, len(test_labels))))

            correct = 0
            """for thread in threads:
                thread.start()"""
            for i in range(1, num_episodes + 1):
                consensus.reset()
                for j in range(size):
                    state = environments[j].reset()
                    internals = jedi_council[j].initial_internals()
                    terminal = False
                    while not terminal:
                        action, internals = jedi_council[j].act(states=dict(features=state['features']), internals=internals,
                                                                independent=True, deterministic=True)
                        distrib = jedi_council[j].tracked_tensors()['agent/policy/action_distribution/probabilities']
                        environments[j].environment.set_agent_classification(distrib)
                        state, terminal, reward = environments[j].execute(actions=action)
                        if terminal:
                            consensus.update([x / sum(distrib[:10]) for x in distrib[:10]], environments[j].environment.agent_pos)
                if np.argmax(consensus.votes) == test_labels[i-1]:
                    correct += 1
                sys.stdout.write('\rTest: Episode {ep} - Accuracy: {acc}% - Decision: {v} - True: {l}'.format(ep=i,
                                                                                                              acc=round((correct / i) * 100, 2),
                                                                                                              v=np.argmax(consensus.votes),
                                                                                                              l=test_labels[i-1]
                                                                                                              ))
            print('\n')
