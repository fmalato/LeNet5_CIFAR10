import copy
import sys
import json

import numpy as np
import tensorflow as tf
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorflow.keras import datasets

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicConvnetGymEnv

from utils import n_images_per_class_new


if __name__ == "__main__":
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    layers = [0, 1, 2, 3]
    image_index = 2
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
    baseline_lr = 1e-4
    e_r = 0.1
    split_ratio = 0.8
    # Reward parameters
    class_penalty = 3.0
    correct_class = 2.0
    illegal_mov = 0.25
    same_position = 0.05
    non_classified = 3.0
    step_reward_multiplier = 0.01
    # Control parameters
    visualize = False
    ########################### PREPROCESSING ##############################
    # Network initialization
    with tf.device('/device:CPU:0'):
        net = DyadicConvNet(num_channels=64, input_shape=(1, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20210421-123951.h5')
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        test_images = np.array(test_images, dtype=np.float32)
        image_idxs = n_images_per_class_new(n=1, labels=test_labels, num_classes=10)
        all_data = {}
        del train_images, train_labels
        for image_index in image_idxs:
            print('\nTesting image {idx}'.format(idx=image_index))
            test_image = test_images[image_index]
            test_image = test_image / 255.0
            test_labels = [test_labels[x] for x in image_idxs]
            # Initializing everything that the env requires to work properly
            RGB_image = [copy.deepcopy(test_image)]
            tmp = []
            # We extract EVERY single representation to avoid doing it at every episode (MEMORY INTENSIVE)
            image = np.reshape(test_image, (1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
            tmp.append(net.extract_features(image, active_layers=layers, last_layer=4))
            test_image = copy.deepcopy(tmp)
            del tmp
            #########################################################################
            # Environment initialization
            environment = DyadicConvnetGymEnv(dataset=test_image,
                                              labels=test_labels[image_index],
                                              images=RGB_image,
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
                                             actions=dict(type=int, num_values=num_actions + num_classes),
                                             max_episode_timesteps=steps_per_episode
                                             )
            # Agent initialization
            dirs = ['models/RL/20210515-120754']
            for directory in dirs:
                check_dir = directory + '/checkpoints/'
                print('\nTesting {dir}'.format(dir=directory))
                old_epochs = 10
                agent = Agent.load(directory=check_dir,
                                   filename='agent-{oe}'.format(oe=old_epochs - 1),
                                   format='hdf5',
                                   environment=environment,
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
                data = {}
                data["ground truth"] = int(test_labels[image_index])
                for layer in layers:
                    for x in range(pow(2, len(layers) - layer)):
                        for y in range(pow(2, len(layers) - layer)):
                            current_key = str((layer, x, y))
                            sys.stdout.write('\rTesting position {pos}'.format(pos=current_key))
                            data[current_key] = {}
                            state = environment.reset()
                            environment.environment.agent_pos = (layer, x, y)
                            state = environment.environment.gen_obs()
                            internals = agent.initial_internals()
                            terminal = False
                            steps = 0
                            ep_distribution = []
                            ep_positions = []
                            ep_actions = []
                            while not terminal:
                                action, internals = agent.act(states=dict(features=state['features']), internals=internals,
                                                              independent=True, deterministic=True)
                                # Marginalizing distribution
                                distrib = agent.tracked_tensors()['agent/policy/action_distribution/probabilities']
                                distrib = [x / sum(distrib[:num_classes]) for x in distrib[:num_classes]]
                                ep_distribution.append(distrib)
                                ep_positions.append(str(environment.environment.agent_pos))
                                ep_actions.append(int(action))
                                environment.environment.set_agent_classification(distrib)
                                state, terminal, reward = environment.execute(actions=action)
                                steps += 1
                                if terminal:
                                    data[current_key]["number of steps"] = steps
                                    data[current_key]["prediction"] = int(action) if int(action) < num_classes else -1
                                    data[current_key]["distribution"] = ep_distribution
                                    data[current_key]["positions"] = ep_positions
                                    data[current_key]["actions"] = ep_actions
                all_data[int(image_index)] = data
            with open(directory + '/stats/each_position.json', 'w+') as f:
                json.dump(all_data, f)
                f.close()
