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
from utils import split_dataset, n_images_per_class_new, build_heatmap


if __name__ == '__main__':
    with tf.device('/device:GPU:0'):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        # Network hyperparameters
        batch_size = 50
        sampling_ratio = 0.99
        discount = 0.999
        num_classes = 10
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
        layers = [2, 3]
        num_epochs = 1
        partial_dataset = False
        if partial_dataset:
            images_per_class = 10
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
        dirs = ['models/RL/20210426-131450']
        for directory in dirs:
            check_dir = directory + '/checkpoints/'
            print('\nTesting {dir}'.format(dir=directory))
            old_epochs = 15
            agent = Agent.load(directory=check_dir,
                               filename='agent-{oe}'.format(oe=old_epochs-1),
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
                                   features=dict(type=float, shape=(147,)),
                               ),
                               actions=dict(type=int, num_values=num_actions+num_classes)
                               )
            # Parameters for test loop
            episode = 0
            correct = 0
            base_correct = 0
            class_attempt = 0
            not_classified = 0
            rewards = []
            performance = {}
            predicted_labels = []
            true_labels = []
            baseline_labels = []
            class_distrib = []
            agent_positions = []
            only_baseline = []
            class_pos = []
            num_images = len(test_labels)
            mov_histogram = {}
            mov_histogram[0] = np.zeros(steps_per_episode).tolist()
            # Test loop
            for i in range(1, len(test_labels) + 1):
                terminal = False
                ep_reward = 0
                state = environment.reset()
                internals = agent.initial_internals()
                current_step = 0
                while not terminal:
                    action, internals = agent.act(states=dict(features=state['features']), internals=internals,
                                                  independent=True, deterministic=True)
                    if heatmap_needed:
                        agent_positions.append(environment.environment.agent_pos)
                    distrib = agent.tracked_tensors()['agent/policy/action_distribution/probabilities']
                    environment.environment.set_agent_classification(distrib)
                    state, terminal, reward = environment.execute(actions=action)
                    if terminal:
                        mov_histogram[0][current_step] += 1
                        if action == test_labels[i - 1]:
                            correct += 1
                        with tf.device('/device:CPU:0'):
                            pred = np.reshape(RGB_images[i-1], (1, 32, 32, 3))
                            pred = net(pred)
                        if int(action) < 10:
                            # Add a classification attempt
                            predicted_labels.append(int(action))
                            true_labels.append(int(test_labels[i-1]))
                            baseline_labels.append(int(np.argmax(pred)))
                            # Marginalized distribution over classification actions
                            class_distrib.append([x / sum(distrib[:10]) for x in distrib[:10]])
                            class_pos.append(str(environment.environment.agent_pos))
                        else:
                            #class_dis = [x / sum(distrib[:10]) for x in distrib[:10]]
                            if np.argmax(pred) == test_labels[i - 1]:
                                base_correct += 1
                            #class_attempt += 1
                            only_baseline.append((i-1, int(np.argmax(pred))))
                    if int(action) < 10:
                        class_attempt += 1
                    ep_reward += reward
                    current_step += 1
                rewards.append(ep_reward)
                avg_reward = np.sum(rewards) / len(rewards)
                if class_attempt == 0:
                    class_attempt = 1
                sys.stdout.write('\rTest: Episode {ep} - Last ep. reward: {last_ep} - Average reward: {cr} - Correct: {ok}% - RCA Correct: {rcaok}% - w/Baseline: {bok}%'
                                 .format(ep=i,
                                         last_ep=round(ep_reward, 2),
                                         cr=round(avg_reward, 3),
                                         ok=round((correct / i) * 100, 2),
                                         rcaok=round((correct / class_attempt) * 100, 2),
                                         bok=round(((correct+base_correct) / i) * 100, 2)))
                sys.stdout.flush()
            print('\n')
            performance['predicted'] = predicted_labels
            performance['true lab'] = true_labels
            performance['baseline'] = baseline_labels
            performance['class distr'] = class_distrib
            performance['class position'] = class_pos
            with open(directory + '/stats/predicted_labels.json', 'w+') as f:
                json.dump(performance, f)
                f.close()
            if histogram_needed:
                with open(directory + '/stats/movement_histogram_test.json', 'w+') as f:
                    json.dump(mov_histogram, f)
            right_when_baseline_wrong = 0
            wrong_when_baseline_right = 0
            agent_baseline_wrong = 0
            agent_baseline_right = 0
            different_class = 0
            same_class = 0
            for i in range(len(performance['predicted'])):
                if predicted_labels[i] == true_labels[i] and baseline_labels[i] == true_labels[i]:
                    agent_baseline_right += 1
                if predicted_labels[i] == true_labels[i] and baseline_labels[i] != true_labels[i]:
                    right_when_baseline_wrong += 1
                if predicted_labels[i] != true_labels[i] and baseline_labels[i] == true_labels[i]:
                    wrong_when_baseline_right += 1
                if predicted_labels[i] != true_labels[i] and baseline_labels[i] != true_labels[i]:
                    agent_baseline_wrong += 1
                    if predicted_labels[i] != baseline_labels[i]:
                        different_class += 1
                    else:
                        same_class += 1
            right = 0
            base_stats = {}
            with open(directory + '/stats/only_baseline.json', 'w+') as f:
                base_stats['baseline pred'] = only_baseline
                ground_truth = []
                for el in only_baseline:
                    if el[1] == test_labels[el[0]]:
                        right += 1
                    ground_truth.append(int(test_labels[el[0]]))
                base_stats['ground truth'] = ground_truth
                json.dump(base_stats, f)
                f.close()
            print('Number of times that agent improves baseline: %d / %d' % (right_when_baseline_wrong, len(test_labels)))
            print('Number of times that agent fails when baseline does not: %d / %d' % (wrong_when_baseline_right, len(test_labels)))
            print('Number of times that both agent and baseline are right: %d / %d' % (agent_baseline_right, len(test_labels)))
            print('Number of times that both agent and baseline are wrong: %d / %d' % (agent_baseline_wrong, len(test_labels)))
            print('    where agent and baseline predict the same class: %d / %d' % (same_class, len(test_labels)))
            print('    where agent and baseline predict different classes: %d / %d' % (different_class, len(test_labels)))
            print('Number of times that baseline produces correct output when agent does not classify: %d / %d' % (right, len(only_baseline)))
            if heatmap_needed:
                build_heatmap(agent_positions, dir=directory, show=False)
