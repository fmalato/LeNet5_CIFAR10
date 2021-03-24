import os
import sys
import datetime
import xlwt
import copy
import json

import numpy as np
import tensorflow as tf
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorflow.keras import datasets

from tensorforce_net import DyadicConvNet
from tensorforce_env import DyadicConvnetGymEnv
from utils import split_dataset, n_images_per_class, shuffle_data


if __name__ == '__main__':
    with tf.device('/device:GPU:0'):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        # Network hyperparameters
        batch_size = 50
        sampling_ratio = 0.99
        discount = 0.999
        num_classes = 10
        lstm_horizon = 5
        steps_per_episode = 15
        policy_lr = 1e-3
        baseline_lr = 1e-2
        e_r = 0.2
        split_ratio = 0.8
        # Reward parameters
        class_penalty = 0.15
        correct_class = 2.0
        illegal_mov = 0.25
        same_position = 0.05
        # Control parameters
        visualize = False
        load_checkpoint = False
        # Train/test parameters
        num_epochs = 100
        images_per_class = 50
        parameters = [batch_size, sampling_ratio, discount, lstm_horizon, steps_per_episode, policy_lr,
                      baseline_lr, e_r, split_ratio, class_penalty, correct_class, illegal_mov, same_position,
                      images_per_class]
        parameters_names = ['Batch Size', 'Sampling Ratio', 'Discount Factor', 'LSTM horizon', 'Steps per Episode',
                            'Policy lr', 'Baseline lr', 'Entropy Reg', 'Dataset Split Ratio', 'Class Penalty',
                            'Correct Classification', 'Illegal Move', 'Same Position', 'Images per Class']
        ########################### PREPROCESSING ##############################
        # Network initialization
        with tf.device('/device:CPU:0'):
            net = DyadicConvNet(num_channels=64, input_shape=(1, 32, 32, 3))
            net.load_weights('models/model_CIFAR10/20210303-125114.h5')
            print('Computing whole dataset features...')
            # Dataset initialization - we don't need test data here
            (train_images, train_labels), (_, _) = datasets.cifar10.load_data()
            img_idxs, labels = n_images_per_class(n=images_per_class, labels=train_labels, num_classes=len(class_names))
            train_images = np.array([train_images[idx] for idx in img_idxs])
            train_labels = np.array(labels)
            train_images = train_images / 255.0
            # Initializing everything that the env requires to work properly
            RGB_images = copy.deepcopy(train_images)
            tmp = []
            distributions = []
            # We extract EVERY single representation to avoid doing it at every episode (MEMORY INTENSIVE)
            for img in train_images:
                img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
                tmp.append(net.extract_features(img))
                # Distribution are computed in the exact same order as training images
                distributions.append(np.reshape(net(img).numpy(), (10,)))
            # Split training and validation set
            train_images, valid_images, train_labels, valid_labels = split_dataset(dataset=tmp, labels=train_labels, ratio=split_ratio)
            train_distrib, valid_distrib, _, _ = split_dataset(dataset=distributions, labels=range(len(distributions)), ratio=split_ratio)
            # Also getting RGB images for visualization TODO: make it 'visualize'-wise without breaking the environment
            train_RGB_imgs = RGB_images[:int(RGB_images.shape[0] * split_ratio)]
            valid_RGB_imgs = RGB_images[int(RGB_images.shape[0] * split_ratio):]
            # We don't need them anymore - Bye bye CNN!
            del tmp, distributions
            del net
            print('Done.\n')
            #########################################################################
        # Shuffling everything there is to shuffle
        train_images, train_labels, train_distrib, train_RGB_imgs = shuffle_data(dataset=train_images, labels=train_labels,
                                                                                 distributions=train_distrib, RGB_imgs=train_RGB_imgs)
        valid_images, valid_labels, valid_distrib, valid_RGB_imgs = shuffle_data(dataset=valid_images, labels=valid_labels,
                                                                                 distributions=valid_distrib, RGB_imgs=valid_RGB_imgs)
        num_episodes = len(train_labels)
        num_images = len(train_labels)
        len_valid = len(valid_labels)
        # Training environment initialization
        environment = DyadicConvnetGymEnv(dataset=train_images,
                                          labels=train_labels,
                                          images=train_RGB_imgs,
                                          distributions=train_distrib,
                                          max_steps=steps_per_episode,
                                          visualize=visualize,
                                          testing=False,
                                          num_layers=4,
                                          class_penalty=class_penalty,
                                          correct_class=correct_class,
                                          illegal_mov=illegal_mov,
                                          same_position=same_position
                                          )
        num_actions = len(environment.actions)
        environment = Environment.create(environment=environment,
                                         states=dict(
                                             features=dict(type=float, shape=(147,)),
                                         ),
                                         actions=dict(type=int, num_values=num_actions+num_classes),
                                         max_episode_timesteps=steps_per_episode
                                         )
        # Validation environment initialization
        valid_environment = DyadicConvnetGymEnv(dataset=valid_images,
                                                labels=valid_labels,
                                                images=valid_RGB_imgs,
                                                distributions=valid_distrib,
                                                max_steps=steps_per_episode,
                                                visualize=visualize,
                                                testing=False,
                                                num_layers=4,
                                                class_penalty=class_penalty,
                                                correct_class=correct_class,
                                                illegal_mov=illegal_mov,
                                                same_position=same_position
                                                )
        num_actions = len(valid_environment.actions)
        valid_environment = Environment.create(environment=valid_environment,
                                               states=dict(
                                                   features=dict(type=float, shape=(147,)),
                                               ),
                                               actions=dict(type=int, num_values=num_actions+num_classes),
                                               max_episode_timesteps=steps_per_episode
                                               )
        # Agent initialization
        if load_checkpoint:
            directory = 'models/RL/20210316-115334'
            old_epochs = 52
            print('Loading checkpoint. Number of old epochs: %d' % old_epochs)
            agent = Agent.load(directory=directory,
                               filename='agent{oe}'.format(oe=old_epochs),
                               format='hdf5',
                               environment=environment,
                               agent='ppo',
                               max_episode_timesteps=steps_per_episode,
                               network=[
                                       dict(type='lstm', size=64, horizon=lstm_horizon, activation='relu'),
                               ],
                               baseline=[
                                   dict(type='lstm', size=64, horizon=lstm_horizon, activation='relu')
                               ],
                               baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                               # TODO: Huge file - find minimum number of parameters
                               summarizer=dict(
                                   directory='data/summaries',
                                   summaries=['action-value', 'entropy', 'reward', 'distribution']
                               ),
                               learning_rate=policy_lr,
                               batch_size=batch_size,
                               tracking=['distribution'],
                               discount=discount,
                               states=dict(
                                   features=dict(type=float, shape=(147,)),
                               ),
                               actions=dict(type=int, num_values=num_actions+num_classes),
                               entropy_regularization=0.01
                               )
        else:
            old_epochs = 0
            agent = Agent.create(environment=environment,
                                 agent='ppo',
                                 max_episode_timesteps=steps_per_episode,
                                 network=[
                                     dict(type='lstm', size=128, horizon=lstm_horizon, activation='relu'),
                                 ],
                                 baseline=[
                                     dict(type='lstm', size=128, horizon=lstm_horizon, activation='relu')
                                 ],
                                 baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                                 # TODO: Huge file - find minimum number of parameters
                                 summarizer=dict(
                                     directory='data/summaries',
                                     summaries=['action-value', 'entropy', 'reward', 'distribution']
                                 ),
                                 learning_rate=policy_lr,
                                 batch_size=batch_size,
                                 tracking=['distribution'],
                                 discount=discount,
                                 states=dict(
                                     features=dict(type=float, shape=(147,)),
                                 ),
                                 actions=dict(type=int, num_values=num_actions+num_classes),
                                 entropy_regularization=dict(
                                        type='linear', unit='episodes', num_steps=num_images*num_epochs/2,
                                        initial_value=e_r, final_value=0.01
                                        )
                                 )
        # Parameters for training loop
        epoch_correct = 0
        current_ep = 0
        # Where to store checkpoints
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if load_checkpoint:
            save_dir = directory + "/"
        else:
            save_dir = 'models/RL/{x}/'.format(x=current_time)
            checkpoints_dir = save_dir + 'checkpoints/'
            stats_dir = save_dir + 'stats/'
        if not load_checkpoint and (save_dir not in os.listdir('models/RL/')):
            os.mkdir(save_dir)
            os.mkdir(checkpoints_dir)
            os.mkdir(stats_dir)
        # Initialization of excel sheet with relevant stats
        xl_sheet = xlwt.Workbook()
        sheet = xl_sheet.add_sheet(current_time)
        title_style = xlwt.easyxf('font: bold on; align: horiz center; pattern: pattern solid, fore_colour orange; borders: left thin, right thin, top thin, bottom thin;')
        data_style = xlwt.easyxf('align: horiz center; borders: left thin, right thin, top thin, bottom thin;')
        # Making first column for the sake of readability
        stat_names = ['Epoch', 'Average Reward', 'Epoch Accuracy', 'RCA Accuracy', 'Valid Accuracy', 'Avg.Class', 'Avg.Move']
        class_terminal_hist = {}
        for col, name in zip(range(len(stat_names)), stat_names):
            sheet.write(0, col, name, title_style)
        # Train/validation loop
        for epoch in range(old_epochs+1, num_epochs):
            # Initializing histrogram for current epoch
            class_terminal_hist[epoch] = list(np.zeros(steps_per_episode))
            epoch_rewards = []
            terminal_hist = list(np.zeros(steps_per_episode))
            for episode in range(num_episodes):
                state = environment.reset()
                cum_reward = 0.0
                terminal = False
                # Episode loop
                while not terminal:
                    action = agent.act(states=dict(features=state['features']), deterministic=False)
                    environment.environment.set_agent_classification(agent.tracked_tensors()['agent/policy/action_distribution/probabilities'])
                    state, terminal, reward = environment.execute(actions=action)
                    agent.observe(terminal=terminal, reward=reward)
                    if terminal:
                        if action == train_labels[episode]:
                            epoch_correct += 1
                    cum_reward += reward
                    epoch_rewards.append(cum_reward)
                    current_ep += 1
                # Stats for current episode
                sys.stdout.write('\rEpoch {epoch} - Episode {ep} - Avg Epoch Reward: {cr} - Accuracy: {ec}%'.format(epoch=epoch + old_epochs,
                                                                                                                    ep=episode,
                                                                                                                    cr=round(sum(epoch_rewards) / current_ep, 3),
                                                                                                                    ec=round((epoch_correct / current_ep)*100, 3)
                                                                                                                    ))
                sys.stdout.flush()
            agent.save(directory=checkpoints_dir,
                       filename='agent-{e}'.format(e=epoch+old_epochs),
                       format='hdf5')
            # Reset correct and episode count
            epoch_accuracy = round((epoch_correct / current_ep) * 100, 2)
            epoch_correct = 0
            current_ep = 0
            # Validating at the end of each epoch
            print('\n')
            rewards = []
            correct = 0
            class_attempt = 0
            mov_attempt = 0
            valid_environment.environment.episodes_count = 0
            for i in range(1, len_valid + 1):
                terminal = False
                ep_reward = 0
                state = valid_environment.reset()
                current_ep_num_steps = 0
                internals_valid = agent.initial_internals()
                while not terminal:
                    action, internals_valid = agent.act(states=dict(features=state['features']), internals=internals_valid,
                                                        independent=True, deterministic=True)
                    valid_environment.environment.set_agent_classification(agent.tracked_tensors()['agent/policy/action_distribution/probabilities'])
                    state, terminal, reward = valid_environment.execute(actions=action)
                    if terminal:
                        if action == valid_labels[i-1]:
                            correct += 1
                            class_terminal_hist[epoch][current_ep_num_steps] += 1
                    ep_reward += reward
                    if int(action) < 10:
                        # Add a classification attempt at timestep t
                        terminal_hist[current_ep_num_steps] += 1
                        # Add a classification attempt
                        class_attempt += 1
                    else:
                        mov_attempt += 1
                    current_ep_num_steps += 1
                rewards.append(ep_reward)
                # Computing stats in real time
                avg_reward = np.sum(rewards) / len(rewards)
                avg_class_attempt = class_attempt / i
                avg_mov_attempt = mov_attempt / i
                # Avoiding division by 0 in stats
                if class_attempt == 0:
                    class_attempt = 1
                sys.stdout.write('\rValidation: Episode {ep} - Average reward: {cr} - Correct: {ok}% - Real Class Attempt Accuracy: {okc}% - Avg. Classification Moves: {ca} - Avg. Movement Moves: {ma}'
                                 .format(ep=i, cr=round(avg_reward, 3),
                                         ok=round((correct / i)*100, 2),
                                         okc=round((correct / class_attempt)*100, 2),
                                         ca=round(avg_class_attempt, 2),
                                         ma=round(avg_mov_attempt, 2)))
                sys.stdout.flush()
            # Compute timestep-by-timestep accuracy - #{correct class at timestep t} / #{total class at timestep t}
            class_terminal_hist[epoch] = [x / y for x, y in zip(class_terminal_hist[epoch], terminal_hist)]
            # At the end of each epoch, write a new line with current data on the excel sheet
            sheet.write(epoch + 1, 0, str(epoch+old_epochs), data_style)
            sheet.write(epoch + 1, 1, str(round(avg_reward, 3)), data_style)
            sheet.write(epoch + 1, 2, str(epoch_accuracy) + "%", data_style)
            sheet.write(epoch + 1, 3, str(round((correct / class_attempt) * 100, 2)) + "%", data_style)
            sheet.write(epoch + 1, 4, str(round((correct / i)*100, 2)) + "%", data_style)
            sheet.write(epoch + 1, 5, str(round((avg_class_attempt / steps_per_episode)*100, 2)) + "%", data_style)
            sheet.write(epoch + 1, 6, str(round((avg_mov_attempt / steps_per_episode)*100, 2)) + "%", data_style)
            # Shuffling data at each epoch
            train_images, train_labels, train_distrib, train_RGB_imgs = shuffle_data(dataset=train_images,
                                                                                     labels=train_labels,
                                                                                     distributions=train_distrib,
                                                                                     RGB_imgs=train_RGB_imgs)
            valid_images, valid_labels, valid_distrib, valid_RGB_imgs = shuffle_data(dataset=valid_images,
                                                                                     labels=valid_labels,
                                                                                     distributions=valid_distrib,
                                                                                     RGB_imgs=valid_RGB_imgs)
            # Setting new permutation of data on envs
            environment.environment.dataset = train_images
            environment.environment.labels = train_labels
            environment.environment.distributions = train_distrib
            environment.environment.images = train_RGB_imgs
            valid_environment.environment.dataset = valid_images
            valid_environment.environment.labels = valid_labels
            valid_environment.environment.distributions = valid_distrib
            valid_environment.environment.images = valid_RGB_imgs
            print('\n')
        # Save excel sheet at the end of training
        idx = 3
        params_column = len(stat_names) + 2
        for name, value in zip(parameters_names, parameters):
            sheet.write(idx, params_column, name, title_style)
            sheet.write(idx, params_column + 1, value, data_style)
            idx += 1
        xl_sheet.save(stats_dir + current_time + ".xlsx")
        with open(stats_dir + 'epochs_hist.json', 'w+') as f:
            json.dump(class_terminal_hist, f)
