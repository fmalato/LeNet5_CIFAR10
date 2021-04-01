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
from utils import split_dataset, n_images_per_class_new, shuffle_data


if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        # Network hyperparameters
        batch_size = 50
        sampling_ratio = 0.99
        discount = 0.999
        num_classes = 10
        lstm_horizon = 5
        lstm_units = 128
        steps_per_episode = 15
        policy_lr = 1e-4
        baseline_lr = 1e-3
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
        num_epochs = 10
        partial_dataset = False
        if partial_dataset:
            images_per_class = 1000
        else:
            images_per_class = 4000
        parameters = [batch_size, sampling_ratio, discount, lstm_units, lstm_horizon, steps_per_episode, policy_lr,
                      baseline_lr, e_r, split_ratio, class_penalty, correct_class, illegal_mov, same_position,
                      images_per_class]
        parameters_names = ['Batch Size', 'Sampling Ratio', 'Discount Factor', 'LSTM Units', 'LSTM horizon', 'Steps per Episode',
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
            train_images = np.array(train_images, dtype=np.float32)
            if partial_dataset:
                img_idxs = n_images_per_class_new(n=images_per_class, labels=train_labels, num_classes=len(class_names))
                train_images = np.array([train_images[idx] for idx in img_idxs])
                train_labels = np.array([train_labels[idx] for idx in img_idxs])
            train_images = train_images / 255.0
            # Initializing everything that the env requires to work properly
            if visualize:
                RGB_images = copy.deepcopy(train_images)
            else:
                RGB_images = None
            tmp = []
            distributions = []
            # We extract EVERY single representation to avoid doing it at every episode (MEMORY INTENSIVE)
            idx = 0
            for img in train_images:
                sys.stdout.write('\rComputing image {current}/{num_img}'.format(current=idx, num_img=train_images.shape[0]))
                img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
                tmp.append(net.extract_features(img))
                idx += 1
            # Split training and validation set
            train_images, valid_images, train_labels, valid_labels = split_dataset(dataset=tmp, labels=train_labels,
                                                                                   ratio=split_ratio, num_classes=num_classes)
            # Also getting RGB images for visualization TODO: make it 'visualize'-wise without breaking the environment
            if visualize:
                train_RGB_imgs = RGB_images[:int(RGB_images.shape[0] * split_ratio)]
                valid_RGB_imgs = RGB_images[int(RGB_images.shape[0] * split_ratio):]
            else:
                train_RGB_imgs = None
                valid_RGB_imgs = None
            # We don't need them anymore - Bye bye CNN!
            del tmp, distributions
            del net
            print('\nDone.\n')
            #########################################################################
        # Shuffling everything there is to shuffle
        train_images, train_labels, train_RGB_imgs = shuffle_data(dataset=train_images, labels=train_labels,
                                                                  RGB_imgs=train_RGB_imgs, visualize=visualize)
        valid_images, valid_labels, valid_RGB_imgs = shuffle_data(dataset=valid_images, labels=valid_labels,
                                                                  RGB_imgs=valid_RGB_imgs, visualize=visualize)
        num_episodes = len(train_labels)
        num_images = len(train_labels)
        len_valid = len(valid_labels)
        # Training environment initialization
        environment = DyadicConvnetGymEnv(dataset=train_images,
                                          labels=train_labels,
                                          images=train_RGB_imgs,
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
            directory = 'models/RL/20210331-162629'
            old_epochs = 10
            print('Loading checkpoint. Number of old epochs: %d' % old_epochs)
            agent = Agent.load(directory=directory + '/checkpoints/',
                               filename='agent-{oe}'.format(oe=old_epochs-1),
                               format='hdf5',
                               environment=environment,
                               agent='ppo',
                               max_episode_timesteps=steps_per_episode,
                               network=[
                                       dict(type='lstm', size=lstm_units, horizon=lstm_horizon, activation='relu'),
                               ],
                               baseline=[
                                   dict(type='lstm', size=lstm_units, horizon=lstm_horizon, activation='relu')
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
                               entropy_regularization=e_r
                               )
        else:
            old_epochs = 0
            agent = Agent.create(environment=environment,
                                 agent='ppo',
                                 max_episode_timesteps=steps_per_episode,
                                 network=[
                                     dict(type='lstm', size=lstm_units, horizon=lstm_horizon, activation='relu'),
                                 ],
                                 baseline=[
                                     dict(type='lstm', size=lstm_units, horizon=lstm_horizon, activation='relu')
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
                                 entropy_regularization=e_r
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
        stat_names = ['Epoch', 'Train Avg Reward', 'Epoch Accuracy', 'Valid Avg Reward', 'Valid Accuracy', 'RCA Accuracy', 'Avg.Class', 'Avg.Move']
        cumulative_accuracy = {}
        for col, name in zip(range(len(stat_names)), stat_names):
            sheet.write(0, col, name, title_style)
        # Train/validation loop
        try:
            for epoch in range(old_epochs, old_epochs+num_epochs):
                epoch_rewards = []
                # Initializing cum accuracy vector
                correct_class_per_timestep = list(np.zeros(steps_per_episode))
                # Keep track of classification attempts per timestep
                class_attempts_per_timestep = list(np.zeros(steps_per_episode))
                for episode in range(num_episodes):
                    state = environment.reset()
                    cum_reward = 0.0
                    terminal = False
                    current_step = 0
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
                        if action < 10:
                            class_attempts_per_timestep[current_step] += 1
                            if action == environment.environment.image_class:
                                correct_class_per_timestep[current_step] += 1
                        current_step += 1
                    epoch_rewards.append(cum_reward)
                    current_ep += 1
                    # Stats for current episode
                    sys.stdout.write('\rEpoch {epoch} - Episode {ep} - Avg Epoch Reward: {cr} - Accuracy: {ec}%'.format(epoch=epoch,
                                                                                                                        ep=episode,
                                                                                                                        cr=round(sum(epoch_rewards) / current_ep, 3),
                                                                                                                        ec=round((epoch_correct / current_ep)*100, 3)
                                                                                                                        ))
                    sys.stdout.flush()
                agent.save(directory=checkpoints_dir,
                           filename='agent-{e}'.format(e=epoch),
                           format='hdf5')
                # Reset correct and episode count
                cum_acc = list(np.zeros(steps_per_episode))
                for x in range(1, steps_per_episode + 1):
                    cum_acc[x - 1] = round(sum(correct_class_per_timestep[:x]) / num_episodes, 3)
                cumulative_accuracy[epoch] = cum_acc
                epoch_accuracy = round((epoch_correct / current_ep) * 100, 2)
                epoch_avg_reward = round(sum(epoch_rewards) / current_ep, 3)
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
                    internals_valid = agent.initial_internals()
                    while not terminal:
                        action, internals_valid = agent.act(states=dict(features=state['features']), internals=internals_valid,
                                                            independent=True, deterministic=True)
                        valid_environment.environment.set_agent_classification(agent.tracked_tensors()['agent/policy/action_distribution/probabilities'])
                        state, terminal, reward = valid_environment.execute(actions=action)
                        if terminal:
                            if action == valid_labels[i-1]:
                                correct += 1
                        ep_reward += reward
                        if int(action) < 10:
                            # Add a classification attempt at timestep t
                            # Add a classification attempt
                            class_attempt += 1
                        else:
                            mov_attempt += 1
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
                # At the end of each epoch, write a new line with current data on the excel sheet
                sheet.write(epoch + 1 - old_epochs, 0, str(epoch), data_style)
                sheet.write(epoch + 1 - old_epochs, 1, str(epoch_avg_reward), data_style)
                sheet.write(epoch + 1 - old_epochs, 2, str(epoch_accuracy) + "%", data_style)
                sheet.write(epoch + 1 - old_epochs, 3, str(round(avg_reward, 3)), data_style)
                sheet.write(epoch + 1 - old_epochs, 4, str(round((correct / i)*100, 2)) + "%", data_style)
                sheet.write(epoch + 1 - old_epochs, 5, str(round((correct / class_attempt) * 100, 2)) + "%", data_style)
                sheet.write(epoch + 1 - old_epochs, 6, str(avg_class_attempt), data_style)
                sheet.write(epoch + 1 - old_epochs, 7, str(avg_mov_attempt), data_style)
                # Shuffling data at each epoch
                train_images, train_labels, train_RGB_imgs = shuffle_data(dataset=train_images,
                                                                          labels=train_labels,
                                                                          RGB_imgs=train_RGB_imgs,
                                                                          visualize=visualize)
                valid_images, valid_labels, valid_RGB_imgs = shuffle_data(dataset=valid_images,
                                                                          labels=valid_labels,
                                                                          RGB_imgs=valid_RGB_imgs,
                                                                          visualize=visualize)
                # Setting new permutation of data on envs
                environment.environment.dataset = train_images
                environment.environment.labels = train_labels
                environment.environment.images = train_RGB_imgs
                valid_environment.environment.dataset = valid_images
                valid_environment.environment.labels = valid_labels
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
            xl_sheet_accuracy = xlwt.Workbook()
            sheet_acc = xl_sheet_accuracy.add_sheet(current_time)
            titles = ['Epoch', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14']
            for col, name in zip(range(len(titles)), titles):
                sheet_acc.write(0, col, name, title_style)
            for key in cumulative_accuracy.keys():
                sheet_acc.write(int(key) + 1 - old_epochs, 0, key, data_style)
                col = 1
                for el in cumulative_accuracy[key]:
                    sheet_acc.write(int(key) + 1, col, el, data_style)
                    col += 1
            xl_sheet_accuracy.save(stats_dir + "cumulative_accuracy.xlsx")
        except KeyboardInterrupt:
            # If we interrupt training, we save the excel sheet anyway
            idx = 3
            params_column = len(stat_names) + 2
            for name, value in zip(parameters_names, parameters):
                sheet.write(idx, params_column, name, title_style)
                sheet.write(idx, params_column + 1, value, data_style)
                idx += 1
            xl_sheet.save(stats_dir + current_time + ".xlsx")
            xl_sheet_accuracy = xlwt.Workbook()
            sheet_acc = xl_sheet_accuracy.add_sheet(current_time)
            titles = ['Epoch', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13',
                      'T14']
            for col, name in zip(range(len(titles)), titles):
                sheet_acc.write(0, col, name, title_style)
            for key in cumulative_accuracy.keys():
                sheet_acc.write(int(key) + 1 - old_epochs, 0, key, data_style)
                col = 1
                for el in cumulative_accuracy[key]:
                    sheet_acc.write(int(key) + 1, col, el, data_style)
                    col += 1
            xl_sheet_accuracy.save(stats_dir + "cumulative_accuracy.xlsx")
