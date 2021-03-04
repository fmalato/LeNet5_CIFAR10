import os
import sys
import datetime

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
        batch_size = 5
        discount = 0.99
        num_classes = 10
        lstm_horizon = 5
        steps_per_episode = 15
        policy_lr = 1e-3
        baseline_lr = 1e-2
        e_r = 0.05
        # Control parameters
        visualize = False
        load_checkpoint = False
        train = True
        # Train/test parameters
        num_epochs = 5
        ########################### PREPROCESSING ##############################
        # Network initialization
        net = DyadicConvNet(num_channels=64, input_shape=(1, 32, 32, 3))
        net.load_weights('models/model_CIFAR10/20210303-125114.h5')
        # Dataset initialization
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        train_images, valid_images, train_labels, valid_labels = split_dataset(dataset=train_images, labels=train_labels, ratio=0.8)
        num_episodes = len(train_labels) * num_epochs
        num_images = len(train_labels)
        len_valid = len(valid_labels)
        class_penalties = [0.3]
        for cp in class_penalties:
            print('Current classification penalty term: {x}'.format(x=cp))
            #########################################################################
            # Training environment initialization
            environment = DyadicConvnetGymEnv(network=net,
                                              dataset=train_images if train else test_images,
                                              labels=train_labels if train else test_labels,
                                              max_steps=steps_per_episode,
                                              visualize=visualize,
                                              num_layers=5,
                                              class_penalty=cp
                                              )
            num_actions = len(environment.actions)
            environment = Environment.create(environment=environment,
                                             states=dict(
                                                 features=dict(type=float, shape=(67,)),
                                             ),
                                             actions=dict(type=int, num_values=num_actions+num_classes),
                                             max_episode_timesteps=steps_per_episode
                                             )
            # Validation environment initialization
            if train:
                valid_environment = DyadicConvnetGymEnv(network=net,
                                                        dataset=valid_images,
                                                        labels=valid_labels,
                                                        max_steps=steps_per_episode,
                                                        visualize=visualize,
                                                        num_layers=5
                                                        )
                num_actions = len(valid_environment.actions)
                valid_environment = Environment.create(environment=valid_environment,
                                                       states=dict(
                                                           features=dict(type=float, shape=(67,)),
                                                       ),
                                                       actions=dict(type=int, num_values=num_actions+num_classes),
                                                       max_episode_timesteps=steps_per_episode
                                                       )
            # Agent initialization
            if load_checkpoint:
                directory = 'models/RL/20210226-140541'
                old_episodes = 200000
                print('Loading checkpoint. Last episode: %d' % old_episodes)
                agent = Agent.load(directory=directory,
                                   filename='agent-{x}'.format(x=old_episodes),
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
                                   # Tensorboard initialized only if training
                                   summarizer=dict(
                                       directory='data/summaries',
                                       summaries='all'
                                   ) if train else None,
                                   learning_rate=policy_lr,
                                   batch_size=batch_size,
                                   tracking=['distribution'],
                                   discount=discount,
                                   states=dict(
                                       features=dict(type=float, shape=(67,)),
                                   ),
                                   actions=dict(type=int, num_values=num_actions+num_classes),
                                   entropy_regularization=e_r
                                   )
            else:
                old_episodes = 0
                agent = Agent.create(environment=environment,
                                     agent='ppo',
                                     network=[
                                         dict(type='lstm', size=64, horizon=lstm_horizon, activation='relu'),
                                     ],
                                     baseline=[
                                         dict(type='lstm', size=64, horizon=lstm_horizon, activation='relu')
                                     ],
                                     baseline_optimizer=dict(optimizer='adam', learning_rate=baseline_lr),
                                     # Tensorboard initialized only if training
                                     summarizer=dict(
                                           directory='data/summaries',
                                           summaries='all'
                                       ) if train else None,
                                     learning_rate=policy_lr,
                                     batch_size=batch_size,
                                     tracking=['distribution'],
                                     discount=discount,
                                     states=dict(
                                         features=dict(type=float, shape=(67,)),
                                     ),
                                     actions=dict(type=int, num_values=num_actions+num_classes),
                                     entropy_regularization=e_r
                                     )
            # Parameters for training loop
            first_time = True
            episode = 0
            epoch_correct = 0
            # Where to store checkpoints
            if load_checkpoint:
                save_dir = directory
            else:
                save_dir = 'models/RL/{x}/'.format(x=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            if not load_checkpoint and (save_dir not in os.listdir('models/RL/')):
                os.mkdir(save_dir)
            # Need to call this now not to overwrite file results
            with open(save_dir + 'validation_stats.txt' if train else 'foo.txt', 'w+') as f:
                # Train/test loop
                while episode <= num_episodes:
                    state = environment.reset()
                    cum_reward = 0.0
                    terminal = False
                    first_step = True
                    if not train:
                        internals = agent.initial_internals()
                    # Episode loop
                    while not terminal:
                        if train:
                            action = agent.act(states=dict(features=state['features']))
                        else:
                            action, internals = agent.act(states=dict(features=state['features']), internals=internals,
                                                          independent=True, deterministic=True)
                        environment.environment.set_agent_classification(agent.tracked_tensors()['agent/policy/action_distribution/probabilities'])
                        state, terminal, reward = environment.execute(actions=action)
                        if train:
                            agent.observe(terminal=terminal, reward=reward)
                        if terminal:
                            if action == valid_labels[i - 1]:
                                epoch_correct += 1
                        cum_reward += reward
                        first_step = False
                    # Stats for current episode
                    sys.stdout.write('\rEpisode {ep} - Cumulative Reward: {cr} - Accuracy: {ec}%'.format(ep=episode+old_episodes,
                                                                                                         cr=round(cum_reward, 3),
                                                                                                         ec=round((epoch_correct / (episode % num_images))*100, 2)))
                    sys.stdout.flush()
                    episode += 1
                    # Saving model at the end of each epoch
                    if episode % num_images == 0 and train:
                        agent.save(directory=save_dir,
                                   filename='agent-{ep}'.format(ep=episode+old_episodes),
                                   format='hdf5')
                        # Reset correct count
                        epoch_correct = 0
                    # Validating at the end of each epoch
                    if episode % num_images == 0 and train:
                        print('\n')
                        rewards = []
                        correct = 0
                        class_attempt = 0
                        mov_attempt = 0
                        valid_environment.environment.episodes_count = 0
                        for i in range(1, len_valid + 1):
                            terminal = False
                            ep_reward = 0
                            obs = valid_environment.reset()
                            internals_valid = agent.initial_internals()
                            while not terminal:
                                action, internals = agent.act(states=dict(features=obs['features']), internals=internals_valid,
                                                              independent=True, deterministic=True)
                                environment.environment.set_agent_classification(agent.tracked_tensors()['agent/policy/action_distribution/probabilities'])
                                state, terminal, reward = valid_environment.execute(actions=action)
                                if terminal:
                                    if action == valid_labels[i-1]:
                                        correct += 1
                                ep_reward += reward
                                if int(action) < 10:
                                    class_attempt += 1
                                else:
                                    mov_attempt += 1
                            rewards.append(ep_reward)
                            avg_reward = np.sum(rewards) / len(rewards)
                            avg_class_attempt = class_attempt / i
                            avg_mov_attempt = mov_attempt / i
                            sys.stdout.write('\rValidation: Episode {ep} - Average reward: {cr} - Correct: {ok}% - Avg. Classification Moves: {ca} - Avg. Movement Moves: {ma}'
                                             .format(ep=i, cr=round(avg_reward, 3),
                                                     ok=round((correct / i)*100, 2),
                                                     ca=round(avg_class_attempt, 2),
                                                     ma=round(avg_mov_attempt, 2)))
                            sys.stdout.flush()
                        f.write('%d, %f, %f, %f\n' % (old_episodes+episode, round(avg_reward, 3),
                                                      round((correct / i)*100, 2), round(avg_class_attempt, 2)))
                        print('\n')
