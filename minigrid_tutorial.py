import tensorflow as tf
import gym_minigrid
import numpy as np
import random

from tensorforce.agents import Agent
from tensorforce.environments import Environment, OpenAIGym
from tensorforce.execution import Runner
from tensorforce.core.parameters import Exponential
from gym_minigrid import wrappers


with tf.device('/device:CPU:0'):
    # Parameters initialization
    steps_per_episode = 150
    num_episodes = 20000
    obs_shape = (7, 7, 3)
    #env_name = 'MiniGrid-Empty-8x8-v0'
    env_name = 'MiniGrid-LavaGapS7-v0'
    #env_name = 'MiniGrid-DistShift1-v0'
    # Environment creation
    env = wrappers.gym.make(env_name)
    env = wrappers.ImgObsWrapper(env)
    num_actions = env.action_space.n
    env = Environment.create(environment=env,
                             max_episode_timesteps=steps_per_episode,
                             states=dict(type='float', shape=obs_shape),
                             actions=dict(type='int', num_values=num_actions),
                             visualize=False
                             )
    # Agent creation
    agent = Agent.create(agent='dqn',
                         environment=env,
                         states=dict(type='float', shape=obs_shape),
                         learning_rate=1e-3,
                         memory=100000,
                         batch_size=steps_per_episode,
                         actions=dict(type='int', num_values=num_actions),
                         exploration=dict(type='linear', unit='timesteps', num_steps=num_episodes*steps_per_episode/2,
                                          initial_value=0.99, final_value=0.2),
                         update_frequency=steps_per_episode,
                         horizon=50
                         )

    runner = Runner(
        agent=agent,
        environment=env
    )
    
    runner.run(num_episodes=num_episodes, evaluation=False)
    agent.save(directory='minigrid_checkpoints/{env}/'.format(env=env_name),
               filename='model-{ep}-{env}'.format(ep=num_episodes, env=env_name))

    ########### TEST with visualization #############

    print('Testing agent')
    if env_name == 'MiniGrid-DistShift1-v0':
        env_name = 'MiniGrid-DistShift2-v0'
    env = wrappers.gym.make(env_name)
    env = wrappers.ImgObsWrapper(env)
    num_actions = env.action_space.n
    env = Environment.create(environment=env,
                             max_episode_timesteps=steps_per_episode,
                             states=dict(type='float', shape=obs_shape),
                             actions=dict(type='int', num_values=num_actions),
                             visualize=True
                             )
    # Agent creation
    agent = Agent.load(directory='minigrid_checkpoints/{env}/'.format(env=env_name),
                       filename='model-{ep}-{env}-1.data-00000-of-00001'.format(ep=num_episodes, env=env_name),
                       environment=env
                       )
    runner = Runner(
        agent=agent,
        environment=env
    )
    runner.run(num_episodes=50, evaluation=True)
