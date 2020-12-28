import tensorflow as tf
import gym_minigrid
import numpy as np

from tensorforce.agents import Agent
from tensorforce.environments import Environment, OpenAIGym
from tensorforce.execution import Runner
from gym_minigrid import wrappers


with tf.device('/device:CPU:0'):
    # Parameters initialization
    steps_per_episode = 50
    num_episodes = 1000
    # Environment creation
    env = wrappers.gym.make('MiniGrid-Empty-6x6-v0')
    env = wrappers.RGBImgPartialObsWrapper(env)
    env = wrappers.DirectionObsWrapper(env)
    env = wrappers.ImgObsWrapper(env)
    num_actions = env.action_space.n
    env = Environment.create(environment=env,
                             max_episode_timesteps=steps_per_episode,
                             states=dict(
                                 state=dict(type='float', shape=(56, 56, 3))
                             ),
                             actions=dict(type='int', num_values=num_actions),
                             visualize=False
                             )
    # Agent creation
    agent = Agent.create(agent='dqn',
                         environment=env,
                         states=dict(
                                        image=dict(type='float', shape=(56, 56, 3))
                                    ),
                         memory=steps_per_episode,
                         batch_size=4,
                         actions=dict(type='int', num_values=num_actions),
                         )

    """runner = Runner(
        agent=agent,
        environment=env
    )
    
    runner.run(num_episodes=num_episodes, evaluation=False)
    runner.run(num_episodes=50, evaluation=True)"""

    # Training loop definition
    acts = []
    for i in range(num_episodes):
        states = env.reset()
        terminal = False
        avg_reward = 0
        while not terminal:
            actions = agent.act(states=dict(image=states))
            states, terminal, reward = env.execute(actions=actions)
            avg_reward += reward
            agent.observe(terminal=terminal, reward=reward)
            acts.append(actions)
        print('Iteration: {i}    Reward: {r}    Chosen actions: {hist}'.format(hist=np.histogram(acts)[0], i=i,
                                                                               r=avg_reward / steps_per_episode))
        acts = []
