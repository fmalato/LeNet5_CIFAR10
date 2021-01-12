import numpy as np
import gym

from enum import IntEnum
from gym import spaces


class DyadicConvnetGymEnv(gym.Env):

    metadata = {'render-modes': ['human']}

    class Actions(IntEnum):
        down = 0
        up_top_left = 1
        up_bottom_left = 2
        up_top_right = 3
        up_bottom_right = 4

    def __init__(self, features, distribution, max_steps):
        super(DyadicConvnetGymEnv, self).__init__()
        assert type(features) == dict, "parameter 'features' must be a dict"
        assert type(distribution) == np.ndarray, "parameter 'distribution' must be a numpy ndarray"
        self.features = features
        self.distribution = distribution
        # Will need this for computing the reward
        self.agent_distribution = None
        self.actions = DyadicConvnetGymEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict({'features': spaces.Box(low=0.0, high=1.0, shape=(67,), dtype=np.float32),
                                              'distribution': spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
                                              })
        self.step_count = 0
        self.agent_pos = None
        self.max_steps = max_steps

    def step(self, action):
        self.step_count += 1
        done = False
        reward = 0.0
        action = dict(action)
        old_pos = self.agent_pos
        if action['action'] == self.actions.down:
            if self.agent_pos[0] < len(self.features) - 1:
                self.agent_pos = (self.agent_pos[0] + 1,
                                  int(self.agent_pos[1]/2),
                                  int(self.agent_pos[2]/2))
        elif action['action'] == self.actions.up_top_left:
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1],
                                  2*self.agent_pos[2])
        elif action['action'] == self.actions.up_top_right:
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1] + 1,
                                  2*self.agent_pos[2])
        elif action['action'] == self.actions.up_bottom_left:
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1],
                                  2*self.agent_pos[2] + 1)
        elif action['action'] == self.actions.up_bottom_right:
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1] + 1,
                                  2*self.agent_pos[2] + 1)
        else:
            assert False, 'unknown action'

        if self.step_count >= self.max_steps:
            done = True

        # TODO: probably a shitty criterion, find a better one
        """if action['distribution'] == np.argmax(self.distribution):
            reward += 1.0"""
        # Punishing the agent for illegal actions
        if old_pos[0] == 0 and action['action'] in [self.actions.up_bottom_right, self.actions.up_top_right,
                                                    self.actions.up_top_left, self.actions.up_bottom_left]:
            reward += -1.0
        elif old_pos[0] == len(self.features) - 1 and action['action'] == self.actions.down:
            reward += -1.0
        else:
            reward += 1.0
        """if old_pos[0] == 0:
            reward += 5.0
        elif old_pos[0] == 1:
            reward += 4.0
        elif old_pos[0] == 2:
            reward += 3.0
        elif old_pos[0] == 3:
            reward += 2.0
        elif old_pos[0] == 4:
            reward += 1.0
            
        if old_pos[0] == 0 and action['action'] in [self.actions.up_bottom_right, self.actions.up_top_right,
                                                    self.actions.up_top_left, self.actions.up_bottom_left]:
            reward += -10.0
        elif old_pos[0] == len(self.features) - 1 and action['action'] == self.actions.down:
            reward += -10.0"""

        obs = self.gen_obs()
        # Why {}?
        return obs, reward, done, {}

    def reset(self):
        # Encoded as (layer, x, y)
        self.agent_pos = (4, 0, 0)
        self.step_count = 0
        obs = {
            'features': np.concatenate((self.features[0][0][0], self.agent_pos), axis=0),
            'distribution': self.distribution
        }

        return obs

    def render(self, mode='human', close=False):
        print('render')

    def close(self):
        print('close')

    def gen_obs(self):
        obs = {
            'features': np.concatenate((self.features[self.agent_pos[0]][self.agent_pos[1]][self.agent_pos[2]],
                                       self.agent_pos), axis=0),
            'distribution': self.distribution
        }

        return obs
