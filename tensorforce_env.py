import numpy as np
import gym
import tensorflow as tf

from enum import IntEnum
from gym import spaces
from tensorflow.keras.losses import CategoricalCrossentropy


class DyadicConvnetGymEnv(gym.Env):

    metadata = {'render-modes': ['human']}

    class Actions(IntEnum):
        down = 0
        up_top_left = 1
        up_bottom_left = 2
        up_top_right = 3
        up_bottom_right = 4

    def __init__(self, features, image_class, distribution, max_steps):
        super(DyadicConvnetGymEnv, self).__init__()
        assert type(features) == dict, "parameter 'features' must be a dict"
        assert type(distribution) == np.ndarray, "parameter 'distribution' must be a numpy ndarray"
        # CNN representation of the image
        self.features = features
        # CNN distribution over selected image
        self.distribution = distribution
        # Ground truth from CIFAR10
        self.image_class = image_class
        self.ground_truth = [1.0 if i == self.image_class else 0.0 for i in range(10)]
        # Will need this for computing the reward
        self.agent_classification = None
        self.actions = DyadicConvnetGymEnv.Actions
        self.action_space = spaces.Dict({'movement': spaces.Discrete(len(self.actions)),
                                         'classification': spaces.Discrete(len(self.ground_truth))
                                         })
        # 64 conv features + 3 positional coding
        self.observation_space = spaces.Dict({'features': spaces.Box(low=0.0, high=1.0, shape=(67,), dtype=np.float32)
                                              })
        self.step_count = 0
        self.agent_pos = None
        self.max_steps = max_steps
        self.agent_reward_loss = CategoricalCrossentropy()

    def step(self, action):
        self.step_count += 1
        done = False
        reward = 0.0
        action = dict(action)
        old_pos = self.agent_pos
        if action['movement'] == self.actions.down:
            if self.agent_pos[0] < len(self.features) - 1:
                self.agent_pos = (self.agent_pos[0] + 1,
                                  int(self.agent_pos[1]/2),
                                  int(self.agent_pos[2]/2))
        elif action['movement'] == self.actions.up_top_left:
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1],
                                  2*self.agent_pos[2])
        elif action['movement'] == self.actions.up_top_right:
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1] + 1,
                                  2*self.agent_pos[2])
        elif action['movement'] == self.actions.up_bottom_left:
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1],
                                  2*self.agent_pos[2] + 1)
        elif action['movement'] == self.actions.up_bottom_right:
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1] + 1,
                                  2*self.agent_pos[2] + 1)
        else:
            assert False, 'unknown action'

        if self.step_count >= self.max_steps:
            done = True

        # Categorical CrossEntropy between ground truth and classifier
        cross_entropy = self.agent_reward_loss(self.ground_truth, self.agent_classification)
        reward += -tf.keras.backend.get_value(cross_entropy)
        if self.image_class == action['classification']:
            reward += 10.0
        # Punishing the agent for illegal actions
        if old_pos[0] == 0 and action['movement'] in [self.actions.up_bottom_right, self.actions.up_top_right,
                                                      self.actions.up_top_left, self.actions.up_bottom_left]:
            reward += -100.0
        elif old_pos[0] == len(self.features) - 1 and action['movement'] == self.actions.down:
            reward += -100.0

        obs = self.gen_obs()
        # Why {}?
        return obs, reward, done, {}

    def reset(self):
        # Encoded as (layer, x, y)
        self.agent_pos = (4, 0, 0)
        self.step_count = 0
        self.ground_truth = [1.0 if i == self.image_class else 0.0 for i in range(10)]
        obs = {
            'features': np.concatenate((self.features[0][0][0], self.agent_pos), axis=0)
        }

        return obs

    def gen_obs(self):
        obs = {
            'features': np.concatenate((self.features[self.agent_pos[0]][self.agent_pos[1]][self.agent_pos[2]],
                                        self.agent_pos), axis=0)
        }

        return obs
