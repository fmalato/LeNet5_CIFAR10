import numpy as np
import gym
import tensorflow as tf

from enum import IntEnum
from gym import spaces
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence


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
        self.ground_truth = [1 if i == self.image_class else 0 for i in range(10)]
        # Will need this for computing the reward
        self.agent_classification = None
        self.actions = DyadicConvnetGymEnv.Actions
        self.action_space = spaces.Dict({'classification': spaces.Discrete(len(self.ground_truth)),
                                         'movement': spaces.Discrete(len(self.actions))
                                         })
        # 64 conv features + 3 positional coding
        self.observation_space = spaces.Dict({'features': spaces.Box(low=0.0, high=1.0, shape=(67,), dtype=np.float32)
                                              })
        self.step_count = 0
        self.agent_pos = None
        self.max_steps = max_steps
        self.agent_reward_loss = CategoricalCrossentropy()
        self.class_reward = 0.0
        self.mov_reward = 0.0
        self.right_old_class = np.max(self.distribution)

    def step(self, action):
        self.step_count += 1
        done = False
        self.class_reward = 0.0
        self.mov_reward = 0.0
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

        """# Categorical CrossEntropy between ground truth and classifier
        cross_entropy = self.agent_reward_loss(self.ground_truth, self.agent_classification)
        self.class_reward = -tf.keras.backend.get_value(cross_entropy)
        # Punishing the agent for illegal actions
        if old_pos[0] == 0 and action['movement'] in [self.actions.up_bottom_right, self.actions.up_top_right,
                                                      self.actions.up_top_left, self.actions.up_bottom_left]:
            self.mov_reward = -0.5
        elif old_pos[0] == len(self.features) - 1 and action['movement'] == self.actions.down:
            self.mov_reward = -0.5"""

        gamma = self.agent_classification[action['classification']]
        c_1 = 1.0 if action['classification'] == self.image_class else -1.0
        delta = self.agent_classification[self.image_class] - self.right_old_class
        c_2 = 0.5
        if old_pos[0] == 0 and action['movement'] in [self.actions.up_bottom_right, self.actions.up_top_right,
                                                      self.actions.up_top_left, self.actions.up_bottom_left]:
            c_3 = -0.3
        elif old_pos[0] == len(self.features) - 1 and action['movement'] == self.actions.down:
            c_3 = -0.3
        else:
            c_3 = 0.3

        reward = gamma * c_1 + delta * c_2 + c_3

        self.right_old_class = self.agent_classification[self.image_class]
        obs = self.gen_obs()
        # Why {}?
        return obs, reward, done, {}

    def reset(self):
        # Encoded as (layer, x, y)
        self.agent_pos = (4, 0, 0)
        self.step_count = 0
        self.ground_truth = [1 if i == self.image_class else 0 for i in range(10)]
        self.right_old_class = np.max(self.distribution)

        obs = {
            'features': np.concatenate((self.features[4][0][0], self.agent_pos), axis=0)
        }
        """obs_feats = []
        for i in range(self.agent_pos[1] - 1, self.agent_pos[1] + 2):
            for j in range(self.agent_pos[2] - 1, self.agent_pos[2] + 2):
                if 0 <= i < self.features[self.agent_pos[0]].shape[0] and \
                   0 <= j < self.features[self.agent_pos[0]].shape[0]:
                    obs_feats.append(self.features[self.agent_pos[0]][i][j])
                else:
                    obs_feats.append(np.zeros((64,)))
        obs = {
            'features': np.concatenate((np.concatenate(obs_feats, axis=0), self.agent_pos), axis=0)
        }"""

        return obs

    def gen_obs(self):
        obs = {
            'features': np.concatenate((self.features[self.agent_pos[0]][self.agent_pos[1]][self.agent_pos[2]],
                                        self.agent_pos), axis=0)
        }
        """obs_feats = []
        for i in range(self.agent_pos[1] - 1, self.agent_pos[1] + 2):
            for j in range(self.agent_pos[2] - 1, self.agent_pos[2] + 2):
                if 0 <= i < self.features[self.agent_pos[0]].shape[0] and \
                        0 <= j < self.features[self.agent_pos[0]].shape[0]:
                    obs_feats.append(self.features[self.agent_pos[0]][i][j])
                else:
                    obs_feats.append(np.zeros((64,)))
        obs = {
            'features': np.concatenate((np.concatenate(obs_feats, axis=0), self.agent_pos), axis=0)
        }"""

        return obs
