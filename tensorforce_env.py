import numpy as np
import gym

from enum import IntEnum
from tensorforce.environments import Environment
from gym import spaces
from gym_minigrid import envs


class DyadicImageEnvironment(Environment):

    def __init__(self, image, net, grid_scale=2, discount=0.9):
        super().__init__()
        self.image = image
        self.prediction = net.predict(np.reshape(self.image, (1, self.image.shape[0], self.image.shape[1], 3)))
        self.discount = discount
        self.state_len = int(self.image.shape[0] / grid_scale)
        tiles = []
        for x in range(0, self.image.shape[0], self.state_len):
            for y in range(0, self.image.shape[1], self.state_len):
                tiles.append(self.image[x:x+self.state_len, y:y+self.state_len, :])
        self.s = {}
        for i in range(len(tiles)):
            self.s[i] = tiles[i]

    def states(self):
        return dict(observation=dict(type='float', shape=(self.state_len, self.state_len, 3)))

    def actions(self):
        return dict(type='int', num_values=1)

    def close(self):
        super().close()

    def reset(self):
        state = dict(observation=self.image[:self.state_len, :self.state_len, :])
        return state

    def execute(self, actions, output):
        next_state = dict(observation=self.s[int(np.argmax(actions))])
        # if the classification is correct get a reward
        if np.argmax(output) == np.argmax(self.prediction):
            reward = 1.0*self.discount
        else:
            reward = -1.0
        # should encourage more accurate predictions by giving a higher reward
        if np.argmax(output) == np.argmax(self.prediction) and np.max(output) > np.max(self.prediction):
            reward += 1.0*self.discount

        return next_state, reward


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
        #assert type(features) == np.ndarray, "parameter 'distribution' must be a numpy ndarray"
        self.features = features
        self.distribution = distribution
        self.actions = DyadicConvnetGymEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict({'features': spaces.Box(low=0.0, high=1.0, shape=(64,), dtype=np.float32),
                                              'distribution': spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
                                              })

        self.step_count = 0

        self.agent_pos = None
        self.max_steps = max_steps

    def step(self, action):
        self.step_count += 1
        done = False
        reward = 0.0

        if action == self.actions.down:
            self.agent_pos = (self.agent_pos[0] + 1 if self.agent_pos[0] < len(self.features[self.agent_pos[0]]) - 1 else 0,
                              int(self.agent_pos[1]/2),
                              int(self.agent_pos[2]/2))
        elif action == self.actions.up_top_left:
            self.agent_pos = (self.agent_pos[0] - 1 if self.agent_pos[0] > 0 else 0,
                              2*self.agent_pos[1],
                              2*self.agent_pos[2])
        elif action == self.actions.up_top_right:
            self.agent_pos = (self.agent_pos[0] - 1 if self.agent_pos[0] > 0 else 0,
                              2*self.agent_pos[1] + 1,
                              2*self.agent_pos[2])
        elif action == self.actions.up_bottom_left:
            self.agent_pos = (self.agent_pos[0] - 1 if self.agent_pos[0] > 0 else 0,
                              2*self.agent_pos[1],
                              2*self.agent_pos[2] + 1)
        elif action == self.actions.up_bottom_right:
            self.agent_pos = (self.agent_pos[0] - 1 if self.agent_pos[0] > 0 else 0,
                              2*self.agent_pos[1] + 1,
                              2*self.agent_pos[2] + 1)
        else:
            assert False, 'unknown action'

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done

    def reset(self):
        # Encoded as (layer, x, y)
        self.agent_pos = (0, 0, 0)
        self.step_count = 0
        obs = {
            'features': self.features[0][0][0],
            'distribution': self.distribution
        }

        return obs

    def render(self, mode='human', close=False):
        print('render')

    def close(self):
        print('close')

    def gen_obs(self):
        obs = {
            'features': self.features[self.agent_pos[0]][self.agent_pos[1]][self.agent_pos[2]],
            'distribution': self.distribution
        }

        return obs

    def set_features(self, features):
        self.features = features

    def set_distribution(self, distribution):
        self.distribution = distribution
