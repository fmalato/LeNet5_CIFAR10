import numpy as np
import tensorflow as tf

from tensorforce.environments import Environment


class DyadicImageEnvironment(Environment):

    def __init__(self, image, net, discount=0.9):
        super().__init__()
        self.image = image
        self.prediction = net.predict(self.image)
        self.discount = discount
        self.s = {
            0: self.image[:, :15, :15, :],
            1: self.image[:, 16:31, :15, :],
            2: self.image[:, :15, 16:31, :],
            3: self.image[:, 16:31, 16:31, :]
        }

    def states(self):
        return dict(type='float', shape=(15, 15, 3))

    def actions(self):
        return dict(type='int', num_values=4)

    def close(self):
        super().close()

    def reset(self):
        state = self.image[:, :15, :15, :]
        return state

    def execute(self, actions, output):
        next_state = self.s[actions[0]]
        reward = 1.0*self.discount if np.argmax(output) == np.argmax(self.prediction) else -1.0
        return next_state, reward
