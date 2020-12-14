import numpy as np

from tensorforce.environments import Environment


class DyadicImageEnvironment(Environment):

    def __init__(self, image, net, discount=0.9):
        super().__init__()
        self.image = image
        self.prediction = net.predict(np.reshape(self.image, (1, 32, 32, 3)))
        self.discount = discount
        self.s = {
            0: self.image[:16, :16, :],
            1: self.image[16:, :16, :],
            2: self.image[:16, 16:, :],
            3: self.image[16:, 16:, :]
        }

    def states(self):
        return dict(observation=dict(type='float', shape=(16, 16, 3)))

    def actions(self):
        return dict(type='int', num_values=1)

    def close(self):
        super().close()

    def reset(self):
        state = dict(observation=self.image[:16, :16, :])
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
