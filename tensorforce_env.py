import numpy as np

from tensorforce.environments import Environment


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
        self.c = 2

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
