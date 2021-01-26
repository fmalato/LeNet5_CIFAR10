import time
import json
import random

import numpy as np
import matplotlib.pyplot as plt


# Check execution time of a line/block of lines
class TimeCounter:

    def __init__(self):
        self.starting_point = None
        self.ending_point = None
        self.stats = {}

    def start(self, fn_name):
        if fn_name not in self.stats.keys():
            self.stats[fn_name] = []
        self.starting_point = time.time()

    def end(self, fn_name):
        self.ending_point = time.time()
        self.stats[fn_name].append(self.ending_point - self.starting_point)

    def save_stats(self):
        for key in self.stats.keys():
            self.stats[key] = np.sum(self.stats[key]) / len(self.stats[key])
        with open('stats.json', 'w+') as f:
            json.dump(self.stats, f)


def select_images(num_images, max_range):
    indexes = []
    for i in range(num_images):
        while True:
            index = random.randint(0, max_range - 1)
            if index not in indexes:
                break

    return indexes


def get_actions(filename):
    actions = []
    with open(filename, 'r') as f:
        for line in f:
            if 'Episode' not in line:
                if not line.startswith(' 0.'):
                    action, distribution = (item.strip() for item in line.split(','))
                    actions.append(int(action))
            else:
                pass

    return actions


def plot_actions(actions):
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    ax.plot(range(len(actions)), actions)
    plt.show()


def illegal_actions(filename):
    pos = 4
    actions = get_actions(filename)
    illegal_actions = 0
    for a in actions:
        if a == 0:
            pos += 1
        else:
            pos -= 1
        if pos < 0:
            illegal_actions += 1
            pos = 0
        elif pos > 4:
            illegal_actions += 1
            pos = 4
    print('Number of illegal actions: %d / %d' % (illegal_actions, len(actions)))
    print('Probability of illegal action: %f' % (illegal_actions/len(actions)))
    # TODO: test probability for each checkpoint

