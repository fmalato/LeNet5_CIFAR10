import time
import json
import random
import ast
from copy import deepcopy

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
    probs = []
    labels = []
    rewards = []
    first_actions = []
    with open(filename, 'r') as f:
        for line in f:
            if 'Episode' not in line:
                if not line.startswith('action'):
                    action, max_prob, class_label, reward, _ = (item.strip() for item in line.split(';'))
                    actions.append([ast.literal_eval(action)[0], ast.literal_eval(action)[1]])
                    probs.append(float(max_prob))
                    labels.append(int(class_label))
                    rewards.append(float(reward.rstrip('\n')))
            else:
                pass
    for i in range(len(actions)):
        if i % 30 == 0:
            first_actions.append(actions[i])

    return actions, probs, labels, rewards, first_actions


def plot_stats(actions, probs, labels, rewards, first_actions):
    fig1, [ax1, ax2] = plt.subplots(2, 1, figsize=(20, 12))
    fig2, [ax3, ax4, ax5] = plt.subplots(1, 3, figsize=(10, 10))
    x_ticks = range(len(actions))
    ax1.plot(x_ticks, probs)
    ax1.set_title('Probabilities (50 episodes)')
    ax1.set_xlabel('timestep')
    ax1.set_ylabel('max probability')
    ax2.plot(x_ticks, rewards)
    ax2.set_title('Rewards (50 episodes)')
    ax2.set_xlabel('timestep')
    ax2.set_ylabel('reward')
    ax3.hist(actions, bins=5, range=[0, 4])
    ax3.set_xlim(0, 4)
    ax3.set_title('Actions (50 episodes)')
    ax4.hist(labels, bins=10, range=[0, 9])
    ax4.set_xlim(0, 9)
    ax4.set_title('Labels (50 episodes)')
    ax5.hist(first_actions, bins=5, range=[0, 4])
    ax5.set_xlim(0, 4)
    ax5.set_title('First action of the episode (50 episodes)')
    plt.show()


def illegal_actions(filename):
    pos = 4
    actions, _, _, _, _ = get_actions(filename)
    illegal_actions = 0
    for a, _ in actions:
        if a[1] == 0:
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


def one_image_per_class(labels, num_classes):
    indexes = []
    selected = []
    for i in range(len(labels)):
        if labels[i] not in selected:
            selected.append([int(labels[i])])
            indexes.append(i)
        if len(indexes) == num_classes:
            break

    return indexes, selected


def n_images_per_class(n, labels, num_classes, starting_index=0):
    indexes = []
    ordered_labels = []
    for j in range(n):
        selected = []
        # avoid repetitions
        i = np.max(indexes)+1 if len(indexes) > 0 else starting_index
        while len(selected) < num_classes:
            if labels[i] not in selected:
                selected.append([int(labels[i])])
                ordered_labels.append([int(labels[i])])
                indexes.append(i)
            i += 1

    return indexes, ordered_labels
