import time
import json
import random
import ast
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from pylab import cm, hsv


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


def divide_by_class(labels, num_classes):
    by_class = [[] for x in range(num_classes)]
    for i in range(num_classes):
        for l in range(len(labels)):
            if labels[l] == i:
                by_class[i].append(l)

    return by_class


def n_images_per_class_new(n, labels, num_classes):
    by_class = divide_by_class(labels, num_classes)
    selected = np.concatenate([x[:n] for x in by_class])
    # We shuffle here because we don't want labels to be divided by class
    np.random.shuffle(selected)
    return selected


def split_dataset(dataset, labels, ratio=0.8, num_classes=10, save_idxs=False):
    splitting_index = int(len(dataset)*ratio/num_classes)
    # We need data to be balanced
    by_class = divide_by_class(labels, num_classes)
    balanced_train_idxs = []
    balanced_valid_idxs = []
    for i in range(num_classes):
        balanced_train_idxs.append(by_class[i][:splitting_index])
        balanced_valid_idxs.append(by_class[i][splitting_index:])
    balanced_train = []
    train_labels = []
    for c in balanced_train_idxs:
        for el in c:
            balanced_train.append(dataset[el])
            train_labels.append(list(labels[el]))
    balanced_valid = []
    valid_labels = []
    for c in balanced_valid_idxs:
        for el in c:
            balanced_valid.append(dataset[el])
            valid_labels.append(list(labels[el]))
    if save_idxs:
        idxs = {}
        idxs['train'] = np.concatenate(balanced_train_idxs).tolist()
        idxs['valid'] = np.concatenate(balanced_valid_idxs).tolist()
        with open('training_idxs.json', 'w+') as f:
            json.dump(idxs, f)

    return balanced_train, balanced_valid, train_labels, valid_labels


def split_dataset_idxs(dataset, labels, train_idxs, valid_idxs):
    print('\nLoading old splits...')
    print('\nOld training: {x}..., old valid: {y}...'.format(x=train_idxs[:10], y=valid_idxs[:10]))
    dataset = list(dataset)
    labels = list(labels)
    return [dataset[x] for x in train_idxs], [dataset[x] for x in valid_idxs], \
           [labels[x] for x in train_idxs], [labels[x] for x in valid_idxs]


def shuffle_data(dataset, labels, RGB_imgs, visualize=False):
    shuffled_data = []
    shuffled_labels = []
    shuffled_RGB = []
    perm = np.random.permutation(len(dataset))
    for i in range(len(dataset)):
        shuffled_data.append(dataset[perm[i]])
        shuffled_labels.append(labels[perm[i]])
        if visualize:
            shuffled_RGB.append(RGB_imgs[perm[i]])

    return np.array(shuffled_data), np.array(shuffled_labels), (np.array(shuffled_RGB) if visualize else None)


def build_heatmap(positions, dir, show=True):
    if not os.path.exists(dir + '/heatmaps/'):
        os.mkdir(dir + '/heatmaps/')
    # A 16x16 image is not readable and is heavily blurred when rescaled
    env = {}
    env[0] = np.zeros((16, 16), dtype=np.int)
    env[1] = np.zeros((8, 8), dtype=np.int)
    env[2] = np.zeros((4, 4), dtype=np.int)
    env[3] = np.zeros((2, 2), dtype=np.int)
    for el in positions:
        env[el[0]][el[1], el[2]] += 1

    for key in env.keys():
        fig, ax = plt.subplots()
        im = ax.imshow(env[key])
        xticks = np.arange(int(env[key].shape[0]))
        yticks = np.arange(int(env[key].shape[1]))
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)

        for i in range(len(xticks)):
            for j in range(len(yticks)):
                text = ax.text(j, i, env[key][i, j],
                               ha="center", va="center", color="black")

        ax.set_title("Layer {x}".format(x=key))
        fig.tight_layout()
        if show:
            plt.show()

        fig.savefig(dir + '/heatmaps/{s}.png'.format(s=key))


def plot_mov_histogram(dir_path, filepath, num_timesteps=15, nrows=4, ncols=5):
    with open(dir_path + filepath, 'r') as f:
        hist = json.load(f)
        f.close()
    steps = list(range(num_timesteps))
    keys = list(hist.keys())
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*nrows, 3*ncols))
    fig.suptitle('Movement histogram epoch by epoch')
    idx = 0
    if nrows == 1 and ncols == 1:
        axs.bar(steps, hist[str(keys[idx])])
    else:
        for x in range(nrows):
            for y in range(ncols):
                axs[x, y].bar(steps, hist[str(keys[idx])])
                axs[x, y].set_title('Epoch {idx}'.format(idx=idx))
                idx += 1
    plt.show()
    fname = os.path.splitext(filepath)[0]
    fig.savefig(fname=dir_path + '{fname}.png'.format(fname=fname))


def analyze_distributions(dir_path, filepath):
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    predicted = data['predicted']
    true_lab = data['true lab']
    baseline = data['baseline']
    distribs = data['class distr']
    both_right = []
    a_right_b_wrong = []
    a_wrong_b_right = []
    a_wrong_b_wrong_same_lab = []
    a_wrong_b_wrong_diff_lab = []
    almost_right_AwBr = 0
    almost_right = 0
    for i in range(len(predicted)):
        if predicted[i] == true_lab[i] and baseline[i] == true_lab[i]:
            both_right.append(np.max(distribs[i]))
        if predicted[i] == true_lab[i] and baseline[i] != true_lab[i]:
            a_right_b_wrong.append(np.max(distribs[i]))
        if predicted[i] != true_lab[i] and baseline[i] == true_lab[i]:
            a_wrong_b_right.append(np.max(distribs[i]))
            if true_lab[i] in np.argsort(distribs[i])[7:]:
                almost_right_AwBr += 1
        if predicted[i] != true_lab[i] and baseline[i] != true_lab[i]:
            if predicted[i] != baseline[i]:
                a_wrong_b_wrong_diff_lab.append(np.max(distribs[i]))
                if true_lab[i] in np.argsort(distribs[i])[7:]:
                    almost_right += 1
            else:
                a_wrong_b_wrong_same_lab.append(np.max(distribs[i]))

    print('Average peak value for A right - B right: {avg}'.format(avg=sum(both_right) / len(both_right)))
    print('Average peak value for A wrong - B right: {avg}'.format(avg=sum(a_wrong_b_right) / len(a_wrong_b_right)))
    print('Average peak value for A right - B wrong: {avg}'.format(avg=sum(a_right_b_wrong) / len(a_right_b_wrong)))
    print('Average peak value for A wrong - B wrong with same label: {avg}'.format(avg=sum(a_wrong_b_wrong_same_lab) / len(a_wrong_b_wrong_same_lab)))
    print('Average peak value for A wrong - B wrong with different label: {avg}'.format(avg=sum(a_wrong_b_wrong_diff_lab) / len(a_wrong_b_wrong_diff_lab)))
    print('Percentage of A wrong - B wrong with diff labels where right label is in top-3 positions: {p}%'.format(
        p=round(almost_right / len(a_wrong_b_wrong_diff_lab), 2) * 100))
    print('Percentage of A wrong - B right where right label is in top-3 positions: {p}%'.format(
        p=round(almost_right_AwBr / len(a_wrong_b_right), 2) * 100))


def error_corr_matrix(dir_path, filepath):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    predicted = data['predicted']
    true_lab = data['true lab']
    num_classes = len(data['class distr'][0])
    corr_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
    for pred, true in zip(predicted, true_lab):
        if pred != true:
            corr_matrix[pred, true] += 1

    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, corr_matrix[i, j],
                           ha="center", va="center", color="black")

    ax.set_title("Error correlation matrix")
    fig.tight_layout()
    plt.show()
    fig.savefig(dir_path + "error_corr_matrix.png")


def classification_position(dir_path, filepath, ):
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    predicted = data['predicted']
    true_lab = data['true lab']
    positions = data['class position']
    num_pred = len(predicted)
    pred_in_pos = {}
    for i in range(num_pred):
        if positions[i] not in pred_in_pos:
            pred_in_pos[positions[i]] = [0, 0]
        if predicted[i] == true_lab[i]:
            pred_in_pos[positions[i]][0] += 1
        pred_in_pos[positions[i]][1] += 1

    """xs = []
    ys = []
    zs = []
    colors = []
    for key in pred_in_pos.keys():
        k = re.sub('[()]', '', key)
        res = tuple(map(int, k.split(', ')))
        xs.append(res[0])
        ys.append(res[1])
        zs.append(res[2])
        colors.append(pred_in_pos[key][1])

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111, projection='3d')
    colmap = cm.ScalarMappable(cmap='hot')
    colmap.set_array(np.array(colors))

    yg = ax.scatter(xs, ys, zs, c=np.array(colors) / max(np.array(colors)), marker='s')
    cb = fig.colorbar(colmap)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Column')
    ax.set_zlabel('Row')

    plt.show()"""

    print(pred_in_pos)

#plot_mov_histogram(dir_path='models/RL/20210421-130304/stats/', filepath='movement_histogram_test.json', nrows=1, ncols=1)
#analyze_distributions('models/RL/20210421-130304/stats/', 'predicted_labels.json')
#error_corr_matrix('models/RL/20210421-130304/stats/', 'predicted_labels.json')
#classification_position('models/RL/20210421-130304/stats/', 'predicted_labels.json')
