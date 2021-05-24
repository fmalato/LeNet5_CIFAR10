import time
import json
import random
import ast
import os
import re
import string

import numpy as np
import matplotlib.pyplot as plt
import graphviz as gv

from mpl_toolkits.axes_grid1 import ImageGrid

from keras.losses import CategoricalCrossentropy
from tensorflow.keras import datasets

#TODO: VOXELS

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
    print('Probability of illegal action: %f' % (illegal_actions / len(actions)))


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
        i = np.max(indexes) + 1 if len(indexes) > 0 else starting_index
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
    splitting_index = int(len(dataset) * ratio / num_classes)
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


def build_heatmap(positions, dir, show=True, per_class=False, class_name='', all_epochs=False, epoch=0):
    if not os.path.exists(dir + '/heatmaps/'):
        os.mkdir(dir + '/heatmaps/')
    if all_epochs and not os.path.exists(dir + '/heatmaps/{e}/'.format(e=epoch)):
        os.mkdir(dir + '/heatmaps/{e}/'.format(e=epoch))
    # A 16x16 image is not readable and is heavily blurred when rescaled
    env = {}
    env[0] = np.zeros((16, 16), dtype=np.int)
    env[1] = np.zeros((8, 8), dtype=np.int)
    env[2] = np.zeros((4, 4), dtype=np.int)
    env[3] = np.zeros((2, 2), dtype=np.int)
    for key in positions.keys():
        for el in positions[key]:
            el = re.sub('[()]', '', el)
            res = tuple(map(int, el.split(', ')))
            env[res[0]][res[1], res[2]] += 1

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

        if per_class:
            fig.savefig(dir + '/heatmaps/{s}_{cn}.png'.format(s=key, cn=class_name))
        elif all_epochs:
            fig.savefig(dir + '/heatmaps/{e}/{e}_{s}.png'.format(e=epoch, s=key))
        else:
            fig.savefig(dir + '/heatmaps/{s}.png'.format(s=key))


def plot_mov_histogram(dir_path, filepath, num_timesteps=15, nrows=4, ncols=5):
    with open(dir_path + filepath, 'r') as f:
        hist = json.load(f)
        f.close()
    steps = list(range(num_timesteps))
    keys = list(hist.keys())
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * nrows, 3 * ncols))
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
    for i in predicted.keys():
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
    print('Average peak value for A wrong - B wrong with same label: {avg}'.format(
        avg=sum(a_wrong_b_wrong_same_lab) / len(a_wrong_b_wrong_same_lab)))
    print('Average peak value for A wrong - B wrong with different label: {avg}'.format(
        avg=sum(a_wrong_b_wrong_diff_lab) / len(a_wrong_b_wrong_diff_lab)))
    print('Percentage of A wrong - B wrong with diff labels where correct label is in top-3 positions: {p}%'.format(
        p=round(almost_right / len(a_wrong_b_wrong_diff_lab), 2) * 100))
    print('Percentage of A wrong - B right where correct label is in top-3 positions: {p}%'.format(
        p=round(almost_right_AwBr / len(a_wrong_b_right), 2) * 100))


def error_corr_matrix(dir_path, filepath):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    predicted = data['predicted']
    true_lab = data['true lab']
    num_classes = len(class_names)
    corr_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
    for key in predicted.keys():
        if predicted[key] != true_lab[key]:
            corr_matrix[predicted[key], true_lab[key]] += 1

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
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")
    fig.tight_layout()
    plt.show()
    fig.savefig(dir_path + "error_corr_matrix.png")


def classification_position(dir_path, filepath):
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    predicted = data['predicted']
    true_lab = data['true lab']
    positions = data['positions']
    pred_in_pos = {}
    for key in predicted.keys():
        steps = len(positions[key]) - 1
        if positions[key][steps] not in pred_in_pos:
            pred_in_pos[positions[key][steps]] = [0, 0]
        if predicted[key] == true_lab[key]:
            pred_in_pos[positions[key][steps]][0] += 1
        pred_in_pos[positions[key][steps]][1] += 1

    print(pred_in_pos)


def heatmap_per_class(dir_path, filepath, num_classes=10):
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    true_lab = data['true lab']
    positions = data['positions']
    predicted = data['predicted']
    movement = {}
    for i in range(num_classes):
        movement[i] = {}
        for key in true_lab.keys():
            if true_lab[key] == i:
                movement[i][key] = positions[key]
    for key in movement.keys():
        build_heatmap(movement[key], dir=dir_path, show=False, per_class=True, class_name=class_names[key])


def heatmap_before_classification(dir_path, filepath):
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    positions = data['positions']
    class_name = 'LAST'
    movement = {}
    for key in positions.keys():
        movement[key] = []
        i = 0
        for el in positions[key]:
            if i == len(positions[key]) - 1:
                movement[key].append(el)
                i = 0
            else:
                i += 1
    build_heatmap(movement, dir_path, show=False, per_class=True, class_name=class_name)


def distributions_over_time(dir_path, filepath, plot=False):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    distributions = data['class distr']
    true_lab = data['true lab']
    predicted = data['predicted']
    cat_cross = CategoricalCrossentropy()
    entropies = {}
    # Non classified - Lost chance test
    """diff = []
    diff_not = []"""
    # Class occurrencies test
    labels = np.zeros((10,))
    labels_f = np.zeros((10,))
    if not plot:
        keys = list(distributions.keys())
    else:
        # Cherry-picked examples for plotting
        keys = ['1', '22', '24', '28']
    for key in keys:
        if key not in entropies.keys():
            entropies[key] = []
        for el in distributions[key]:
            margin = [x / sum(el[:10]) for x in el[:10]]
            one_hot_label = [1.0 if x == true_lab[key] else 0.0 for x in range(len(margin))]
            entropies[key].append(cat_cross(margin, one_hot_label).numpy())
        """print(
            'Number of steps: {sn} - Starting entropy: {se} - Ending entropy: {ee} - Predicted: {p} - Ground truth: {gt}'.format(
                sn=len(entropies[key]),
                se=entropies[key][0],
                ee=entropies[key][len(entropies[key]) - 1],
                p=predicted[key] if key in predicted.keys() else 'Non predicted',
                gt=true_lab[key]))"""
        # Non classified - Lost chance test
        """if key in predicted.keys():
            diff.append(abs(entropies[key][len(entropies[key]) - 1] - entropies[key][len(entropies[key]) - 2]))
        else:
            diff_not.append(abs(entropies[key][len(entropies[key]) - 1] - entropies[key][len(entropies[key]) - 2]))
    print(np.average(diff), np.average(diff_not))"""
        # Class occurrencies test
        if key not in predicted.keys() and entropies[key][len(entropies[key]) - 1] < 3:
            labels[true_lab[key]] += 1
        if key not in predicted.keys() and entropies[key][len(entropies[key]) - 1] > 3:
            labels_f[true_lab[key]] += 1
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_ylabel('Occurrences')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.plot(class_names, list(labels), label='Correct non classified')
    ax.plot(class_names, list(labels_f), label='Wrong not classified')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=9,
               ncol=2, mode="expand", borderaxespad=0.)
    fig.savefig(dir_path + 'non_class_occurrences.png')
    plt.show()
    print(labels, labels_f)

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 10))
        axs[0, 0].plot(range(len(entropies[keys[0]])), entropies[keys[0]])
        axs[0, 0].set_title('Right classification')
        axs[0, 0].set_xlabel('Timesteps')
        axs[0, 0].set_ylabel('Entropy')
        axs[0, 1].plot(range(len(entropies[keys[1]])), entropies[keys[1]])
        axs[0, 1].set_title('Non classified episode')
        axs[0, 1].set_xlabel('Timesteps')
        axs[0, 1].set_ylabel('Entropy')
        axs[1, 0].plot(range(len(entropies[keys[2]])), entropies[keys[2]])
        axs[1, 0].set_title('Missed opportunity')
        axs[1, 0].set_xlabel('Timesteps')
        axs[1, 0].set_ylabel('Entropy')
        axs[1, 1].plot(range(len(entropies[keys[3]])), entropies[keys[3]])
        axs[1, 1].set_title('Wrong classification')
        axs[1, 1].set_xlabel('Timesteps')
        axs[1, 1].set_ylabel('Entropy')
        plt.show()
        fig.savefig(dir_path + 'entropies.png')


def generate_graph(data, title='', xlabel='', ylabel='', show=False, save=True, save_name='fig', legend=''):
    x_axis = range(len(data[0]))
    figure, ax = plt.subplots(1, 1, figsize=(8, 6))
    if len(data) > 1:
        for el, label in zip(data, legend):
            ax.plot(x_axis, el, label=label)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=9,
                   ncol=2, mode="expand", borderaxespad=0.)
    else:
        for el in data:
            ax.plot(x_axis, el)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show:
        plt.show()
    if save:
        figure.savefig('figures/' + save_name + '.png')


def each_position(dir_path, filepath):
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    for key in data.keys():
        actions = {}
        for k in data[key].keys():
            if k != "ground truth":
                if data[key][k]["prediction"] not in actions.keys():
                    actions[int(data[key][k]["prediction"])] = 0
                if 0 <= int(data[key][k]["prediction"]) < 10:
                    actions[int(data[key][k]["prediction"])] += 1
                else:
                    if -1 not in actions.keys():
                        actions[-1] = 0
                    actions[-1] += 1
        print(actions)


def image_grid(nrows, ncols, images_dir, name):
    image_paths = sorted(os.listdir(images_dir))
    imgs = []
    for img in image_paths:
        if os.path.isfile(img):
            imgs.append(plt.imread(images_dir + img))
    fig = plt.figure(figsize=(2 * ncols, 2 * nrows))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.1, share_all=True
                     )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for ax, im in zip(grid, imgs):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.savefig(images_dir + "{name}.png".format(name=name), bbox_inches='tight')
    plt.show()


def pattern_corr(dir_path, filepath):
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    actions = {}
    for key in data.keys():
        actions[key] = []
        for k in data[key].keys():
            if k != 'ground truth':
                actions[key].append(data[key][k]["actions"][:len(data[key][k]["actions"]) - 1])

    #TODO: work on visualization with graphviz


def num_different_patterns(dir_path, filepath):
    with open(dir_path + filepath, 'r') as f:
        data = json.load(f)
        f.close()
    actions = []
    for key in data.keys():
        for k in data[key].keys():
            if k != 'ground truth':
                actions.append(str(data[key][k]["actions"][:len(data[key][k]["actions"]) - 1]))
    print(len(list(dict.fromkeys(actions))))


def plot_voxels_path(position, filename):
    x, y, z = np.indices((20, 20, 60))
    if position[0] == 0:
        offset = 2
        z_ax = (0, 1)
    elif position[0] == 1:
        offset = 6
        z_ax = (15, 16)
    elif position[0] == 2:
        offset = 8
        z_ax = (30, 31)
    else:
        offset = 9
        z_ax = (40, 41)

    layer_0 = (x >= 2) & (x < 18) & (y >= 2) & (y < 18) & (z < 1)
    layer_1 = (x >= 6) & (x < 14) & (y >= 6) & (y < 14) & (z >= 15) & (z < 16)
    layer_2 = (x >= 8) & (x < 12) & (y >= 8) & (y < 12) & (z >= 30) & (z < 31)
    layer_3 = (x >= 9) & (x < 11) & (y >= 9) & (y < 11) & (z >= 40) & (z < 41)
    pos = (x >= position[1] + offset) & (x < position[1] + 1 + offset) & \
          (y >= position[2] + offset) & (y < position[2] + 1 + offset) & \
          (z >= z_ax[0]) & (z < z_ax[1])

    voxels = layer_0 | layer_1 | layer_2 | layer_3 | pos

    colors = np.empty(voxels.shape, dtype=object)
    colors[layer_0] = 'blue'
    colors[layer_1] = 'blue'
    colors[layer_2] = 'blue'
    colors[layer_3] = 'blue'
    colors[pos] = 'red'
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    ax.auto_scale_xyz(X=x, Y=y, Z=z)
    ax.set_title('Current position: {pos}'.format(pos=position))

    plt.savefig('voxels/{fname}.png'.format(fname=filename))

# plot_mov_histogram(dir_path='models/RL/20210428-125328/stats/', filepath='movement_histogram_test.json', nrows=1, ncols=1)
#analyze_distributions('models/RL_CIFAR-100/{x}/stats/'.format(x=el), 'predicted_labels.json')
#error_corr_matrix('models/RL/20210515-120754/stats/', 'predicted_labels.json')
#classification_position('models/RL_CIFAR-100/{x}/stats/'.format(x=el), 'predicted_labels.json')
#heatmap_per_class('models/RL/20210428-125328/stats/', 'predicted_labels.json')
#heatmap_before_classification('models/RL/20210515-120754/stats/', 'predicted_labels.json')
"""for el in os.listdir('models/RL_CIFAR-100/'):
    distributions_over_time('models/RL_CIFAR-100/{x}/stats/'.format(x=el), 'predicted_labels.json', plot=True)"""
#each_position('models/RL/20210428-125328/stats/', 'each_position.json')
#image_grid(2, 2, 'models/RL/20210428-125328/heatmaps/', 'heatmap_grid')
"""#with open('models/RL/20210515-120754/stats/each_position.json', 'r') as f:
    data = json.load(f)
    f.close()
positions = {}
i = 0
for k in data.keys():
    for key in data[k].keys():
        if key != 'ground truth':
            positions[i] = data[k][key]['positions']
            i += 1
build_heatmap(positions, 'heatmaps/')    #TODO: adapt this to new data type"""
"""generate_graph([],
               title='Comparison between training and validation average rewards',
               xlabel='Epochs',
               ylabel='Avg Reward',
               show=False,
               save=True,
               save_name='train_val_rew',
               legend=['Training Avg Reward', 'Validation Avg Reward'])"""

# Training acc
"""[32.21, 55.33, 62.37, 65.11, 66.6, 67.91, 68.64, 69.11, 69.95, 70.08, 70.28, 70.77, 70.94, 70.84, 71.17,
                 72.0, 73.1, 73.12, 73.56, 73.34, 73.73, 73.81, 73.5, 73.75, 73.52, 73.72, 73.89, 73.63, 73.64, 73.39]"""
# Valid RCA acc
"""[45.68, 44.44, 62.98, 59.21, 64.44, 59.01, 57.86, 59.8, 66.5, 66.87, 64.85, 66.38, 64.12, 65.25, 66.04,
                 67.78, 67.36, 66.88, 67.15, 67.69, 67.23, 67.92, 67.27, 68.05, 67.36, 67.7, 67.94, 67.42, 67.36, 67.33],
                [56.63, 68.93, 67.98, 67.01, 68.14, 68.67, 69.65, 66.75, 68.52, 68.54, 67.01, 66.94, 66.86, 68.3, 69.38,
                 69.78, 69.41, 69.26, 69.56, 69.18, 69.47, 69.12, 69.34, 69.54, 69.59, 69.43, 69.27, 69.36, 69.41, 69.48]"""
# Classification
"""[0.8066 * 100, 0.6447 * 100, 0.9264 * 100, 0.8836 * 100, 0.9457 * 100, 0.8593 * 100, 0.8307 * 100,
                 0.8959 * 100, 0.9705 * 100, 0.9756 * 100, 0.9678 * 100, 0.9916 * 100, 0.959 * 100, 0.9553 * 100,
                 0.9519 * 100, 0.9714 * 100, 0.9705 * 100, 0.9656 * 100, 0.9653 * 100, 0.9784 * 100, 0.9677 * 100,
                 0.9826 * 100, 0.9701 * 100, 0.9786 * 100, 0.968 * 100, 0.9751 * 100, 0.9808 * 100, 0.972 * 100,
                 0.9705 * 100, 0.9691 * 100]"""
# Movement
"""[5.3199, 7.776, 4.9, 5.759, 5.4486, 6.9698, 8.6732, 7.8924, 6.4587, 6.6214, 7.2806, 5.2721, 7.1392,
                 6.5761, 7.518, 7.1594, 7.6638, 7.873, 7.9247, 7.5301, 8.3381, 7.6874, 8.4006, 8.3559, 9.0197, 8.6715,
                 8.6415, 8.9895, 9.0945, 9.0845]"""
# Rewards
"""[-0.084, 0.654, 0.926, 1.039, 1.109, 1.163, 1.204, 1.217, 1.251, 1.257, 1.274, 1.301, 1.293, 1.29, 1.299,
                 1.331, 1.374, 1.379, 1.389, 1.395, 1.416, 1.416, 1.418, 1.431, 1.432, 1.439, 1.455, 1.448, 1.43, 1.44],
                [-0.726, -0.374, 1.114, 0.981, 1.222, 1.037, 1.104, 1.228, 1.419, 1.444, 1.441, 1.361, 1.39, 1.374, 1.482,
                 1.542, 1.583, 1.576, 1.583, 1.582, 1.633, 1.626, 1.625, 1.642, 1.618, 1.649, 1.662, 1.669, 1.673, 1.665]"""

# Splitting CIFAR-100
"""(train_images, train_labels), (_, _) = datasets.cifar100.load_data()
by_class = divide_by_class(train_labels, num_classes=100)
split_ratio = 0.8
train_idx = []
valid_idx = []
split_data = {}
num_examples = 50
for el in by_class:
    x = el[:num_examples]
    idx_split = int(len(x) * split_ratio)
    train_idx.append(x[:idx_split])
    valid_idx.append(x[idx_split:])
split_data['train'] = np.concatenate(train_idx).tolist()
split_data['valid'] = np.concatenate(valid_idx).tolist()
with open('training_idxs_cifar100_partial.json', 'w+') as f:
    json.dump(split_data, f)
    f.close()"""

#num_different_patterns('models/RL/20210428-125328/stats/', 'each_position.json')
