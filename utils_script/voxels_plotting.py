import json
import os
import re

from PIL import Image

from utils import plot_voxels_path

if __name__ == "__main__":
    # Indices of the images in each_position.json that have to be animated
    idxs = [460, 662]
    # Positions of interest for each image
    starting_positions = ['(0, 7, 8)']
    with open('../models/RL/20210428-125328/stats/each_position.json', 'r') as f:
        data = json.load(f)
        f.close()
    positions = {}
    i = 0
    for k in data.keys():
        for key in data[k].keys():
            if key != 'ground truth':
                positions[i] = data[k][key]['positions']
                i += 1
    for x in positions.keys():
        if positions[x][len(positions[x]) - 1] == '(1, 3, 7)':
            print(x, positions[x][0], len(positions[x]))
    for idx in idxs:
        step = 0
        for pos in positions[idx]:
            el = re.sub('[()]', '', pos)
            pos = tuple(map(int, el.split(', ')))
            plot_voxels_path(pos, filename=str(positions[idx][0]) + '_' + str(step))
            step += 1

    for starting_pos in starting_positions:
        imgs = [fp for fp in sorted(os.listdir('voxels/')) if starting_pos in fp]
        img = []
        for el in imgs:
            img.append(Image.open('voxels/' + el))
        img[0].save(fp='voxels/gif_{x}/gif_{x}.gif'.format(x=starting_pos), format='GIF', append_images=img, save_all=True, duration=2000, loop=5)
