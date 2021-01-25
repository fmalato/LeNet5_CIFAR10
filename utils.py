import time
import json
import random
import numpy as np


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
