"""
Some utilities.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
from collections import OrderedDict
import sys
import os
import numpy as np


class OrderedSet:
    """
    Implement an ordered set.
    """
    def __init__(self, contents=()):
        """
        In default, the item in contents are set as keys, the corresponding values are None.
        """
        self.set = OrderedDict((c, None) for c in contents)

    def __contains__(self, key):
        """
        Key exists or not.
        """
        return key in self.set

    def __iter__(self):
        """
        Iteration over keys.
        """
        return iter(self.set.keys())

    def __len__(self):
        """
        Number of key-value pairs.
        """
        return len(self.set)

    def add(self, key):
        self.set[key] = None

    def clear(self):
        self.set.clear()

    def index(self, key):
        """
        Get the index of given key.
        """
        if key not in self.set.keys():
            print('Item not in set!')
            exit(1)
        idx = 0
        for k in self.set.keys():
            if key == k:
                break
            idx += 1
        return idx

    def pop(self):
        """
        Remove the first key-value pair.
        """
        if self.__len__() == 0:
            print('Set is empty!')
            exit(1)
        item = next(iter(self.set))
        del self.set[item]
        return item

    def remove(self, key):
        """
        Remove the chosen key-value pair according to the chosen key.
        """
        if self.__len__() == 0:
            print("Set is empty!")
            exit(1)
        if key not in self.set.keys():
            print('Item not in set!')
            exit(1)
        del self.set[key]

    def to_list(self):
        """
        Turn the keys into a list.
        """
        return [k for k in self.set]

    def update(self, contents):
        for c in contents:
            self.add(c)


class RepeatableSet:
    """
    A dict with key being the item, value being the item's occurrence number.
    """
    def __init__(self):
        self.set = {}

    def __contains__(self, item):
        return item in self.set

    def add(self, item):
        if item in self.set:
            self.set[item] += 1
        else:
            self.set[item] = 1

    def clear(self):
        self.set.clear()

    def remove(self, item):
        if item not in self.set.keys():
            print('Item not in set!')
            exit(1)
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]


class ReversibleMap:
    def __init__(self):
        self.map = {}
        self.inverse_map = {}

    def __setitem__(self, key, value):
        self.map[key] = value
        self.inverse_map[value] = key
        # key-value pair should be unique
        assert len(self.map) == len(self.inverse_map)

    def __getitem__(self, key):
        return self.map[key]

    def __len__(self):
        return len(self.map)


def compute_cdf(arr, num_bins=100):
    """
    This func returns x, y for plt.plot(x, y).
    """
    values, base = np.histogram(arr, bins=num_bins)
    cumulative = np.cumsum(values)
    return base[:-1], cumulative / float(cumulative[-1])


def progress_bar(count, total, status='', pattern='#', back='-'):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = pattern * filled_len + back * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s  %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

    if count == total:
        print('')


def convert_indices_to_mask(indices, mask_len):
    mask = np.zeros([1, mask_len])
    for idx in indices:
        mask[0, idx] = 1
    return mask


def list2str(num_list):
    return ' '.join([str(num) for num in num_list])


def create_folder(folder_path):
    """
    Create folder if necessary.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def moving_average(arr_x, N):
    # TODO: why not use np.mean?
    return np.convolve(arr_x, np.ones(N) / N, mode='valid')
