"""
Some utilities.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
from collections import OrderedDict
import sys
import os
import networkx as nx
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


def is_dag(num_stages, adj_mat):
    """
    Judge a job (represented by adj_mat) is a DAG or not.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(num_stages))
    for i in range(num_stages):
        for j in range(num_stages):
            if adj_mat[i, j] == 1:
                graph.add_edge(i, j)
    return nx.is_directed_acyclic_graph(graph)


def get_stages_order(stage, stages_order):
    """
    Use DFS to get the topological order of stages for a given job (DAG).
    """
    parent_idx = []
    parent_map = {}      # bridge the idx and the corresponding stage
    for s in stage.parent_stages:
        parent_idx.append(s.idx)
        parent_map[s.idx] = s
    parent_idx.sort()
    for idx in parent_idx:
        get_stages_order(parent_map[idx], stages_order)
    if stage.idx not in stages_order:
        stages_order.append(stage.idx)


def get_descendants(stage):
    """
    Recursively get the descendants of given stage.
    """
    if len(stage.descendants) > 0:
        return stage.descendants
    stage.descendants = [stage]
    for child_stage in stage.child_stages:
        child_descendants = get_descendants(child_stage)
        for cd in child_descendants:
            # avoid repeat
            if cd not in stage.descendants:
                stage.descendants.append(cd)
    return stage.descendants


def progress_bar(count, total, status='', pattern='#', back='-'):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = pattern * filled_len + back * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s  %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

    if count == total:
        print('')


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


def nonzero_min(arr_x):
    y = arr_x.copy()
    return min(y.remove(0))


def increase_var(var, max_var, increase_rate):
    if var + increase_rate <= max_var:
        var += increase_rate
    else:
        var = max_var
    return var


def decrease_var(var, min_var, decrease_rate):
    if var - decrease_rate >= min_var:
        var -= decrease_rate
    else:
        var = min_var
    return var


def truncate_experiences(bool_list):
    """
    Truncate experience.
    Example: bool_list = [True, False, True], return [0, 2, 3]
    """
    batch_points = [idx for idx, bool_v in enumerate(bool_list) if bool_v]
    batch_points.append(len(bool_list))
    return batch_points
