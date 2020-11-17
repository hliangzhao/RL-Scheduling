from collections import OrderedDict
import sys


class OrderedSet:
    """
    Implement an ordered set.
    """
    def __init__(self, contents=()):
        self.set = OrderedDict((c, None) for c in contents)

    def __contains__(self, item):
        return item in self.set

    def __iter__(self):
        return iter(self.set.keys())

    def __len__(self):
        return len(self.set)

    def add(self, item):
        self.set[item] = None

    def clear(self):
        self.set.clear()

    def index(self, item):
        """
        Get hte index of given item (search as the key).
        """
        if item not in self.set.keys():
            print('Item not in set!')
            return None
        idx = 0
        for i in self.set.keys():
            if item == i:
                break
            idx += 1
        return idx

    def pop(self):
        """
        Remove the first item.
        """
        if self.__len__() == 0:
            print('Set is empty!')
            return None
        item = next(iter(self.set))
        del self.set[item]
        return item

    def remove(self, item):
        """
        Remove the given item.
        """
        if self.__len__() == 0:
            print("Set is empty!")
        if item not in self.set.keys():
            print('Item not in set!')
        del self.set[item]

    def to_list(self):
        return [k for k in self.set]

    def update(self, contents):
        for c in contents:
            self.add(c)


class RepeatableSet:
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
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]


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
