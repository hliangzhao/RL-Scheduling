"""
This module defines the wall-clock time, which is used to measure the runtime from beginning to now.
"""


class WallTime:
    """
    Define the time horizon to track record of current time (slot).
    Each task and stage should has this as a property.
    """
    def __init__(self):
        self.cur_time = 0.0

    def update(self, new_time):
        self.cur_time = new_time

    def increment(self, tick):
        self.cur_time += tick

    def reset(self):
        self.cur_time = 0.0
