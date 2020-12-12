"""
This module defines the wall-clock time, which is used to measure the running time from beginning to now.
"""


class WallTime:
    """
    Define the time horizon to track current time (slot).
    """
    def __init__(self):
        self.cur_time = 0.0

    def update(self, new_time):
        self.cur_time = new_time

    def reset(self):
        self.cur_time = 0.0
