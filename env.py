"""
This module defines the basic concept in the environment, including tasks, stages, jobs, executors, etc.
    Author: Hailiang Zhao and Cheng Zhang
"""
import numpy as np
import heapq
import itertools
import utils
import networkx as nx


class Task:
    """
    This class defines the basic component of a job, i.e. task.
    """
    def __init__(self, idx, duration, global_time):
        self.idx = idx
        self.duration = duration
        self.global_time = global_time

        self.start_time = np.nan
        self.finish_time = np.nan
        self.executor = None
        self.stage = None  # self.stage is assigned when this stage is initialized

    def schedule(self, start_time, duration, executor):
        """
        Allocate the chosen executor to this task, and update the start time, duration, and finish time.
        """
        assert np.isnan(self.start_time) and \
               np.isnan(self.finish_time) and \
               (self.executor is None)
        self.start_time = start_time
        self.duration = duration
        self.finish_time = self.start_time + self.duration

        # allocate the executor to this task, bind this task to the executor
        self.executor = executor
        self.executor.task = self
        self.executor.stage = self.stage
        self.executor.job = self.stage.job

    def get_duration(self):
        """
        If task has started, get the remaining execution time (compared with current time).
        ! Otherwise, return self.duration?
        """
        if np.isnan(self.start_time) or (self.global_time < self.start_time):
            return self.duration  # may need change
        return max(0, self.finish_time - self.global_time.cur_time)

    def reset(self):
        self.start_time, self.finish_time, self.executor = np.nan, np.nan, None


class Stage:
    def __init__(self, idx, tasks, task_duration, global_time, np_random):
        """
        Initialize a stage.
        :param idx:
        :param tasks:
        :param task_duration:
        :param global_time:
        :param np_random: ! what is this?
        """
        self.idx = idx
        self.tasks = tasks
        self.task_duration = task_duration
        self.global_time = global_time
        self.np_random = np_random

        self.num_tasks = len(tasks)
        self.num_finished_tasks = 0
        self.next_task_idx = 0
        self.no_more_task = False
        self.all_tasks_done = False
        self.finish_time = np.inf

        self.executors = utils.OrderedSet()

        self.parent_stages = []
        self.child_stages = []
        self.descendants = []
        self.job = None          # ! when initialize self.job?
        for task in self.tasks:
            task.stage = self

    def get_duration(self):
        return sum([task.get_duration() for task in self.tasks])

    def is_runnable(self):
        """
        Stage is runnable if and only if all its parent stages are finished (and itself is not finished).
        """
        if self.no_more_task or self.all_tasks_done:
            return False
        for stage in self.parent_stages:
            if not stage.all_tasks_done:
                return False
        return True

    def reset(self):
        for task in self.tasks:
            task.reset()
        self.executors.clear()
        self.num_finished_tasks = 0
        self.next_task_idx = 0
        self.no_more_task = False
        self.all_tasks_done = False
        self.finish_time = np.inf

    def sample_executor_key(self, num_executors):
        pass

    def schedule(self, executor):
        pass


def get_stages_order_dfs(stage, stages_order):
    """
    Use DFS to get the topological order of stages for a given job (DAG).
    """
    parent_idx = []
    parent_map = []      # bridge the idx and the corresponding stage
    for stage in stage.parent_stages:
        parent_idx.append(stage.idx)
        parent_map[stage.idx] = stage
    parent_idx.sort()
    for idx in parent_idx:
        get_stages_order_dfs(parent_map[idx], stages_order)
    if stage.idx not in stages_order:
        stages_order.append(stage.idx)


class Job:
    """
    A job is modeled as a DAG, which consists of stages (DAG's nodes).
    """
    def __init__(self, stages, adj_mat, name):
        assert len(stages) == adj_mat.shape[0] and adj_mat.shape[0] == adj_mat.shape[1]
        self.name = name
        self.stages = stages
        self.adj_mat = adj_mat
        self.num_stages = len(stages)
        self.num_finished_stages = 0

        self.executors = utils.OrderedSet()
        assert is_dag(self.num_stages, self.adj_mat)


def is_dag(num_stages, adj_mat):
    pass


class Executor:
    """
    This class defines the executor. It could be a physical machine or VM.
    """
    def __init__(self, idx):
        self.idx = idx
        self.task, self.stage, self.job = [None] * 3

    def detach_stage(self):
        """
        Detach this executor from the bind stage.
        """
        if (self.stage is not None) and (self in self.stage.executors):
            self.stage.executors.remove(self)
        self.stage, self.task = [None] * 2

    def detach_job(self):
        """
        Detach this executor from the bind job and corresponding stage (of this job).
        """
        if (self.job is not None) and (self in self.job.executors):
            self.job.executors.remove(self)
        self.job = None
        self.detach_stage()

    def reset(self):
        self.task, self.stage, self.job = [None] * 3


class GlobalTime:
    """
    Define the current time.
    Each task should has this as a property.
    """
    def __init__(self):
        self.cur_time = 0.

    def update(self, new_time):
        self.cur_time = new_time

    def increment(self, delta):
        self.cur_time += delta

    def reset(self):
        self.cur_time = 0.


class TimeLine:
    """
    ! What is this?
    ! key is priority, item is task?
    """
    def __init__(self):
        self.priority_queue = []
        self.counter = itertools.count()      # count starts from 0

    def __len__(self):
        return len(self.priority_queue)

    def peek(self):
        """
        Peek the first (key, item) pair without pop it.
        """
        if len(self.priority_queue) > 0:
            key, _, item = self.priority_queue[0]
            return key, item
        return None, None

    def push(self, key, item):
        heapq.heappush(self.priority_queue, (key, next(self.counter), item))

    def pop(self):
        """
        Pop the first (key, item) pair from the heap.
        """
        if len(self.priority_queue) > 0:
            key, _, item = heapq.heappop(self.priority_queue)
            return key, item
        return None, None

    def reset(self):
        self.priority_queue = []
        self.counter = itertools.count()
