"""
This module defines the scheduling events.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
from env import Timeline, TimeHorizon
from env import Executor, ExecutorCommit, FreeExecutors, MovingExecutors
from env import RewardCalculator
import numpy as np
import utils
from params import args


class Schedule:
    """
    Define the scheduling events.
    """
    def __init__(self):
        self.np_random = np.random.RandomState()
        self.time_horizon = TimeHorizon()
        self.timeline = Timeline()

        self.executors = utils.OrderedSet()
        for exec_idx in range(args.exec_cap):
            self.executors.add(Executor(exec_idx))
        self.free_executors = FreeExecutors(self.executors)
        self.moving_executors = MovingExecutors()
        self.exec_commit = ExecutorCommit()

        self.stage_selected = set()               # TODO: what for?
        self.reward_calculator = RewardCalculator()

        self.exec_to_schedule = None
        self.src_job = None
        self.num_src_exec = -1

    def seed(self, seed):
        self.np_random.seed(seed)

    def add_job(self, job):
        self.moving_executors.add_job(job)
        self.free_executors.add_job(job)
        self.exec_commit.add_job(job)

    def dispatch_executor(self, executor, frontier_changed):
        """
        If the frontier stages changed, dispatch the executor to another stage, and update the exec_to_schedule and related vars.
        Otherwise, keep the executor working on its stage.
        """
        # TODO: executor.stage should not be None?
        if executor.stage is not None and not executor.stage.no_more_task:
            # the stage this executor working on is not finished, no change required
            task = executor.stage.schedule(executor)
            self.timeline.push(task.finish_time, task)
            return
        if frontier_changed:
            # consult all free executors
            src_job = executor.job
            # TODO: what self.exec_commit[executor.stage] is?
            if len(self.exec_commit[executor.stage]) > 0:
                # directly fulfill the commitment
                self.exec_to_schedule = {executor}
                self.schedule()
            else:
                # free this executor
                self.free_executors.add(src_job, executor)
            # executor.job may change after self.schedule(), update is necessary
            self.exec_to_schedule = utils.OrderedSet(self.free_executors[src_job])
            self.src_job = src_job
            self.num_src_exec = len(self.free_executors[src_job])
        else:
            # only need to schedule this executor
            self.exec_to_schedule = {executor}
            if len(self.exec_commit[executor.stage]) > 0:
                # directly fulfill the commitment
                self.schedule()
            else:
                # consult all executors on this stage
                # len(self.exec_to_schedule) != self.num_source_exec can happen
                self.src_job = executor.job
                self.num_src_exec = len(executor.stage.executors)

    def schedule(self):
        pass

    def backup_schedule(self, executor):
        pass

    def get_frontier_stages(self):
        pass

    def get_exec_limits(self):
        pass

    def observe(self):
        pass

    def saturated(self, node):
        pass

    def step(self, next_stage, limit):
        pass

    def remove_job(self, job):
        pass

    def reset(self, max_time=np.inf):
        pass
