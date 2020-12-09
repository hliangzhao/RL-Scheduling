"""
Task is the smallest component of a job.
A stage is consists of several tasks which can be run in parallel.
"""
import numpy as np


class Task:
    """
    This class defines the basic component of a job, i.e. task.
    Task is the smallest execution unit. It can be executed by only one executor, and can not be preempted during its execution.
    """
    def __init__(self, idx, rough_duration, wall_time):
        """
        Initialize a task.
        :param idx: task index
        :param rough_duration: how much time this task required to run on an executor in average
            Here the duration is a rough duration because it is an estimate (average from historical data).
        :param wall_time: records current time
        """
        self.idx = idx
        self.duration = rough_duration
        self.wall_time = wall_time
        self.stage = None              # assigned when the stage which it belongs to is initialized

        # start_time and finish_time are settled only when the task is being scheduled
        self.start_time = np.nan       # task's execution begin time
        self.finish_time = np.nan      # task's execution finish time
        self.executor = None           # the executor which run this task

    def schedule(self, start_time, duration, executor):
        """
        Dispatch the chosen executor to this task at start_time, then update the duration and the finish_time.
        finish_time can be fixed because task execution is non-preemptive.
        """
        # this task should never be scheduled beforehand
        assert np.isnan(self.start_time) and np.isnan(self.finish_time) and self.executor is None
        self.start_time = start_time
        self.duration = duration
        self.finish_time = self.start_time + self.duration

        # bind
        self.executor = executor
        self.executor.task = self
        self.executor.stage = self.stage
        self.executor.job = self.stage.job

    def get_duration(self):
        """
        ====================== not used ======================
        Get the remaining execution time for finishing this task.
        Note that what it calculated is the pure 'execution' time!
        """
        if np.isnan(self.start_time) or (self.wall_time.cur_time < self.start_time):
            # the former: this task is not scheduled yet
            # the later:  this task is scheduled, but not the right time to execute yet
            return self.duration
        return max(0, self.finish_time - self.wall_time.cur_time)

    def reset(self):
        """
        Reset is used to handle online job arrival.
        """
        self.start_time, self.finish_time, self.executor = np.nan, np.nan, None
