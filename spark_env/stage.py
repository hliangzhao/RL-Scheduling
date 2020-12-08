"""
This module defines the component of a job, i.e. stage.
If we take the job as a DAG, then stage is the node of the DAG.
"""
import numpy as np
import utils
from params import args


class Stage:
    """
    This class defines the stage of a job. If we model a job as a DAG, then stage is the node in this DAG.
    A stage is composed of tasks which can be executed in parallel.
    """
    def __init__(self, idx, tasks, task_duration, wall_time, np_random):
        """
        Initialize a stage.
        :param idx: stage index
        :param tasks: tasks included in this stage
        :param task_duration: a dict records various execution time of this stage on different executors under different scenarios.
            The dict looks like
                {
                    'fresh_durations': {
                        e_1: [fresh durations (with warmup delay included) of a single task of this stage
                              when e_1 executors are allocated to this stage],
                        e_2: [...],
                        ...
                        e_N: [...]
                    },
                    'first_wave': {
                        e_1: [first wave durations of a single task of this stage when e_1 executors are
                              allocated to this stage],
                        e_2: [...],
                        ...
                        e_N: [...]
                    },
                    'rest_wave': {
                        e_1: [rest wave durations of a single task of this stage when e_1 executors are
                              allocated to this stage],
                        e_2: [...],
                        ...
                        e_N: [...]
                    }
                }
            Here e_1, ..., e_N are 2, 5, 10, 80, 100, respectively.
            The authors only collect the task execution time under the number of e_i executors.
            In data/tpch-queries/task_durations/*.pdf, the data points records the durations collected.
                - the green data points mean the 'fresh_durations';
                - the red data points mean the 'first_wave';
                - the blue data points mean the 'rest_wave'.
            The sum of the 'first_wave' and 'rest_wave' data points num is the number of tasks in this stage.
        :param wall_time: records current time
        :param np_random: isolated random generator
        """
        self.idx = idx
        self.tasks = tasks
        for task in self.tasks:
            task.stage = self
        self.task_duration = task_duration
        self.wall_time = wall_time
        self.np_random = np_random

        self.num_tasks = len(tasks)
        self.num_finished_tasks = 0
        self.next_task_idx = 0         # the next wait-for-scheduling task' index
        self.no_more_task = False
        self.all_tasks_done = False
        self.finish_time = np.inf

        self.executors = utils.OrderedSet()      # executors which marked as running in this stage

        # these vars are initialized when the corresponding job is initialized
        self.parent_stages, self.child_stages, self.descendant_stages = [], [], []
        self.job = None

    def get_duration(self):
        """
        This function calculates the total remaining execution time for finishing this stage.
        Note that what we calculated is the pure 'execution' time!
        """
        return sum([task.get_duration() for task in self.tasks])

    def is_runnable(self):
        """
        Stage is runnable if and only if all its parent stages are finished while itself is not yet finished.
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
        """
        Note that the authors only collect the task durations under 2, 5, ..., 80, 100 executors on one stage.
        However, the executors on one stage could be any integer <= args.num_exec.
        Thus, to use the collected data, we need to map the real executors num to one of [2, 5, ..., 80, 100] (args.executor_data_point).
        The map follows the principle of proximity.
        :param num_executors: real executors num which bound to this stage
        :return: a number in args.executor_data_point
        """
        (left_exec, right_exec) = self.job.executor2interval[num_executors]
        if left_exec == right_exec:
            executor_key = left_exec
        else:
            # choose the nearest executor key
            rand_point = self.np_random.randint(1, right_exec - left_exec + 1)
            if rand_point <= num_executors - left_exec:
                executor_key = left_exec
            else:
                executor_key = right_exec

        if executor_key not in self.task_duration['first_wave']:        # omit .keys()
            # the num of executors is more than the num of tasks in this tage, thus we do not have the record
            # in this case, we choose the maximum executor key collected as a substitute
            largest_key = 0
            for e in self.task_duration['first_wave']:
                if e > largest_key:
                    largest_key = e
            executor_key = largest_key

        return executor_key

    def schedule(self, executor):
        """
        Allocate an executor to the exactly wait-for-scheduling task of this stage.
        To faithfully simulate the actual scenario, the execution time of a task on the same executor could be different.
        We record the execution time under three circumstances (saved in dataset):
            - it is the first time the executor runs on the job (of this stage) ---> 'fresh_duration' (context switch delay counts);
            - the executor has run on the previous stages of the job (of this stage) but is fresh to this stage ---> 'first_wave';
            - the executor has run on this stage beforehand but is fresh to the wait-for-scheduling task ---> 'rest_wave'.
        :return: the scheduled task
        """
        assert self.next_task_idx < self.num_tasks
        task = self.tasks[self.next_task_idx]
        num_executors = len(self.job.executors)
        assert num_executors > 0

        # get the duration of the wait-for-scheduling task when executing on this executor
        # executor_key represents the number of executors allocated to this stage.
        # this value impacts the real execution time of each task of this stage
        executor_key = self.sample_executor_key(num_executors)
        if executor.task is None or executor.task.stage.job != task.stage.job:
            # this executor never execute a task/stage of the job (of this stage) beforehand, use 'fresh_durations'
            if len(self.task_duration['fresh_durations'][executor_key]) > 0:
                # randomly retrieve a fresh duration from recorded historical data
                fresh_durations = self.task_duration['fresh_durations'][executor_key]
                duration = fresh_durations[np.random.randint(len(fresh_durations))]
            else:
                # dataset does not has this record, manually add the read-from-args warmup delay to a sampled first_wave record from historical data
                first_wave = self.task_duration['first_wave'][executor_key]
                duration = first_wave[np.random.randint(len(first_wave))] + args.warmup_delay

        elif executor.task is not None and executor.task.stage == task.stage and \
                len(self.task_duration['rest_wave'][executor_key]) > 0:
            # this executor is running on this stage now
            # as a result, the task duration should be retrieved from 'rest_wave' (if we have the record)
            rest_wave = self.task_duration['rest_wave'][executor_key]
            duration = rest_wave[np.random.randint(len(rest_wave))]

        else:
            # this executor runs on the job (of this stage) beforehand but is fresh to this stage
            # as a result, the task duration should be retrieved from 'first_wave'
            if len(self.task_duration['first_wave'][executor_key]) > 0:
                # retrieve the first wave from dataset
                first_wave = self.task_duration['first_wave'][executor_key]
                duration = first_wave[np.random.randint(len(first_wave))]
            else:
                # first wave data is non-exist in the dataset, use fresh duration instead
                # (this condition should happen rarely)
                fresh_durations = self.task_duration['fresh_durations'][executor_key]
                duration = fresh_durations[np.random.randint(len(fresh_durations))]

        # detach the old stage from this executor (not that this executor is local to the job of this stage)
        executor.detach_stage()

        # schedule this task
        task.schedule(self.wall_time.cur_time, duration, executor)
        self.executors.add(executor)    # the add() operation is interpreted as: mark this exec as running in this stage
        executor.stage = self

        # update stage info, if finished, remove itself from the set of frontier stages
        self.next_task_idx += 1
        self.no_more_task = self.next_task_idx >= self.num_tasks
        if self.no_more_task:
            if self in self.job.frontier_stages:
                self.job.frontier_stages.remove(self)

        return task
