"""
This module defines the basic concept in the environment, including tasks, stages, jobs, executors, etc.
    Author: Hailiang Zhao (extended from https://github.com/hongzimao/decima-sim)
"""
import numpy as np
import heapq
import itertools
import utils
import networkx as nx
from params import args


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
        self.stage = None      # self.stage is assigned when this stage is initialized

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
            return self.duration          # ! may need change
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
        self.descendants = []    # ! when this initialized?
        self.job = None          # self.job is assigned when this job is initialized
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
        """
        Allocate an executor to this stage.
        Note that the tasks of a stage is executed in wave. How many waves we need is decided by how many
        executors it allocated and how much time ist required..
        """
        assert self.next_task_idx < self.num_tasks
        task = self.tasks[self.next_task_idx]
        num_executors = len(self.job.executors)
        assert num_executors > 0      # ! if this job is not finished, its should has at least one executor allocated? What if it is the first?

        executor_key = self.sample_executor_key(num_executors)
        if executor.task is None or executor.task.stage.job != task.stage.job:
            # this executor never runs a task of this job beforehand
            # as a result, we need to add a warmup delay (interpreted as context switch cost) manually
            if len(self.task_duration['warmup'][executor_key]) > 0:
                # retrieve the warmup delay from historical data
                warmup_duration = self.task_duration['warmup'][executor_key]
                duration = warmup_duration[np.random.randint(len(warmup_duration))]
            else:
                # manually add the warmup delay from args
                first_wave = self.task_duration['first_wave'][executor_key]
                duration = first_wave[np.random.randint(len(first_wave))] + args.warmup_delay
        elif executor.task is not None and executor.task.stage == task.stage and \
                len(self.task_duration['rest_wave'][executor_key]) > 0:
            # this executor is running on this stage now
            # as a result, the task duration should be retrieved from 'rest_wave'
            rest_wave = self.task_duration['rest_wave'][executor_key]
            duration = rest_wave[np.random.randint(len(rest_wave))]
        else:
            # this executor runs on this job beforehand but is fresh to this node
            # as a result, the task duration should be retrieved from 'first_wave' without warmup delay
            pass
        pass


class StageDuration:
    """
    Why we need this?
    """
    def __init__(self, stage):
        self.stage = stage
        self.next_unscheduled_task_idx = 0
        self.duration = self.stage.get_duration()

        # ! what these vars mean?
        self.descendant_work = 0
        self.descendant_critical_path = 0


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
    A job is modeled as a DAG, where nodes are stages (DAG's nodes), edges are data shuffle.
    Notice that each job ends with a single final stage in Spark. If not, you can add a final
    virtual stage with zero computation cost.
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

        self.frontier_stages = utils.OrderedSet()
        # is this necessary when initialization?
        for stage in self.stages:
            if stage.is_runnable():
                self.frontier_stages.add(stage)

        # assign this job to its stages
        for stage in self.stages:
            stage.job = self

        self.arrived = False
        self.finished = False
        self.start_time = None
        self.finish_time = np.inf

        # map an executor to an interval
        self.executor2interval = get_executor_interval_map()

    def get_duration(self):
        return sum([stage.get_duration() for stage in self.stages])

    def reset(self):
        for stage in self.stages:
            stage.reset()
        self.num_finished_stages = 0
        self.executors.clear()
        self.frontier_stages.clear()
        for stage in self.stages:
            if stage.is_runnable():
                self.frontier_stages.add(stage)
        self.arrived = False
        self.finished = False                # ! self.start_time?
        self.finish_time = np.inf

    def update_frontier_stages(self, stage):
        is_changed = False
        for child in stage.child_stages:
            if child.is_runnable() and child.idx not in self.frontier_stages:
                self.frontier_stages.add(child)
                is_changed = True
        return is_changed


class JobDuration:
    """
    Why we need this?
    """
    def __init__(self, job):
        self.job = job
        self.stages_duration = {stage: StageDuration(stage) for stage in self.job.stages}

        # initialize descendant_work and descendant_critical_path for each stage
        for stage in self.job.stages:
            self.stages_duration[stage].descendant_work = \
                np.sum([self.stages_duration[s].duration for s in stage.descendants])
            self.stages_duration[stage].descendant_critical_path = \
                np.sum([s.tasks[0].duration for s in stage.descendants])

        self.job_duration = np.sum([self.stages_duration[s].duration for s in self.job.stages])
        self.stages_finished = {}       # ! no need to be a dict (a set is appropriate)

    def update_duration(self):
        wait2remove_duration = 0
        for stage in self.job.stages:
            if stage not in self.stages_finished and stage.all_tasks_done:
                wait2remove_duration += self.stages_duration[stage].duration
                self.stages_finished[stage] = stage
        self.job_duration -= wait2remove_duration


def is_dag(num_stages, adj_mat):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_stages))
    for i in range(num_stages):
        for j in range(num_stages):
            if adj_mat[i, j] == 1:
                graph.add_edge(i, j)
    return nx.is_directed_acyclic_graph(graph)


def merge_dags(jobs):
    """
    Merge jobs (DAGs) into a global DAG.
    How we merged: The sink stage of previous job is pointed to the source stages of the following job.
    Continue this process until the last job.
    ! how to set the data shuffle?
    """
    num_total_stages = sum([job.num_stages for job in jobs])
    stages = []
    adj_mat = np.zeros([num_total_stages, num_total_stages])

    # base is used to set global idx for all stages
    # sink_stages stores the sink stages of previous job
    # (for the first job, the previous is non-exist, thus it is empty)
    base, sink_stages = 0, []
    for job in jobs:
        num_stages = job.num_stages
        # reset global idx and based on this to update adj mat
        for stage in job.stages:
            stage.idx += base
            stages.append(stage)
        adj_mat[base: base + num_stages, base: base + num_stages] = job.adj_mat

        # since the second job, create a link from the sink stage of previous job
        # to the source stages of this job
        if base != 0:
            for i in range(num_stages):
                if np.sum(job.adj_mat[:, i]) == 0:
                    assert len(job.stages[i].parent_stages) == 0
                    adj_mat[base - 1, base + i] = 1

        # source stages of current job
        source_stages = []
        for stage in job.stages:
            if len(stage.parent_stages) == 0:
                source_stages.append(stage)

        # add a directed link from the sink stage of previous job to source stages of current job
        for src_s in source_stages:
            for sin_s in sink_stages:
                sin_s.child_stages.append(src_s)
                src_s.parent_stages.append(sin_s)

        # update this job's sink stage for the next job
        # sink stage should be only one
        sink_stages = []
        for stage in job.stages:
            if len(stage.child_stages) == 0:
                sink_stages.append(stage)
        assert len(sink_stages) == 1

        base += num_stages

    assert len(stages) == adj_mat.shape[0]
    return Job(stages, adj_mat, 'globally-merged-job')


def get_executor_interval_map():
    """
    What is this?
    :return:
    """
    executor2interval = dict()
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


class FreeExecutor:
    """
    This class defines the executors bind to each job and the free executor pool.
    """
    def __init__(self, executors):
        # free_executors[None] is the pool for free executors
        self.free_executors = {None: utils.OrderedSet()}
        for e in executors:
            self.free_executors[None].add(e)

    def __getitem__(self, job):
        return self.free_executors[job]

    def contain_executor(self, job, executor):
        if executor in self.free_executors[job]:
            return True
        return False

    def pop(self, job):
        """
        Pop the first executor of given job.
        """
        executor = next(iter(self.free_executors[job]))
        self.free_executors[job].remove(executor)
        return executor

    def add(self, job, executor):
        """
        ! Is this executor belongs to job?
        """
        if job is None:
            executor.detach_job()
        else:
            executor.detach_stage()
        self.free_executors[job].add(executor)

    def remove(self, executor):
        """
        Remove the executor from its bind job.
        """
        self.free_executors[executor.job].remove(executor)

    def add_job(self, job):
        self.free_executors[job] = utils.OrderedSet()

    def remove_job(self, job):
        """
        Retrieve the given job's executors and put back to the free executor pool.
        """
        for executor in self.free_executors[job]:
            executor.detach_job()
            self.free_executors[None].add(executor)
        del self.free_executors[job]

    def reset(self, executors):
        self.free_executors = {None: utils.OrderedSet()}
        for e in executors:
            self.free_executors[None].add(e)


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
