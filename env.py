"""
This module defines the basic objects in the scheduling environment, including tasks, stages, jobs, executors, etc.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
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
        """
        Initialize a task.
        :param idx: task index
        :param duration: how much time this task required to run on an executor in average
        :param global_time: records current time slot
        """
        self.idx = idx
        self.duration = duration
        self.global_time = global_time

        self.start_time = np.nan       # start_time is the task's execution begin time, not arrived time
        self.finish_time = np.nan
        self.executor = None
        self.stage = None      # self.stage is assigned when this stage is initialized

    def schedule(self, start_time, duration, executor):
        """
        Allocate the chosen executor to this task, and update the start time, duration, and finish time.
        """
        # this task should never be scheduled beforehand
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
        Get the remaining time for finishing this task.
        """
        if np.isnan(self.start_time) or (self.global_time < self.start_time):
            return self.duration
        return max(0, self.finish_time - self.global_time.cur_time)

    def reset(self):
        self.start_time, self.finish_time, self.executor = np.nan, np.nan, None


class Stage:
    def __init__(self, idx, tasks, task_duration, global_time, np_random):
        """
        Initialize a stage.
        :param idx: stage index
        :param tasks: tasks included in this stage
        :param task_duration: a dict looks like
                                {'first_wave': [a list of first wave duration on each executor],
                                 'rest_wave': [a list of first rest duration on each executor],
                                 'fresh_durations': [a list of warmup delay on each executor]}
        :param global_time: records current time slot
        :param np_random: used for randomly sample executor key
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

        # these vars are initialized when the corresponding job is initialized
        self.parent_stages = []
        self.child_stages = []
        self.descendants = []
        self.job = None
        for task in self.tasks:
            task.stage = self

    def get_duration(self):
        """
        TODO: when this func is called? Tasks are executed in parallel! Why we need this?
        :return:
        """
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
        """
        TODO: explain this
        :param num_executors: ? not executor's index?
        :return:
        """
        (left_exec, right_exec) = self.job.executor2interval[num_executors]
        if left_exec == right_exec:
            executor_key = left_exec
        else:
            rand_data_point = self.np_random.randint(1, right_exec - left_exec + 1)
            if rand_data_point <= num_executors - left_exec:
                executor_key = left_exec
            else:
                executor_key = right_exec

        if executor_key not in self.task_duration['first_wave']:
            # executor_key = max(self.task_duration['first_wave']).copy()
            largest_key = 0
            for e in self.task_duration['first_wave']:
                if e > largest_key:
                    largest_key = e
            executor_key = largest_key

        return executor_key

    def schedule(self, executor):
        """
        Allocate an executor to this stage.
        Note that the tasks of a stage is executed in wave. How many waves we need is decided by how many
        executors it allocated and how much execution time it required.
        """
        assert self.next_task_idx < self.num_tasks
        task = self.tasks[self.next_task_idx]
        num_executors = len(self.job.executors)
        assert num_executors > 0      # TODO: if this job is not finished, its should has at least one executor allocated? What if it is the first?

        # calculate the actual duration
        executor_key = self.sample_executor_key(num_executors)
        if executor.task is None or executor.task.stage.job != task.stage.job:
            # this executor never runs a task of this job beforehand
            # as a result, we need to add a warmup delay (TODO: interpreted as context switch cost?)
            if len(self.task_duration['fresh_durations'][executor_key]) > 0:
                # retrieve the warmup delay from historical data
                warmup_duration = self.task_duration['fresh_durations'][executor_key]
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
            # this executor runs on this job beforehand but is fresh to this stage
            # as a result, the task duration should be retrieved from 'first_wave' without warmup delay
            if len(self.task_duration['first_wave'][executor_key]) > 0:
                # retrieve the first wave from historical data
                first_wave = self.task_duration['first_wave'][executor_key]
                duration = first_wave[np.random.randint(len(first_wave))]
            else:
                # first wave is non-exist, use warmup delay instead (this condition should happen rarely)
                warmup_duration = self.task_duration['fresh_durations'][executor_key]
                duration = warmup_duration[np.random.randint(len(warmup_duration))]

        # detach the old stage from this executor
        executor.detach_stage()

        # schedule the next-need-to-run task
        task.schedule(self.global_time.cur_time, duration, executor)
        self.executors.add(executor)
        executor.stage = self

        # update stage info
        self.next_task_idx += 1
        self.no_more_task = self.next_task_idx >= self.num_tasks
        if self.no_more_task:
            if self in self.job.frontier_stages:
                self.job.frontier_staes.remove(self)

        return task


class StageDuration:
    """
    TODO: Why we need this?
    """
    def __init__(self, stage):
        self.stage = stage
        self.next_unscheduled_task_idx = 0
        self.duration = self.stage.get_duration()

        # TODO: what these vars mean?
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
        # TODO: is this necessary when initialization?
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
        self.finished = False                # TODO: what about start_time?
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
    TODO: Why we need this?
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
        self.stages_finished = {}       # TODO: no need to be a dict (a set is appropriate)

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
    How we merged: Add a directed link from the sink stage of previous job to the source stages of the next job.
    Continue this process until the last job.
    TODO: how to set the data shuffle?
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
        # reset global idx, and update adj mat based on this
        for stage in job.stages:
            stage.idx += base
            stages.append(stage)
        adj_mat[base: base + num_stages, base: base + num_stages] = job.adj_mat

        # since the second job, create a link from the sink stage of previous job
        # to the source stages of this job
        if base != 0:
            for i in range(num_stages):
                if np.sum(job.adj_mat[:, i]) == 0:
                    # get the source stages
                    assert len(job.stages[i].parent_stages) == 0
                    adj_mat[base - 1, base + i] = 1

        # source stages of current job
        source_stages = []
        for stage in job.stages:
            if len(stage.parent_stages) == 0:
                source_stages.append(stage)

        # update relationship
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
    return Job(stages, adj_mat, args.query_type + '-globally_merged_job')


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


class MovingExecutor:
    def __init__(self):
        """
        self.moving_executors: TODO: moving from or moving to?
        self.stage_track: TODO: explain
        """
        self.moving_executors = {}     # executor: stage
        self.stage_track = {}          # stage: set of executors

    def __contains__(self, executor):
        return executor in self.moving_executors

    def __getitem__(self, executor):
        return self.moving_executors[executor]

    def __len__(self):
        return len(self.moving_executors)

    def add(self, executor, stage):
        executor.detach_job()
        self.moving_executors[executor] = stage
        self.stage_track[stage].add(executor)

    def pop(self, executor):
        if executor in self.moving_executors:
            stage = self.moving_executors[executor]
            self.stage_track[stage].remove(executor)
            del self.moving_executors[executor]
        else:
            # TODO: this job is complete by the time the executor arrives?
            stage = None
        return stage

    def count(self, stage):
        return len(self.stage_track[stage])

    def add_job(self, job):
        for stage in job.stages:
            self.stage_track[stage] = set()

    def remove_job(self, job):
        pass


def get_executor_interval_map():
    """
    Generate args.exec_cap executors, each with different data points pair, such as (5, 5), (5, 10), (90, 100), etc.
    TODO: What is data points pair? The var for computation power?
    e.g.:
        args.executor_data_point example: [5, 10, 20, 40, 50, 60, 80, 100]
        args.exec_cap example: 100
        output:
            {0: (5, 5),
             1: (5, 5),
             2: (5, 5),
             3: (5, 5),
             4: (5, 5),
             5: (5, 5),
             6: (5, 10),
             7: (5, 10),
             8: (5, 10),
             9: (5, 10),
             10: (10, 10),
             ...
             99: (80, 100),
             100: (100, 100)}
    :return: the generated map executor2interval
    """
    executor2interval = dict()

    # the left most
    # i = 0 ---> e: 0 ~ args.executor_data_point[0]
    for e in range(args.executor_data_point[0] + 1):
        executor2interval[e] = (args.executor_data_point[0], args.executor_data_point[0])

    # the center (without head and tail)
    # i: 0 ~ len(args.executor_data_point) - 2 --->
    for i in range(len(args.executor_data_point) - 1):
        # e: executor_data_point[i] + 1 ~ executor_data_point[i + 1] - 1
        for e in range(args.executor_data_point[i] + 1, args.executor_data_point[i + 1]):
            executor2interval[e] = (args.executor_data_point[i], args.executor_data_point[i + 1])
        # e: executor_data_point[i + 1]
        e = args.executor_data_point[i + 1]
        executor2interval[e] = (args.executor_data_point[i + 1], args.executor_data_point[i + 1])

    # the residual
    if args.exec_cap > args.executor_data_point[-1]:
        # e: executor_data_point[i_max] + 1 ~ args.exec_cap
        for e in range(args.executor_data_point[-1] + 1, args.exec_cap + 1):
            executor2interval[e] = (args.executor_data_point[-1], args.executor_data_point[-1])

    return executor2interval


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
    Timeline stores the pair (arrived_time_slot, job).
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


def generate_one_tpch_job(dataset_path, query_size, query_idx, global_time, np_random):
    """
    New a job instance with loaded data.
    :param dataset_path:
    :param query_size:
    :param query_idx:
    :param global_time:
    :param np_random:
    :return:
    """
    assert args.query_type == 'tpch'
    query_path = dataset_path + query_size + '/'
    adj_mat = np.load(query_path + 'adj_mat_' + str(query_idx) + '.npy', allow_pickle=True)
    task_durations = np.load(query_path + 'task_duration_' + str(query_idx) + '.npy', allow_pickle=True)

    num_stages = adj_mat.shape[0]
    stages = []
    # new each stage instance
    for s in range(num_stages):
        task_duration = task_durations[s]
        e = next(iter(task_duration['first_wave']))
        num_tasks = len(task_duration['first_wave'][e]) + len(task_duration['rest_wave'][e])

        # remove warmup delay from first wave duration
        clean_first_wave = {}
        for e in task_duration['first_wave']:
            clean_first_wave[e] = []
            warmup_durations = utils.RepeatableSet()
            for d in task_duration['fresh_durations'][e]:
                warmup_durations.add(d)
            for d in task_duration['first_wave'][e]:
                if d not in warmup_durations:
                    clean_first_wave[e].append(d)
                else:
                    warmup_durations.remove(d)
        last_first_wave = []
        for e in sorted(clean_first_wave.keys()):
            if len(clean_first_wave[e]) == 0:
                clean_first_wave[e] = last_first_wave
            last_first_wave = clean_first_wave[e]
        task_duration['first_wave'] = clean_first_wave

        rough_duration = np.mean(
            [d for fwd in task_duration['first_wave'].values() for d in fwd] +
            [d for rwd in task_duration['rest_wave'].values() for d in rwd] +
            [d for wud in task_duration['fresh_durations'].values() for d in wud]
        )

        # generate this stage and corresponding tasks
        tasks = []
        for t in range(num_tasks):
            # the tasks in the same stage share the execution duration
            task = Task(t, rough_duration, global_time)
            tasks.append(task)
        stage = Stage(s, tasks, task_duration, global_time, np_random)
        stages.append(stage)

    # setup parent and child node info
    for p in range(num_stages):
        for c in range(num_stages):
            if adj_mat[p, c] == 1:
                # will ajd_mat[i, i] be 1?
                stages[p].child_stages.append(stages[c])
                stages[c].parent_stages.append(stages[p])
    # setup descendant node info
    for stage in stages:
        if len(stage.parent_stages) == 0:
            stage.descendants = get_descendants(stage)

    # finally, new the job instance
    return Job(stages, adj_mat, name=args.query_type + '-' + query_size + '-' + str(query_idx))


def generate_tpch_jobs(np_random, timeline, global_time):
    assert args.job_folder == './tpch-queries'
    tpch_size = len(args.tpch_size)
    arrived_jobs = utils.OrderedSet()       # store already arrived jobs
    time_slot = 0                           # slot index of millisecond
    for _ in range(args.num_init_dags):
        query_size = args.tpch_size[np_random.randint(tpch_size)]
        query_idx = np_random.randint(args.tpch_num) + 1
        # new a job instance
        job = generate_one_tpch_job(args.job_folder, query_size, query_idx, global_time, np_random)
        job.start_time = time_slot
        job.arrived = True
        arrived_jobs.add(job)

    # generate future jobs
    for _ in range(args.num_stream_dags):
        # generate job arrival time according to in poisson distribution
        time_slot += int(np_random.exponential(args.stream_interval))
        # query size and idx are sampled from uniform distribution
        query_size = args.tpch_size[np_random.randint(tpch_size)]
        query_idx = np_random.randint(args.tpch_num) + 1
        # new a job instance
        job = generate_one_tpch_job(args.job_folder, query_size, query_idx, global_time, np_random)
        job.start_time = time_slot
        timeline.push(time_slot, job)

    return arrived_jobs


def load_alibaba_cluster_trace_jobs():
    """
    TODO: Add alibaba cluster trace for offline training.
    :return:
    """
    assert args.query_type == 'alibaba'
    pass
