"""
This module defines the functions of generating jobs.
By adjusting args, we can easily implemented batched arrivals and stream arrivals.
"""
import numpy as np
import os
import utils
from params import args
from spark_env.task import Task
from spark_env.stage import Stage
from spark_env.job import Job


def generate_one_tpch_job(dataset_path, query_size, query_idx, wall_time, np_random):
    """
    New a TPC-H query job instance.
    """
    assert args.query_type == 'tpch'
    query_path = os.path.dirname(__file__) + dataset_path + query_size + '/'
    adj_mat = np.load(query_path + 'adj_mat_' + str(query_idx) + '.npy', allow_pickle=True)
    task_durations = np.load(query_path + 'task_duration_' + str(query_idx) + '.npy', allow_pickle=True).item()
    # assert adj_mat.shape[0] == adj_mat.shape[1] == len(task_durations)

    num_stages = adj_mat.shape[0]
    stages = []
    # new each stage instance
    for s in range(num_stages):
        task_duration = task_durations[s]
        e = next(iter(task_duration['first_wave']))
        num_tasks = len(task_duration['first_wave'][e]) + len(task_duration['rest_wave'][e])

        # remove warmup delay from first wave duration
        # a dict has the same shape with task_duration['first_wave'], it is used to replace the original first_wave
        # i.e., {e_1: [list of durations], e_2: [list of durations], ..., e_N: [list of durations]}
        cleaned_first_wave = dict()
        for e in task_duration['first_wave']:
            cleaned_first_wave[e] = []
            fresh_durations = utils.RepeatableSet()
            for d in task_duration['fresh_durations'][e]:
                fresh_durations.add(d)                      # fresh_durations stores all the fresh durations under e
            for d in task_duration['first_wave'][e]:
                if d not in fresh_durations:
                    cleaned_first_wave[e].append(d)         # a duration is clean iff it didn't appeared in task_duration['fresh_durations'][e]
                else:
                    fresh_durations.remove(d)

        # if cleaned_first_wave[e] is empty, we can fill it with the nearest neighbour's first wave records
        # however, we can find that this remedy is flawed because the cleaned_first_wave with the smallest executor key can still be empty!
        # that's why the authors' default param for args.executor_data_point is [5, 10, 20, 40, 50, 60, 80, 100], where the '2' is non-exist!
        last_first_wave = []
        for e in sorted(cleaned_first_wave.keys()):
            if len(cleaned_first_wave[e]) == 0:
                cleaned_first_wave[e] = last_first_wave
            last_first_wave = cleaned_first_wave[e]
        task_duration['first_wave'] = cleaned_first_wave

        # get the estimate of duration for each task of this stage
        rough_duration = np.mean(
            [d for fwd in task_duration['first_wave'].values() for d in fwd] +      # '+' is equal to .extend
            [d for rwd in task_duration['rest_wave'].values() for d in rwd] +
            [d for wud in task_duration['fresh_durations'].values() for d in wud]
        )

        # generate this stage and corresponding tasks
        tasks = []
        for t in range(num_tasks):
            # the tasks in the same stage share the execution duration
            task = Task(idx=t, rough_duration=rough_duration, wall_time=wall_time)
            tasks.append(task)
        stage = Stage(idx=s, tasks=tasks, task_duration=task_duration, wall_time=wall_time, np_random=np_random)
        stages.append(stage)

    # setup parent and child nodes info
    for p in range(num_stages):
        for c in range(num_stages):
            if adj_mat[p, c] == 1:
                stages[p].child_stages.append(stages[c])
                stages[c].parent_stages.append(stages[p])
    # setup descendant node info
    for stage in stages:
        if len(stage.parent_stages) == 0:
            stage.descendant_stages = get_descendants(stage)

    # finally, new the job instance
    return Job(stages=stages, adj_mat=adj_mat, name=args.query_type + '-' + query_size + '-' + str(query_idx))


def get_descendants(stage):
    """
    Recursively get the descendants of given stage.
    This func is called when generating a job instance.
    """
    if len(stage.descendant_stages) > 0:
        return stage.descendant_stages
    stage.descendant_stages = [stage]
    for child_stage in stage.child_stages:
        child_descendants = get_descendants(child_stage)
        for cd in child_descendants:
            # avoid repeat
            if cd not in stage.descendant_stages:
                stage.descendant_stages.append(cd)
    return stage.descendant_stages


def generate_tpch_jobs(np_random, timeline, wall_time):
    """
    Randomly generate jobs with different size and shape.
    In this func, we generate all jobs with poisson distribution (the gap between the start time of fore-and-aft jobs follows
    the exponential distribution).
    """
    arrived_jobs = utils.OrderedSet()       # store already arrived jobs
    tm = 0                                  # slot index of millisecond
    for _ in range(args.num_init_jobs):
        query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
        query_idx = np_random.randint(args.tpch_num) + 1
        # new a job instance
        job = generate_one_tpch_job(args.job_folder, query_size, query_idx, wall_time, np_random)
        job.start_time = tm
        job.arrived = True
        arrived_jobs.add(job)

    # generate future jobs without adding to arrived_jobs but pushing into timeline
    for _ in range(args.num_stream_jobs):
        # job arrival interval follows a exponential distribution
        tm += int(np_random.exponential(args.stream_interval))
        # query size and idx are sampled from uniform distribution
        query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
        query_idx = np_random.randint(args.tpch_num) + 1
        # new a job instance
        job = generate_one_tpch_job(args.job_folder, query_size, query_idx, wall_time, np_random)
        job.start_time = tm
        timeline.push(tm, job)

    return arrived_jobs


def generate_alibaba_jobs():
    """
    TODO: Generate jobs from alibaba cluster trace for online training
    """
    assert args.query_type == 'alibaba'
    assert args.job_folder == './data/alibaba-cluster-trace/'
    pass


def generate_jobs(np_random, timeline, wall_time):
    """
    TODO: update this func to support alibaba cluster trace
    """
    if args.query_type == 'tpch':
        jobs = generate_tpch_jobs(np_random, timeline, wall_time)
        return jobs
    # elif args.query_type == 'alibaba':
    #     jobs = generate_alibaba_jobs(np_random, timeline, wall_time)
    else:
        print('Invalid query type' + args.query_type)
        exit(1)


def get_stages_order(stage, stages_order):
    """
    ========= This func will be used afterwards, it's useless now =========
    Use DFS to get the topological order of stages for a given job (DAG).
    """
    parent_idx = []
    parent_map = {}      # {stage_idx: stage}
    for s in stage.parent_stages:
        parent_idx.append(s.idx)
        parent_map[s.idx] = s
    parent_idx.sort()
    for idx in parent_idx:
        get_stages_order(parent_map[idx], stages_order)
    if stage.idx not in stages_order:
        stages_order.append(stage.idx)
