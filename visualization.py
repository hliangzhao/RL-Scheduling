"""
This module defines the functions to visualize executor usage and job run tm.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
from params import args
import utils
import numpy as np
import matplotlib.pyplot as plt


def show_exec_usage(jobs, save_path):
    # get job completion time
    jct = int(np.ceil(np.max(
        [job.finish_time for job in jobs]
    )))
    job_durations = [job.finish_time - job.start_time for job in jobs]
    exec_occupation = np.zeros(jct)
    exec_limit = np.ones(jct) * args.exec_cap
    num_jobs_in_sys = np.zeros(jct)

    for job in jobs:
        for stage in job.stages:
            for task in stage.tasks:
                exec_occupation[int(task.start_time): int(task.finish_time)] += 1
        num_jobs_in_sys[int(job.start_time): int(job.finish_time)] += 1

    exec_usage = np.sum(exec_occupation) / np.sum(exec_limit)
    fig = plt.Figure()
    plt.subplot(2, 1, 1)
    plt.plot(utils.moving_average(exec_occupation, 10000))
    plt.ylabel('Number of busy executors')
    plt.title('Executor usage: ' + str(exec_usage) + '\n Average JCT: ' + str(np.mean(job_durations)))

    plt.subplot(2, 1, 2)
    plt.plot(num_jobs_in_sys)
    plt.xlabel('Time (ms)')
    plt.ylabel('Number of jobs in the system')

    fig.savefig(save_path)
    plt.close(fig)


def show_job_time(jobs, executors, save_path, plot_total_time=None, plot_type='stage'):
    all_tasks = []
    # compute each job's finish time, so that we can visualize it later
    jobs_finish_time = []
    jobs_duration = []
    for job in jobs:
        job_finish_time = 0
        for stage in job.stages:
            for task in stage.tasks:
                all_tasks.append(task)
                if task.finish_time > job_finish_time:
                    job_finish_time = task.finish_time
        jobs_finish_time.append(job_finish_time)
        assert job_finish_time == job.finish_time
        jobs_duration.append(job_finish_time - job.start_time)

    # visualize them in a canvas
    if plot_total_time is None:
        canvas = np.ones([len(executors), int(max(jobs_finish_time))]) * args.canvas_base
    else:
        canvas = np.ones([len(executors), int(plot_total_time)]) * args.canvas_base

    base = 0
    bases = {}  # {job: base}

    for job in jobs:
        bases[job] = base
        base += job.num_stages

    for task in all_tasks:
        start_time = int(task.start_time)
        finish_time = int(task.finish_time)
        exec_id = task.executor.idx

        if plot_type == 'stage':
            canvas[exec_id, start_time: finish_time] = bases[task.stage.job] + task.stage.idx
        elif plot_type == 'app':
            canvas[exec_id, start_time: finish_time] = jobs.index(task.stage.job)

    fig = plt.figure()
    # canvas
    # plt.imshow(canvas, interpolation='nearest', aspect='auto')
    # plt.colorbar()
    # plot each job finish time
    for finish_time in jobs_finish_time:
        plt.plot([finish_time, finish_time], [- 0.5, len(executors) - 0.5], 'r')
    plt.title('average JCT: ' + str(np.mean(jobs_duration)))
    fig.savefig(save_path)
    plt.close(fig)
