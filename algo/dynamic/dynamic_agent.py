"""
This module implements the resource dynamic partition scheduler.
"""
import numpy as np
from params import args
from algo.agent import Agent
from spark_env import Stage, Job


class DynamicAgent(Agent):
    """
    This policy implements a heuristic, which dynamically partition the cluster resource.
    """
    def __init__(self):
        super(DynamicAgent, self).__init__()

    def get_action(self, obs):
        jobs, src_job, num_src_exec, frontier_stages, exec_limits, exec_commit, moving_executors, action_map = obs

        # explicitly compute unfinished jobs
        num_unfinished_jobs = sum(
            [any(s.next_task_idx + exec_commit.stage_commit[s] + moving_executors.count(s) < s.num_tasks for s in job.stages)
             for job in jobs]
        )

        # compute exec cap
        exec_cap = int(np.ceil(args.exec_cap / max(1, num_unfinished_jobs)))

        exec_map = {}
        for job in jobs:
            exec_map[job] = len(job.executors)
        # count moving_executors in
        for stage in moving_executors.moving_executors.values():
            exec_map[stage.job] += 1
        # count exec_commit in
        for item in exec_commit.commit:
            if isinstance(item, Job):
                job = item
            elif isinstance(item, Stage):
                job = item.job
            elif item is None:
                job = None
            else:
                print('Source', item, 'unknown')
                exit(1)
            for stage in exec_commit.commit[item]:
                if stage is not None and stage.job != job:
                    exec_map[stage.job] += exec_commit.commit[item][stage]

        if src_job is not None:
            for stage in src_job.frontier_stages:
                if stage in frontier_stages:
                    return stage, num_src_exec
            for stage in frontier_stages:
                if stage.job == src_job:
                    return stage, num_src_exec

        # the src_job is finished or non-exist (None)
        for job in jobs:
            if exec_map[job] < exec_cap:
                next_stage = None
                for stage in job.frontier_stages:
                    if stage in frontier_stages:
                        next_stage = stage
                        break
                if next_stage is None:
                    for stage in frontier_stages:
                        if stage in job.stages:
                            next_stage = stage
                            break
                if next_stage is not None:
                    use_exec = min(
                        # TODO: should be next_stage?
                        stage.num_tasks - stage.next_task_idx - exec_commit.stage_commit[stage] - moving_executors.count(stage),
                        exec_cap - exec_map[job],
                        num_src_exec
                    )
                    return stage, use_exec

        # more exec than tasks
        return None, num_src_exec
