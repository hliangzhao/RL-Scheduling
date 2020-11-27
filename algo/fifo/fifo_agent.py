"""
This module implements the default scheduler implemented in Spark, i.e., FIFO (first come first serve).
     Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
from algo.agent import Agent


class FIFOAgent(Agent):
    """
    FIFO policy statistically partition the cluster resource.
    """
    def __init__(self, exec_cap):
        super(Agent, self).__init__()
        # set exec limit
        self.exec_cap = exec_cap
        # executor assignment map
        self.exec_map = {}

    def get_action(self, obs):
        jobs, src_job, num_src_exec, frontier_stages, exec_limits, exec_commit, moving_executors, action_map = obs
        for job in jobs:
            if job not in self.exec_map:
                self.exec_map[job] = 0
        for job in list(self.exec_map):
            if job not in jobs:
                del self.exec_map[job]

        if src_job is not None:
            # this job is immediately schedulable
            for stage in src_job.frontier_stages:
                if stage in frontier_stages:
                    return stage, num_src_exec
            for stage in frontier_stages:
                if stage.job == src_job:
                    return stage, num_src_exec

        # the src_job is finished or non-exist (None)
        for job in jobs:
            if self.exec_map[job] < self.exec_cap:
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
                        self.exec_cap - self.exec_map[job],
                        num_src_exec
                    )
                    self.exec_map[job] += use_exec
                    return stage, use_exec

        # more exec than tasks
        return None, num_src_exec
