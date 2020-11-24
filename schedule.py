"""
This module defines the scheduling events.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
import numpy as np
import heapq
import itertools
from collections import OrderedDict
from params import args
from env import TimeHorizon
from env import Executor, FreeExecutors, MovingExecutors
from env import generate_jobs
from env import Task, Job
import utils


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

        self.stage_selected = set()
        self.reward_calculator = RewardCalculator()

        # added by me
        self.exec_to_schedule = None
        self.src_job = None
        self.num_src_exec = -1
        self.jobs = None
        self.action_map = None
        self.finished_jobs = None
        self.max_time = None

    def step(self, next_stage, limit):
        """
        One (scheduling event) step forward.
        """
        assert next_stage not in self.stage_selected
        self.stage_selected.add(next_stage)

        executor = next(iter(self.exec_to_schedule))
        src = executor.job if executor.stage is None else executor.stage

        # get the num of valid executors for dispatching
        if next_stage is not None:
            use_exec = min(next_stage.num_tasks - next_stage.next_task_idx -
                           self.exec_commit.stage_commit[next_stage] -
                           self.moving_executors.count(next_stage), limit)
        else:
            use_exec = limit
        assert use_exec > 0

        self.exec_commit.add(src, next_stage, use_exec)
        self.num_src_exec -= use_exec
        assert self.num_src_exec >= 0

        if self.num_src_exec == 0:
            # a new scheduling round
            self.stage_selected.clear()
            self.schedule()

        # run to the next event in the timeline
        while len(self.timeline) > 0 and self.num_src_exec == 0:
            # consult agent by putting executors in src_exec
            time_slot, item = self.timeline.pop()
            self.time_horizon.update(time_slot)  # forward to this time slot

            # according to the type of item, take different action
            if isinstance(item, Task):
                # task finish event
                finished_task = item
                stage = finished_task.stage
                stage.num_finished_tasks += 1

                # update frontier stages
                frontier_changed = False
                if stage.num_finished_tasks == stage.num_tasks:
                    assert not stage.all_tasks_done
                    stage.all_tasks_done = True
                    stage.job.num_finished_stages += 1
                    stage.finish_time = self.time_horizon.cur_time
                    frontier_changed = stage.job.update_frontier_stages(stage)

                # dispatch this executor to a new job
                self.dispatch_executor(finished_task.executor, frontier_changed)

                # update job completion status
                if stage.job.num_finished_stages == stage.job.num_stages:
                    assert not stage.job.finished
                    stage.job.finished = True
                    stage.job.finish_time = self.time_horizon.cur_time
                    self.remove_job(stage.job)

            elif isinstance(item, Job):
                # new job arrives event
                job = item
                assert not job.arrived
                job.arrived = True
                # inform agent that a new job arrives when stream is enabled
                self.jobs.add(job)
                self.add_job(job)
                self.action_map = get_act2stage(self.jobs)
                # dispatch existing free executors to this new job
                if len(self.free_executors[None]) > 0:
                    self.exec_to_schedule = utils.OrderedSet(self.free_executors[None])
                    self.src_job = None
                    self.num_src_exec = len(self.free_executors[None])

            elif isinstance(item, Executor):
                # the event that an executor arrives at some job at some time
                executor = item
                # get the destination (stage) of this executor
                stage = self.moving_executors.pop(executor)
                if stage is not None:
                    # the job (of this stage) is not yet finished when this executor arrives
                    executor.job = stage.job
                    stage.job.executors.add(executor)
                if stage is not None and not stage.no_more_task:
                    # this stage is schedulable
                    if stage in stage.job.frontier_stages:
                        # this stage is immediately runnable
                        task = stage.schedule(executor)
                        self.timeline.push(task.finish_time, task)
                    else:
                        # add this executor to the free executor pool of this job
                        self.free_executors.add(executor.job, executor)
                else:
                    # this stage is saturated or this job is finished, but the executor still arrives to it
                    # in this case, use backup schedule policy
                    self.backup_schedule(executor)

            else:
                print('Illegal event type!')
                exit(1)

        # compute reward
        reward = self.reward_calculator.get_reward(self.jobs, self.time_horizon.cur_time)
        # no more decision to make, jobs all done or time is up
        done = self.num_src_exec == 0 and (len(self.timeline) == 0 or self.time_horizon.cur_time >= self.max_time)
        if done:
            assert self.time_horizon.cur_time >= self.max_time or len(self.jobs) == 0
        return self.observe(), reward, done

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
        # TODO: 'assert executor.stage != None' is unnecessary?
        if executor.stage is not None and not executor.stage.no_more_task:
            # the stage which this executor is working on is not finished, no change required
            task = executor.stage.schedule(executor)
            self.timeline.push(task.finish_time, task)
            return
        if frontier_changed:
            # consult all free executors
            src_job = executor.job
            # TODO [bug]: what self.exec_commit[executor.stage] is?
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
            self.exec_to_schedule = {executor}           # TODO: not OrderedSet()?
            if len(self.exec_commit[executor.stage]) > 0:
                # directly fulfill the commitment
                self.schedule()
            else:
                # consult all executors on this stage
                # len(self.exec_to_schedule) != self.num_source_exec can happen
                self.src_job = executor.job
                self.num_src_exec = len(executor.stage.executors)

    def schedule(self):
        executor = next(iter(self.exec_to_schedule))
        src = executor.job if executor.stage is None else executor.stage
        # schedule executors from the src (stage or job) until the commitment is fulfilled
        while len(self.exec_commit[src]) > 0 and len(self.exec_to_schedule) > 0:
            stage = self.exec_commit.pop(src)
            executor = self.exec_to_schedule.pop()

            if self.free_executors.contain_executor(executor.job, executor):
                self.free_executors.remove(executor)

            if stage is None:
                if executor.job is not None and any([not s.no_more_task for s in executor.job.stages]):
                    self.free_executors.add(executor.job, executor)
                else:
                    # this stage is silent, make the executor idle
                    self.free_executors.add(None, executor)

            elif not stage.no_more_task:
                # stage is not saturated
                if executor.job == stage.job:
                    if stage in stage.job.frontier_stages:
                        # this task is immediately runnable, schedule it
                        task = stage.schedule(executor)
                        self.timeline.push(task.finish_time, task)
                    else:
                        self.free_executors.add(executor.job, executor)
                else:
                    # need to move this executor to another job
                    self.timeline.push(self.time_horizon.cur_time + args.moving_delay, executor)
                    self.moving_executors.add(executor, stage)

            else:
                self.backup_schedule(executor)

    def backup_schedule(self, executor):
        """
        This func is used as backup policy. We add this because a random policy or a learned policy in early
        iterations might schedule no executor to jobs. This func makes sure that all executors are working conservative.

        The agent should learn to not rely on this func.
        """
        backup_scheduled = False
        if executor.job is not None:
            # try to schedule on current job firstly
            for stage in executor.job.frontier_stages:
                if not self.saturated(stage):
                    # schedule this executor to the next-to-run task of this stage
                    task = stage.schedule(executor)
                    self.timeline.push(task.finish_time, task)
                    backup_scheduled = True
                    break
        if not backup_scheduled:
            # try to schedule on any available stage
            schedulable_stages = self.get_frontier_stages()
            if len(schedulable_stages) > 0:
                stage = next(iter(schedulable_stages))
                self.timeline.push(self.time_horizon.cur_time + args.moving_delay, executor)
                self.moving_executors.add(executor, stage)
                backup_scheduled = True
        if not backup_scheduled:
            # no available stage, mark this executor as idle
            self.free_executors.add(executor.job, executor)

    def saturated(self, stage):
        """
        If saturated, all tasks of this stage have been finished.
        """
        expected_task_idx = stage.next_task_idx + self.exec_commit.stage_commit[stage] + self.moving_executors.count(stage)
        return expected_task_idx >= stage.num_tasks

    def get_frontier_stages(self):
        """
        Get all the frontier stages.
        In this class, 'frontier' can be interpreted as itself is unsaturated but all its parent stages are saturated.
        """
        frontier_stages = utils.OrderedSet()
        for job in self.jobs:
            for stage in job.stages:
                if stage not in self.stage_selected and not self.saturated(stage):
                    all_parents_saturated = True
                    for parent in stage.parent_stages:
                        if not self.saturated(parent):
                            all_parents_saturated = False
                            break
                    if all_parents_saturated:
                        frontier_stages.add(stage)
        return frontier_stages

    def get_exec_limits(self):
        """
        Get the minimum executor limit for each job.
        """
        exec_lmt = {}              # {job: int}
        for job in self.jobs:
            if self.src_job == job:
                cur_exec = self.num_src_exec
            else:
                cur_exec = 0
            exec_lmt[job] = len(job.executors) - cur_exec
        return exec_lmt

    def observe(self):
        return self.jobs, self.src_job, self.num_src_exec, self.get_frontier_stages(), \
               self.get_exec_limits(), self.exec_commit, self.moving_executors, self.action_map

    def remove_job(self, job):
        """
        Remove the job when it is finished.
        """
        for executor in job.executors:
            executor.detach_job()
        self.exec_commit.remove_job(job)
        self.free_executors.remove_job(job)
        self.moving_executors.remove_job(job)
        self.jobs.remove(job)
        self.finished_jobs.add(job)
        self.action_map = get_act2stage(self.jobs)

    def reset(self, max_time=np.inf):
        self.max_time = max_time
        self.time_horizon.reset()
        self.timeline.reset()
        self.exec_commit.reset()
        self.moving_executors.reset()
        self.reward_calculator.reset()
        self.finished_jobs = utils.OrderedSet()
        self.stage_selected.clear()
        for executor in self.executors:
            executor.reset()
        self.free_executors.reset(self.executors)

        # regenerate jobs (note that self.jobs are only currently arrived jobs)
        self.jobs = generate_jobs(self.np_random, self.timeline, self.time_horizon)
        self.action_map = get_act2stage(self.jobs)
        for job in self.jobs:
            self.add_job(job)
        self.src_job = None
        # all executors are schedulable
        self.num_src_exec = len(self.executors)
        self.exec_to_schedule = utils.OrderedSet(self.executors)


class Timeline:
    """
     Stores the pair (time_slot, job/task/executor).
     The time slot could be
        - the arrival time (of a job),
        - the finish time (of a task),
        - the time an executor arrives at some job
    """
    def __init__(self):
        """
        self.priority_queue stores the tuple: (key, counter, item).
        """
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


class RewardCalculator:
    """
    Use the execution time for now to calculate the reward.
    For every job still in system, reward will add the negative of the job's executing time till now.
    Obviously, longer each job's execution time, more punishment the Agent receives.
    """
    def __init__(self):
        self.jobs = set()                   # jobs that not finished during [prev_time, cur_time)
        self.prev_time = 0                  # previous reward calculation time

    def get_reward(self, jobs, cur_time):
        reward = 0
        for job in jobs:
            self.jobs.add(job)

        if args.learn_obj == 'mean':
            for job in list(self.jobs):
                reward -= (min(job.finish_time, cur_time) - max(job.start_time, self.prev_time)) \
                          / args.reward_scale
                if job.finished:
                    self.jobs.remove(job)
        elif args.learn_job == 'makespan':
            reward -= (cur_time - self.prev_time) / args.reward_scale
        else:
            print('Unsupported learn objective!')
            exit(1)

        self.prev_time = cur_time
        return reward

    def reset(self):
        self.jobs.clear()
        self.prev_time = 0


class ExecutorCommit:
    """
    TODO: How this works?
    """
    def __init__(self):
        self.commit = {}             # {stage/job: OrderedDict(stage: amount)}
        self.stage_commit = {}       # {stage: amount}
        self.backward = {}           # {stage: set(stages/jobs)}

    def __getitem__(self, src):
        return self.commit[src]

    def add(self, src, stage, amount):
        """
        Add a commitment.
        :param src: could be job or stage
        :param stage:
        :param amount:
        """
        # if non-exist then create
        if stage not in self.commit[src]:
            self.commit[src][stage] = 0
        # add
        self.commit[src][stage] += amount
        self.stage_commit[stage] += amount
        self.backward[stage].add(src)

    def pop(self, src):
        assert src in self.commit
        assert len(self.commit[src]) > 0

        stage = next(iter(self.commit[src]))
        # deduct
        self.commit[src][stage] -= 1
        self.stage_commit[stage] -= 1
        assert self.commit[src][stage] >= 0
        assert self.stage_commit[stage] >= 0
        # remove if amount is zero
        if self.commit[src][stage] == 0:
            del self.commit[src][stage]
            self.backward[stage].remove(src)

        return stage

    def add_job(self, job):
        self.commit[job] = OrderedDict()
        for stage in job.stages:
            self.commit[stage] = OrderedDict()
            self.stage_commit[stage] = 0
            self.backward[stage] = set()

    def remove_job(self, job):
        assert len(self.commit[job]) == 0
        del self.commit[job]
        for stage in job.stages:
            assert len(self.commit[stage]) == 0
            del self.commit[stage]

            for src in self.backward[stage]:
                del self.commit[src][stage]
            del self.backward[stage]
            del self.stage_commit[stage]

    def reset(self):
        self.commit = {None: OrderedDict()}
        self.stage_commit = {None: 0}
        self.backward = {None: set()}


def get_act2stage(jobs):
    """
    Get the translation from the action (an integer between [0, num_stages_in_all_jobs]) to the corresponding stage.
    """
    act2stage = utils.ReversibleMap()
    act = 0
    for job in jobs:
        for stage in job.stages:
            act2stage[act] = stage
            act += 1
    return act2stage


def get_frontier_acts(jobs):
    """
    Mapping all the frontier stages to the corresponding action numbers.
    """
    frontier_acts = []
    base = 0
    for job in jobs:
        for stage in job.frontier_stages:
            frontier_acts.append(base + stage.idx)        # TODO [bug]: should it be stage.idx?
        base += job.num_stages
    return frontier_acts
