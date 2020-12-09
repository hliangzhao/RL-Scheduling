"""
This module defines the executors, free executors, and moving executors.
"""
import utils


class Executor:
    """
    This class defines the executor. It is the basic operator for jobs in Spark.
    """
    def __init__(self, idx):
        self.idx = idx
        self.task, self.stage, self.job = None, None, None

    def detach_stage(self):
        """
        Detach this executor from the bound stage.
        """
        if (self.stage is not None) and (self in self.stage.executors):
            self.stage.executors.remove(self)
        self.stage, self.task = None, None

    def detach_job(self):
        """
        Detach this executor from the bound job and corresponding stage (of this job).
        """
        if (self.job is not None) and (self in self.job.executors):
            self.job.executors.remove(self)
        self.job = None
        self.detach_stage()

    def reset(self):
        self.task, self.stage, self.job = None, None, None


class FreeExecutors:
    """
    This class defines a dict, where the key is the job, the value is the set of bound executors.
    What the key-value pairs mean:
        - These bound executors (value) are 'free' currently (and temporarily) but are decided to be dispatched to
        the corresponding job (key).
        - When the key is None, the value means the free executors that do not bind to any job, which means these jobs
        are free to be dispatched and scheduled to any stage of jobs exist in system now.
    """
    def __init__(self, executors):
        self.free_executors = {None: utils.OrderedSet()}
        for e in executors:
            self.free_executors[None].add(e)

    def __getitem__(self, job):
        """
        Get the dispatched executors of the given job.
        """
        return self.free_executors[job]

    def contain_executor(self, job, executor):
        """
        Judge whether the given executor is dispatched to the given job.
        """
        if executor in self.free_executors[job]:
            return True
        return False

    def pop(self, job):
        """
        Pop the first executor of the given job.
        """
        executor = next(iter(self.free_executors[job]))
        self.free_executors[job].remove(executor)
        return executor

    def add(self, job, executor):
        """
        If the given job is None, 'add' means that adding the executor to free exec pool.
        Thus we need to detach every job of this exec.
        """
        if job is None:
            # the exec is available to every job
            executor.detach_job()
        else:
            # the exec is available to this job, detach the old stage it bound to
            executor.detach_stage()
        self.free_executors[job].add(executor)

    def remove(self, executor):
        """
        Remove the given executor from its bound job.
        """
        self.free_executors[executor.job].remove(executor)

    def add_job(self, job):
        self.free_executors[job] = utils.OrderedSet()

    def remove_job(self, job):
        """
        Release the given job's executors and put them back to the free executor pool.
        """
        for executor in self.free_executors[job]:
            executor.detach_job()
            self.free_executors[None].add(executor)
        del self.free_executors[job]

    def reset(self, executors):
        self.free_executors = {None: utils.OrderedSet()}
        for e in executors:
            self.free_executors[None].add(e)


class MovingExecutors:
    """
    This class **temporarily** stores the next-to-go stage of some executors which are not free at that time.
    We need it because we do not want to invoke the agent too many times (which affects the performance obviously).
    """
    def __init__(self):
        """
        self.moving_executors: which job's which stage this executor is going to be moved to
        self.stage_track: if the executor is going to be moved to the stage, record it
        """
        self.moving_executors = {}     # {executor: stage}
        self.stage_track = {}          # {stage: (set of executors waiting for being dispatched to this stage)}

    def __contains__(self, executor):
        return executor in self.moving_executors

    def __getitem__(self, executor):
        """
        Get the stage which the exec is going to be moved to.
        """
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
            # this job is complete by the time the executor arrives
            stage = None
        return stage

    def count(self, stage):
        """
        Get the num fo executors waiting to be scheduled to the given stage.
        This var is used to calculate how many new execs to be dispatched to it.
        """
        return len(self.stage_track[stage])

    def add_job(self, job):
        for stage in job.stages:
            self.stage_track[stage] = set()

    def remove_job(self, job):
        for stage in job.stages:
            for executor in self.stage_track[stage]:
                del self.moving_executors[executor]
            del self.stage_track[stage]

    def reset(self):
        self.moving_executors = {}
        self.stage_track = {}
