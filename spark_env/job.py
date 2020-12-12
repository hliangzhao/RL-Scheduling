"""
This module defines the object to be scheduled, i.e. job.
"""
import numpy as np
# import networkx as nx
import utils
from params import args


class Job:
    """
    A job is modeled as a DAG, where nodes are stages (DAG's nodes), edges are data shuffle.
    Notice that each job ends with a single final stage in Spark. If not, you can add a final
    virtual stage with zero computation cost.
    """
    def __init__(self, stages, adj_mat, name):
        # assert len(stages) == adj_mat.shape[0] == adj_mat.shape[1]
        self.name = name
        self.stages = stages
        self.adj_mat = adj_mat
        self.num_stages = len(stages)
        self.num_finished_stages = 0

        self.executors = utils.OrderedSet()
        # assert self.is_dag(self.num_stages, self.adj_mat)

        self.frontier_stages = utils.OrderedSet()
        for stage in self.stages:
            if stage.is_runnable():
                self.frontier_stages.add(stage)

        for stage in self.stages:
            stage.job = self

        self.arrived = False
        self.finished = False
        self.start_time = None         # job.start_time is the arrival time of this job, different from task.start_time
        self.finish_time = np.inf

        self.executor2interval = self.get_executor_interval_map()

    @staticmethod
    def get_executor_interval_map():
        """
        Map the executor to the corresponding interval. The left and right number of each interval belongs to
        args.executor_data_point.
        The reason we need this is explained in Stage.__init__().
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

    def get_duration(self):
        """
        ====================== not used ======================
        Get the total remaining execution time of this job.
        """
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
        self.finished = False
        self.finish_time = np.inf

    def update_frontier_stages(self, stage):
        changed = False
        for child in stage.child_stages:
            # [bug]: what self.frontier_stages stores is child stages, not their idx!
            if child.is_runnable():
                if child not in self.frontier_stages:
                    self.frontier_stages.add(child)
                    changed = True
        return changed

    # @staticmethod
    # def is_dag(num_stages, adj_mat):
    #     """
    #     Judge a job (represented by adj_mat) is a DAG or not.
    #     """
    #     graph = nx.DiGraph()
    #     graph.add_nodes_from(range(num_stages))
    #     for i in range(num_stages):
    #         for j in range(num_stages):
    #             if adj_mat[i, j] == 1:
    #                 graph.add_edge(i, j)
    #     return nx.is_directed_acyclic_graph(graph)
