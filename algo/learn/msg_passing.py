"""
This module defines the message passing path, which is used for graph embedding.
"""
import utils
from params import args
from algo.learn.sparse_op import *


class MsgPassing:
    """
    Return vars' shape:
    - msg_mats: [depth tf.SparseMat], each of shape (total_num_stages, total_num_stages)
    - msg_masks: [depth tf.SparseMat], each of shape (total_num_stages, depth)
    - job_summ_backward_map: a np.mat of shape (total_num_stages, num_jobs)
    - running_job_mat: a tf.SparseMat of shape (1, num_jobs)
    """
    def __init__(self):
        """
        msg_mat records the parent-children relation in each message passing step.
        msg_masks is the set of stages (nodes) doing message passing at each step.
        """
        self.jobs = utils.OrderedSet()
        self.msg_mats = []
        self.msg_masks = []
        self.job_summ_backward_map = None
        self.running_job_mat = None

    def get_msg_path(self, jobs):
        """
        Check whether the set of jobs changes. If changed, compute the message passing path.
        """
        if len(self.jobs) != len(jobs):
            jobs_changed = True
        else:
            jobs_changed = not(all(job_i is job_j for (job_i, job_j) in zip(self.jobs, jobs)))

        if jobs_changed:
            self.msg_mats, self.msg_masks = self.get_msg(jobs)
            self.job_summ_backward_map = self.get_job_summ_backward_map(jobs)
            self.running_job_mat = self.get_running_job_mat(jobs)
            self.jobs = utils.OrderedSet(jobs)

        return self.msg_mats, self.msg_masks, self.job_summ_backward_map, self.running_job_mat, jobs_changed

    def reset(self):
        self.jobs = utils.OrderedSet()
        self.msg_mats = []
        self.msg_masks = []
        self.job_summ_backward_map = None
        self.running_job_mat = None

    def get_msg(self, jobs):
        """
        :return: (1) msg_mats, a list of depth tf.SparseMat, each of shape (total_num_stages, total_num_stages), and
                 (2) msg_masks, a list of depth tf.SparseMat, each of shape (total_num_stages, depth)
        """
        msg_mats, msg_masks = [], []
        for job in jobs:
            msg_mat, msg_mask = self.get_bottom_up_paths(job)
            msg_mats.append(msg_mat)
            msg_masks.append(msg_mask)
        if len(jobs) > 0:         # TODO: I think it should be 1
            msg_mats = merge_sparse_mats(msg_mats, args.max_depth)
            msg_masks = merge_masks(msg_masks)
        return msg_mats, msg_masks

    @staticmethod
    def get_job_summ_backward_map(jobs):
        """
        Masks indicate whether a stage with global index belongs to a job.
        :return: a np.mat of shape (total_num_stages, num_jobs)
        """
        total_num_stages = int(np.sum([job.num_stages for job in jobs]))
        job_summ_backward_map = np.zeros([total_num_stages, len(jobs)])

        base = 0
        j_idx = 0
        for job in jobs:
            for stage in job.stages:
                job_summ_backward_map[base + stage.idx, j_idx] = 1
            base += job.num_stages
            j_idx += 1

        return job_summ_backward_map

    @staticmethod
    def get_running_job_mat(jobs):
        """
        Masks indicate whether each job is running. A job is running iff it is unfinished.
        Obviously, the input jobs only include currently arrived jobs.
        :return: a tf.SparseMat of shape (1, num_jobs)
        """
        running_job_row_idx, running_job_col_idx = [], []
        running_job_data = []
        running_job_shape = (1, len(jobs))

        job_idx = 0
        for job in jobs:
            if not job.finished:   # get unfinished jobs' summary
                running_job_row_idx.append(0)
                running_job_col_idx.append(job_idx)
                running_job_data.append(1)
            job_idx += 1

        return tf.SparseTensorValue(
            indices=np.mat([running_job_row_idx, running_job_col_idx]).transpose(),
            values=running_job_data,
            dense_shape=running_job_shape
        )

    def get_bottom_up_paths(self, job):
        """
        The paths start from all leave stages and end with frontier unfinished stages (stages whose parents are finished).
        """
        num_stages = job.num_stages
        msg_mats = []
        msg_masks = np.zeros([args.max_depth, num_stages])

        # get frontier stages
        frontier = self.get_init_frontier(job, args.max_depth)
        msg_level = {}
        for s in frontier:
            msg_level[s] = 0

        # pass msg
        for depth in range(args.max_depth):
            new_frontier = set()
            parent_visited = set()  # save some computation
            for s in frontier:
                for parent in s.parent_stages:
                    if parent not in parent_visited:
                        cur_level = 0
                        children_all_in_frontier = True
                        for child in parent.child_stages:
                            if child not in frontier:
                                children_all_in_frontier = False
                                break
                            if msg_level[child] > cur_level:
                                cur_level = msg_level[child]
                        # children all ready
                        if children_all_in_frontier:
                            if parent not in msg_level or cur_level + 1 > msg_level[parent]:
                                # parent has deeper message passed
                                new_frontier.add(parent)
                                msg_level[parent] = cur_level + 1
                        # mark parent as visited
                        parent_visited.add(parent)

            if len(new_frontier) == 0:
                break
            sp_mat = SparseMat(dtype=np.float32, shape=(num_stages, num_stages))
            for s in new_frontier:
                for c in s.child_stages:
                    sp_mat.add(row=s.idx, col=c.idx, data=1)
                msg_masks[depth, s.idx] = 1
            msg_mats.append(sp_mat)

            # there might be residual stages that can directly pass msg to its parents (e.g., tpc-h 17, stage 0, 2, 4)
            # in this case, it needs twp msg passing steps
            for s in frontier:
                parents_all_in_frontier = True
                for p in s.parent_stages:
                    if p not in msg_level:
                        parents_all_in_frontier = False
                        break
                if not parents_all_in_frontier:
                    new_frontier.add(s)

            frontier = new_frontier
        for _ in range(depth, args.max_depth):
            msg_mats.append(SparseMat(dtype=np.float32, shape=(num_stages, num_stages)))
        return msg_mats, msg_masks

    @staticmethod
    def get_init_frontier(job, depth):
        """
        Get all stages of given job.
        """
        srcs = set(job.stages)
        for d in range(depth):           # TODO: why repeating depth times???
            new_srcs = set()
            for s in srcs:
                if len(s.child_stages) == 0:
                    new_srcs.add(s)
                else:
                    new_srcs.update(s.child_stages)
            srcs = new_srcs
        return srcs


def merge_masks(masks):
    """
    Merge masks (matrices).
    e.g.,
    [0, 1, 0]  [0, 1]  [0, 0, 0, 1]
    [0, 0, 1]  [1, 0]  [1, 0, 0, 0]
    [1, 0, 0]  [0, 0]  [0, 1, 1, 0]

    to

    a list of
    [0, 1, 0, 0, 1, 0, 0, 0, 1]^T,
    [0, 0, 1, 1, 0, 1, 0, 0, 0]^T,
    [1, 0, 0, 0, 0, 0, 1, 1, 0]^T,
    where the depth is 3.
    """
    merged_masks = []
    for d in range(args.max_depth):
        merged_mask = []
        for mask in masks:
            merged_mask.append(mask[d:d+1, :].transpose())
        if len(merged_mask) > 0:
            merged_mask = np.vstack(merged_mask)
        merged_masks.append(merged_mask)
    return merged_masks
