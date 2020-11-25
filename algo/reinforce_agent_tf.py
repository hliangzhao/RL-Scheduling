"""
This module defines the REINFORCE agent.
This agent consists of a graph neural network and a policy network, and  is trained through REINFORCE algorithm.
Implemented with tensorflow 1.14.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from time import gmtime, strftime
from params import args
from agent import Agent
import utils


class GraphCNN:
    """
    This Graph Convolutional Neural Network is used to get embedding features of each stage (node)
    via parameterized message passing scheme.
    """
    def __init__(self, inputs, input_dim, hidden_dims, output_dim, max_depth, activate_fn, scope='gcn'):
        self.inputs = inputs
        self.input_dim = input_dim       # length of original feature x
        self.hidden_dims = hidden_dims   # dim of hidden layers of f and g
        self.output_dim = output_dim     # length of embedding feature e
        self.max_depth = max_depth       # TODO: num of jobs
        self.activate_fn = activate_fn
        self.scope = scope

        self.adj_mats = [tf.sparse_placeholder(tf.float32, [None, None]) for _ in range(self.max_depth)]
        self.masks = [tf.placeholder(tf.float32, [None, 1]) for _ in range(self.max_depth)]

        # h: x  -->  x'
        self.prep_weights, self.prep_bias = init(self.input_dim, self.hidden_dims, self.output_dim, self.scope)
        # f: x' -->  e
        self.proc_weights, self.proc_bias = init(self.output_dim, self.hidden_dims, self.output_dim, self.scope)
        # g: e  -->  e
        self.agg_weights, self.agg_bias = init(self.output_dim, self.hidden_dims, self.output_dim, self.scope)

        self.outputs = self.embedding()

    def embedding(self):
        """
        Embedding features are passing among stages (nodes) of each graph.
        The info is flowing from sink stages to source stages.
        """
        x = self.inputs
        for layer in range(len(self.prep_weights)):
            x = tf.matmul(x, self.prep_weights[layer]) + self.prep_bias[layer]
            x = self.activate_fn(x)

        for d in range(self.max_depth):
            y = x
            for layer in range(len(self.proc_weights)):
                y = tf.matmul(y, self.proc_weights[layer]) + self.proc_bias[layer]
                y = self.activate_fn(y)
            y = tf.sparse_tensor_dense_matmul(self.adj_mats[d], y)

            # aggregate children embedding features
            for layer in range(len(self.agg_weights)):
                y = tf.matmul(y, self.agg_weights[layer]) + self.agg_bias[layer]
                y = self.activate_fn(y)
            y *= self.masks[d]
            x += y
        return x


class GraphSNN:
    """
    The Graph Summarization Neural Network is used to get the DAG level and global level embedding features.
    """
    def __init__(self, inputs, input_dim, hidden_dims, output_dim, activate_fn, scope='gsn'):
        """
        Use stage (node) level summarization to obtain the DAG level summarization.
        Use DAG level summarization to obtain the global embedding summarization.
        """
        self.inputs = inputs
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activate_fn = activate_fn
        self.scope = scope

        self.levels = 2
        self.summary_mats = [tf.sparse_placeholder(tf.float32, [None, None]) for _ in range(self.levels)]
        self.dag_summ_weights, self.dag_summ_bias = init(self.input_dim, self.hidden_dims, self.output_dim, self.scope)
        self.global_summ_weights, self.global_summ_bias = init(self.output_dim, self.hidden_dims, self.output_dim, self.scope)

        self.summaries = self.summarize()

    def summarize(self):
        x = self.inputs
        summaries = []
        s = x
        # DAG level summary
        for layer in range(len(self.dag_summ_weights)):
            s = tf.matmul(s, self.dag_summ_weights[layer]) + self.dag_summ_bias[layer]
            s = self.activate_fn(s)
        s = tf.sparse_tensor_dense_matmul(self.summary_mats[0], s)
        summaries.append(s)

        # global level summary
        for layer in range(len(self.global_summ_weights)):
            s = tf.matmul(s, self.global_summ_weights[layer]) + self.global_summ_bias[layer]
            s = self.activate_fn(s)
        s = tf.sparse_tensor_dense_matmul(self.summary_mats[1], s)
        summaries.append(s)

        return summaries


def init(input_dim, hidden_dims, output_dim, scope):
    """
    Parameter initialization.
    """
    weights, bias = [], []
    cur_in_dim = input_dim

    for hid_dim in hidden_dims:
        weights.append(glorot_init(shape=[cur_in_dim, hid_dim], scope=scope))
        bias.append(zeros(shape=[hid_dim], scope=scope))
        cur_in_dim = hid_dim

    weights.append(glorot_init(shape=[cur_in_dim, output_dim], scope=scope))
    bias.append(zeros(shape=[output_dim], scope=scope))

    return weights, bias


def glorot_init(shape, dtype=tf.float32, scope='default'):
    """
    The initialization method proposed by Xavier Glorot & Yoshua Bengio.
    """
    with tf.variable_scope(scope):
        init_range = np.sqrt(6. / (shape[0] + shape[1]))
        return tf.Variable(tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=dtype))


class ReinforceAgent(Agent):
    def __init__(self, sess, stage_input_dim, job_input_dim, hidden_dims, output_dim, max_depth, executor_levels, activate_fn,
                 eps=1e-6, optimizer=tf.train.AdamOptimizer, scope='reinforce_agent'):
        super(ReinforceAgent, self).__init__()
        self.sess = sess
        self.stage_input_dim = stage_input_dim
        self.job_input_dim = job_input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.executor_levels = executor_levels
        self.max_depth = max_depth
        self.activate_fn = activate_fn
        self.eps = eps
        self.optimizer = optimizer
        self.scope = scope

        self.msg_passing = MsgPassing()
        # dim 0 = total_num_stages
        self.stage_inputs = tf.placeholder(tf.float32, [None, self.stage_input_dim])
        # dim 0 = total_num_jobs
        self.job_inputs = tf.placeholder(tf.float32, [None, self.job_input_dim])
        self.gcn = GraphCNN(
            self.stage_inputs, self.stage_input_dim, self.hidden_dims, self.output_dim,
            self.max_depth, self.activate_fn, self.scope
        )
        self.gsn = GraphSNN(
            tf.concat([self.stage_inputs, self.gcn.outputs], axis=1),
            self.stage_input_dim + self.output_dim, self.hidden_dims, self.output_dim,
            self.activate_fn, self.scope
        )
        # valid mask for stage action, of shape (batch_size, total_num_stages)
        self.stage_valid_mask = tf.placeholder(tf.float32, [None, None])
        # valid mask for executor limit on jobs, of shape (batch_size, num_jobs * num_exec_limits)
        self.job_valid_mask = tf.placeholder(tf.float32, [None, None])
        # map back the job summarization to each stage, of shape (total_num_stages, num_jobs)
        self.job_summ_backward_map = tf.placeholder(tf.float32, [None, None])

        # map gcn outputs and raw_inputs to to action probability distribution
        # stage_act_probs: of shape (batch_size, total_num_stages)
        # TODO: job_act_probs: should be of shape (batch_size, total_num_jobs, num_limits)?
        self.stage_act_probs, self.job_act_probs = self.actor_network(
            self.stage_inputs, self.gcn.outputs, self.job_inputs,
            self.gsn.summaries[0], self.gsn.summaries[1],
            self.stage_valid_mask, self.job_valid_mask,
            self.job_summ_backward_map, self.activate_fn
        )

        # get next stage
        # stage_acts is of shape (batch_size, 1)
        logits = tf.log(self.stage_act_probs)
        noise = tf.random_uniform(tf.shape(logits))   # TODO: why add noise?
        self.stage_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 1)

        # get next job
        # job_acts is of shape (batch_size, total_num_jobs, 1)
        logits = tf.log(self.job_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.job_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 2)

        # selected action of stage, a 0-1 vec of shape (batch_size, total_num_stages)
        self.stage_act_vec = tf.placeholder(tf.float32, [None, None])
        # selected action of job, a 0-1 vec of shape (batch_size, total_num_jobs, num_limits)
        self.job_act_vec = tf.placeholder(tf.float32, [None, None, None])

        # stage selected action probability
        self.selected_stage_prob = tf.reduce_sum(
            tf.multiply(self.stage_act_probs, self.stage_act_vec),
            reduction_indices=1,
            keep_dims=True
        )
        # job selected action probability
        self.selected_job_prob = tf.reduce_sum(
            tf.reduce_sum(tf.multiply(self.job_act_probs, self.job_act_vec), reduction_indices=2),
            reduction_indices=1,
            keep_dims=True
        )

        # advantage term from Monte Carlo or critic, of shape (batch_size, 1)
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # use entropy to promote exploration (decay over time)
        self.entropy_weight = tf.placeholder(tf.float32, ())

        # actor loss
        self.adv_loss = tf.reduce_sum(tf.multiply(
            tf.log(self.selected_stage_prob * self.selected_job_prob + self.eps),
            -self.adv
        ))

        # stage entropy
        self.stage_entropy = tf.reduce_sum(tf.multiply(
            self.stage_act_probs,
            tf.log(self.stage_act_probs + self.eps)
        ))

        # prob on each job
        self.prob_on_each_job = tf.reshape(
            tf.sparse_tensor_dense_matmul(self.gsn.summary_mats[0], tf.reshape(self.stage_act_probs, [-1, 1])),
            [tf.reshape(self.stage_act_probs)[0], -1]
        )

        # job entropy
        self.job_entropy = tf.reduce_sum(tf.multiply(
            self.prob_on_each_job,
            tf.reduce_sum(tf.multiply(self.job_act_probs, tf.log(self.job_act_probs + self.eps)), reduction_indices=2)
        ))

        # entropy loss
        self.entropy_loss = self.stage_entropy + self.job_entropy
        # normalize entropy over batch size
        self.entropy_loss /= tf.log(tf.cast(tf.shape(self.stage_act_probs)[1], tf.float32)) + tf.log(float(len(self.executor_levels)))

        # total loss
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # params
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope
        )

        # operations for setting network params
        self.input_params, self.set_params_op = self.define_params_op()

        # actor gradients
        self.act_gradients = tf.gradients(self.act_loss, self.params)
        # adaptive learning rate
        self.lr = tf.placeholder(tf.float32, shape=[])
        # optimizer
        self.act_opt = self.optimizer(self.lr).minimize(self.act_loss)
        # apply gradients
        self.apply_grads = self.optimizer(self.lr).apply_gradients(zip(self.act_gradients, self.params))
        # define network param saver and where to restore models
        self.saver = tf.train.Saver(max_to_keep=args.num_saved_models)
        self.sess.run(tf.global_variables_initializer())
        if args.saved_model is not None:
            self.saver.restore(self.sess, args.saved_model)

    def actor_network(self, stage_inputs, gcn_outputs, job_inputs, gsn_job_summary, gsn_global_summary,
                      stage_valid_mask, job_valid_mask, gsn_summ_backward_map, activate_fn):
        """
        This is the forwarding computation of the policy network.
        Part of content of this should be in __init__ when using torch.
        """
        batch_size = tf.shape(stage_valid_mask)[0]

        # reshape the raw input into batch format
        stage_inputs_reshape = tf.reshape(stage_inputs, [batch_size, -1, self.stage_input_dim])
        job_inputs_reshape = tf.reshape(job_inputs, [batch_size, -1, self.job_input_dim])
        gcn_outputs_reshape = tf.reshape(gcn_outputs, [batch_size, -1, self.output_dim])

        gsn_job_summary_reshape = tf.reshape(gsn_job_summary, [batch_size, -1, self.output_dim])
        gsn_summ_backward_map_extend = tf.tile(tf.expand_dims(gsn_summ_backward_map, axis=0), [batch_size, 1, 1])  # add a dim for batch and copy
        gsn_job_summ_extend = tf.matmul(gsn_summ_backward_map_extend, gsn_job_summary_reshape)

        gsn_global_summary_reshape = tf.reshape(gsn_global_summary, [batch_size, -1, self.output_dim])
        gsn_global_summary_extend_job = tf.tile(gsn_job_summary_reshape, [1, tf.shape(gsn_job_summary_reshape)[1], 1])
        gsn_global_summary_extend_stage = tf.tile(gsn_global_summary_reshape, [1, tf.shape(gsn_job_summ_extend)[1], 1])

        with tf.variable_scope(self.scope):
            # part 1: the probability distribution over stage selection
            merge_stage = tf.concat(
                [stage_inputs_reshape, gcn_outputs_reshape, gsn_job_summ_extend, gsn_global_summary_extend_stage],
                axis=2
            )
            stage_hidden0 = tf_layers.fully_connected(merge_stage, 32, activation_fn=activate_fn)
            stage_hidden1 = tf_layers.fully_connected(stage_hidden0, 16, activation_fn=activate_fn)
            stage_hidden2 = tf_layers.fully_connected(stage_hidden1, 8, activation_fn=activate_fn)
            stage_outputs = tf_layers.fully_connected(stage_hidden2, 1, activation_fn=None)
            stage_outputs = tf.reshape(stage_outputs, [batch_size, -1])    # (batch_size, total_num_stages)

            stage_valid_mask = (stage_valid_mask - 1) * 10000.    # to make those stages which cannot be chosen have very low prob
            stage_outputs = stage_outputs + stage_valid_mask
            stage_outputs = tf.nn.softmax(stage_outputs, dim=-1)

            # part 2: the probability distribution over executor limits
            merge_job = tf.concat(
                [job_inputs_reshape, gsn_job_summary_reshape, gsn_global_summary_extend_job],
                axis=2
            )
            expanded_state = expand_act_on_state(merge_job, [lvl / 50. for lvl in self.executor_levels])
            job_hidden0 = tf_layers.fully_connected(expanded_state, 32, activation_fn=activate_fn)
            job_hidden1 = tf_layers.fully_connected(job_hidden0, 16, activation_fn=activate_fn)
            job_hidden2 = tf_layers.fully_connected(job_hidden1, 8, activation_fn=activate_fn)
            job_outputs = tf_layers.fully_connected(job_hidden2, 1, activation_fn=None)
            job_outputs = tf.reshape(job_outputs, [batch_size, -1])  # (batch_size, num_jobs * num_exec_limits)

            job_valid_mask = (job_valid_mask - 1) * 10000.
            job_outputs = job_outputs + job_valid_mask
            # reshape to (batch_size, num_jobs, num_exec_limits)
            job_outputs = tf.reshape(job_outputs, [batch_size, -1, len(self.executor_levels)])
            job_outputs = tf.nn.softmax(job_outputs, dim=-1)

            return stage_outputs, job_outputs

    def define_params_op(self):
        input_params = []
        for param in self.params:
            input_params.append(tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def apply_gradients(self, gradients, lr):
        self.sess.run(
            self.apply_grads,
            feed_dict={
                i: d for i, d in zip(self.act_gradients + [self.lr], gradients + [lr])
            }
        )

    def gcn_forward(self, stage_inputs, summ_mats):
        return self.sess.run(
            [self.gsn.summaries],
            feed_dict={
                i: d for i, d in zip([self.stage_inputs] + self.gsn.summary_mats, [stage_inputs] + summ_mats)
            }
        )

    def get_gradients(self):
        pass

    def predict(self):
        pass

    def set_params(self, input_params):
        self.sess.run(
            self.set_params_op,
            feed_dict={
                i: d for i, d in zip(self.input_params, input_params)
            }
        )

    def translate_state(self, obs):
        pass

    def get_valid_masks(self):
        pass

    def invoke_model(self, obs):
        pass

    def get_action(self, obs):
        pass


class Logger:
    """
    The tensorflow logger.
    """
    def __init__(self, sess, var_list):
        self.sess = sess
        self.summary_vars = []
        for var in var_list:
            tf_var = tf.Variable(0.)
            tf.summary.scalar(var, tf_var)
            self.summary_vars.append(tf_var)
        self.summary_ops = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(args.result_folder + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    def log(self, ep, values):
        assert len(self.summary_vars) == len(values)
        feed_dict = {self.summary_vars[i]: values[i] for i in range*len(values)}
        summary_str = self.sess.run(self.summary_ops, feed_dict=feed_dict)
        self.writer.add_summary(summary=summary_str, global_step=ep)
        self.writer.flush()


class MsgPassing:
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
            self.msg_mats, self.msg_masks = MsgPassing.get_msg(jobs)
            self.job_summ_backward_map = MsgPassing.get_job_summ_backward_map(jobs)
            self.running_job_mat = self.get_running_job_mat(jobs)
            self.jobs = utils.OrderedSet(jobs)

        return self.msg_mats, self.msg_masks, self.job_summ_backward_map, self.running_job_mat, jobs_changed

    def reset(self):
        self.jobs = utils.OrderedSet()
        self.msg_mats = []
        self.msg_masks = []
        self.job_summ_backward_map = None
        self.running_job_mat = None

    @staticmethod
    def get_msg(jobs):
        msg_mats, msg_masks = [], []
        for job in jobs:
            msg_mat, msg_mask = MsgPassing.get_bottom_up_paths(job)
            msg_mats.append(msg_mat)
            msg_masks.append(msg_mask)
        if len(jobs) > 0:
            msg_mats = merge_sparse_mats(msg_mats, args.max_depth)
            msg_masks = merge_masks(msg_masks)
        return msg_mats, msg_masks

    @staticmethod
    def get_job_summ_backward_map(jobs):
        """
        Compute backward mapping from stage to job idx.
        """
        total_stages_num = int(np.sum([job.num_stages for job in jobs]))
        job_summ_backward_map = np.zeros([total_stages_num, len(jobs)])

        base = 0
        job_idx = 0
        for job in jobs:
            for stage in job.stages:
                job_summ_backward_map[base + stage.idx, job_idx] = 1
            base += job.num_stages
            job_idx += 1
        return job_summ_backward_map

    @staticmethod
    def get_running_job_mat(jobs):
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

    @staticmethod
    def get_bottom_up_paths(job):
        """
        The paths start from all leave nodes and end with frontier unfinished nodes (nodes whose parents are finished).
        """
        num_stages = job.num_stages
        msg_mats = []
        msg_masks = np.zeros([args.max_depth, num_stages])

        # get frontier stages
        frontiers = MsgPassing.get_init_frontier(job, args.max_depth)
        msg_level = {}
        for s in frontiers:
            msg_level[s] = 0

        # pass msg
        for d in range(args.max_depth):
            new_frontiers = set()
            parent_visited = set()
            for s in frontiers:
                for p in s.parent_stages:
                    if p not in parent_visited:
                        cur_level = 0
                        children_all_in_frontier = True
                        for c in p.child_stages:
                            if c not in frontiers:
                                children_all_in_frontier = False
                                break
                            if msg_level[c] > cur_level:
                                cur_level = msg_level[c]
                        if children_all_in_frontier:
                            if p not in msg_level or cur_level + 1 > msg_level[p]:
                                new_frontiers.add(p)
                                msg_level[p] = cur_level + 1
                        parent_visited.add(p)
            if len(new_frontiers) == 0:
                break
            sp_mat = SparseMat(dtype=np.float32, shape=(num_stages, num_stages))
            for s in new_frontiers:
                for c in s.child_stages:
                    sp_mat.add(row=s.idx, col=c.idx, data=1)
                msg_masks[d, s.idx] = 1
            msg_mats.append(sp_mat)

            # there might be residual stages that can directly pass msg to its parents (e.g., tpc-h 17, stage 0, 2, 4)
            # in thi case, it needs twp msg passing steps
            for s in frontiers:
                parents_all_in_frontier = True
                for p in s.parent_stages:
                    if p not in msg_level:
                        parents_all_in_frontier = False
                        break
                if not parents_all_in_frontier:
                    new_frontiers.add(s)

            frontiers = new_frontiers
        for _ in range(d, args.max_depth):   # TODO: replace d with args.max_depth - 1
            msg_mats.append(SparseMat(dtype=np.float32, shape=(num_stages, num_stages)))
        return msg_mats, msg_masks

    @staticmethod
    def get_init_frontier(job, depth):
        """
        Get the initial set of frontier nodes.
        """
        srcs = set(job.stages)
        for d in range(depth):
            new_srcs = set()
            for s in srcs:
                if len(s.child_stages) == 0:
                    new_srcs.add(s)
                else:
                    new_srcs.update(s.child_stages)
            srcs = new_srcs
        return srcs


def get_unfinished_stages_summ_mat(jobs):
    """
    Add a connection from the unfinished stages to the summarized node.
    """
    total_num_stages = np.sum([job.num_stages for job in jobs])
    summ_row_idx, summ_col_idx, summ_data = [[]] * 3
    summ_shape = (len(jobs), total_num_stages)

    base = 0
    job_idx = 0
    for job in jobs:
        for stage in job.stages:
            if not stage.all_tasks_done:
                summ_row_idx.append(job_idx)
                summ_col_idx.append(base + stage.idx)
                summ_data.append(1)

        base += job.num_stages
        job_idx += 1

    return tf.SparseTensorValue(
        indices=np.mat([summ_row_idx, summ_col_idx]).transpose(),
        values=summ_data,
        dense_shape=summ_shape
    )


class SparseMat:
    """
    Define ths sparse matrix and operations on it.
    """
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.row, self.col, self.data = [[]] * 3

    def add(self, row, col, data):
        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def get_row(self):
        return np.array(self.row)

    def get_col(self):
        return np.array(self.col)

    def get_data(self):
        return np.array(self.data)

    def to_tf_sparse_mat(self):
        indices = np.mat([self.row, self.col]).transpose()
        return tf.SparseTensorValue(indices, self.data, self.shape)


def merge_sparse_mats(sparse_mats, depth):
    """
    Merge multiple sparse matrices (which have the same shape) to a global sparse matrix on its diagonal.
    The merge operation is taken on each depth separately.
    e.g., for the first depth of three matrices, from
    [0, 1, 0]    [0, 1, 0]    [0, 0, 1]
    [1, 0, 0]    [0, 0, 1]    [0, 1, 0]
    [0, 0, 1]    [1, 0, 0]    [0, 1, 0]

    we have

    [0, 1, 0]
    [1, 0, 0]   ..  ..    ..  ..
    [0, 0, 1]
              [0, 1, 0]
     ..  ..   [0, 0, 1]   ..  ..
              [1, 0, 0]
                        [0, 0, 1]
     ..  ..    ..  ..   [0, 1, 0]
                        [0, 1, 0]
    :param sparse_mats: a list of sparse matrices. Each matrix has the 3rd dim, depth
    :param depth: the depth of each sparse matrix, which is orthogonal to the planar operation above
    :return: a list of merged global sparse matrix. The list length is the depth
    """
    global_sp_mat = []
    for d in range(depth):
        row_idx, col_idx, data = [[]] * 3
        shape = 0
        base = 0
        for mat in sparse_mats:
            row_idx.append(mat[d].get_row() + base)
            col_idx.append(mat[d].get_col() + base)
            data.append(mat[d].get_data())
            shape += mat[d].shape[0]
            base += mat[d].shape[0]
        row_idx = np.hstack(row_idx)
        col_idx = np.hstack(col_idx)
        data = np.hstack(data)

        indices = np.mat([row_idx, col_idx]).transpose()
        global_sp_mat.append(tf.SparseTensorValue(indices, data, (shape, shape)))
    return global_sp_mat


def expand_sparse_mats(sparse_mats, exp_step):
    """
    Make a stack of the same sparse matrix to a global one on its diagonal. The expand operation is taken on each depth separately.
    Here the depth is the length of sparse_mats.
    e.g.,
    On one depth of sparse_mats, we have
    [0, 1, 0]    [0, 1, 0]
    [1, 0, 0]    [1, 0, 0]  ..  ..   ..  ..
    [0, 0, 1]    [0, 0, 1]
                          [0, 1, 0]
              to  ..  ..  [1, 0, 0]  ..  ..
                          [0, 0, 1]
                                   [0, 1, 0]
                  ..  ..   ..  ..  [1, 0, 0]
                                   [0, 0, 1]

    where exp_step is 3.
    :param sparse_mats: of type tf.SparseTensorValue
    :param exp_step: expand times
    :return: a list of expanded global sparse matrix. The list length is the same as sparse_mats
    """
    global_sp_mat = []
    depth = len(sparse_mats)
    for d in range(depth):
        row_idx, col_idx, data = [[]] * 3
        shape = 0
        base = 0
        for i in range(exp_step):
            indices = sparse_mats[d].indices.transpose()
            row_idx.append(np.squeeze(np.asarray(indices[0, :]) + base))
            col_idx.append(np.squeeze(np.asarray(indices[1, :]) + base))
            data.append(sparse_mats[d].values)
            shape += sparse_mats[d].dense_shape[0]
            base += sparse_mats[d].dense_shape[0]
        row_idx = np.hstack(row_idx)
        col_idx = np.hstack(col_idx)
        data = np.hstack(data)
        indices = np.mat([row_idx, col_idx]).transpose()
        global_sp_mat.append(tf.SparseTensorValue(indices, data, (shape, shape)))
    return global_sp_mat


def merge_and_extend_sparse_mats(sparse_mats):
    """
    Transform multiple sparse matrices (which have the same shape) to a global sparse matrix on its diagonal.
    e.g.,
    list of
    [1, 0, 1, 1] [0, 0, 0, 1]
    [1, 1, 1, 1] [0, 1, 1, 1]
    [0, 0, 1, 1] [1, 1, 1, 1]

    to

    [1, 0, 1, 1]
    [1, 1, 1, 1]    ..  ..
    [0, 0, 1, 1]
                 [0, 0, 0, 1]
       ..  ..    [0, 1, 1, 1]
                 [1, 1, 1, 1]
    Compared with merge_sparse_mats(), the input sparse_mats here do not have the 3rd dim.
    :param sparse_mats: a batch of sparse matrices. Each matrix has only 2 dims (not has depth)
    :return: a transformed global matrix
    """
    batch_size = len(sparse_mats)
    row_idx, col_idx, data = [[]] * 3
    shape = (sparse_mats[0].dense_shape[0] * batch_size, sparse_mats[0].dense_shape[1] * batch_size)

    row_base, col_base = 0, 0
    for b in range(batch_size):
        indices = sparse_mats[b].indices.transpose()
        row_idx.append(np.squeeze(np.asarray(indices[0, :]) + row_base))
        col_idx.append(np.squeeze(np.asarray(indices[1, :]) + col_base))
        data.append(sparse_mats[b].values)
        row_base += sparse_mats[b].dense_shape[0]
        col_base += sparse_mats[b].dense_shape[1]

    row_idx = np.hstack(row_idx)
    col_idx = np.hstack(col_idx)
    data = np.hstack(data)
    indices = np.mat([row_idx, col_idx]).transpose()
    return tf.SparseTensorValue(indices, data, shape)


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


def expand_act_on_state(state, sub_acts):
    batch_size = tf.shape(state)[0]
    num_stages = tf.shape(state)[1]
    num_features = state.shape[2].value
    expand_dim = len(sub_acts)

    # replicate the state
    state = tf.tile(state, [1, 1, expand_dim])
    state = tf.reshape(state, [batch_size, num_stages * expand_dim, num_features])

    sub_acts = tf.constant(sub_acts, dtype=tf.float32)
    sub_acts = tf.reshape(sub_acts, [1, 1, expand_dim])
    sub_acts = tf.tile(sub_acts, [1, 1, num_stages])
    sub_acts = tf.reshape(sub_acts, [1, num_stages * expand_dim, 1])
    sub_acts = tf.tile(sub_acts, [batch_size, 1, 1])    # now the first tow dim of sub_acts are the same as state

    # concatenate
    concat_state = tf.concat([state, sub_acts], axis=2)   # dim2 = num_features + 1
    return concat_state


def leaky_relu(features, alpha=0.3, name=None):
    """
    Implement the leaky RELU activate function.
    f(x) = x if x > 0 else alpha * x.
    """
    with ops.name_scope(name, 'LeakyRELU', [features, alpha]):
        features = ops.convert_to_tensor(features, name='features')
        alpha = ops.convert_to_tensor(alpha, name='alpha')
        return math_ops.maximum(alpha * features, features)


def masked_outer_product(a, b, mask):
    """

    :param a: of shape (batch_size, num_stages)
    :param b: of shape (batch_size, num_exec_limit * num_jobs)
    :param mask: TODO: shape?
    :return:
    """
    batch_size, num_stages = tf.shape(a)
    num_limits = tf.shape(b)[1]
    a = tf.reshape(a, [batch_size, num_stages, 1])
    b = tf.reshape(b, [batch_size, 1, num_limits])
    out_product = tf.reshape(a * b, [batch_size, -1])   # '-1' = num_stages * num_exec_limit

    out_product = tf.transpose(out_product)
    out_product = tf.boolean_mask(out_product, mask)
    out_product = tf.transpose(out_product)
    return out_product


def ones(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        return tf.Variable(tf.ones(shape, dtype=dtype))


def zeros(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        return tf.Variable(tf.zeros(shape, dtype=dtype))
