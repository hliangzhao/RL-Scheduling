"""
This module defines the agent of our algorithm.
This agent consists of a graph neural network and a policy network, and  is trained through REINFORCE algorithm.
Implemented with tensorflow 1.14.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from time import gmtime, strftime
from params import args
from agent import Agent


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
        pass


class Logger:
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


class SparseMat:
    """
    Define ths sparse matrix on operations on it.
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
    e.g.,
    [0, 1, 0]    [0, 1, 0]    [0, 0, 1]
    [1, 0, 0]    [0, 0, 1]    [0, 1, 0]
    [0, 0, 1]    [1, 0, 0]    [0, 1, 0]

    to

    [0, 1, 0]
    [1, 0, 0]   ..  ..    ..  ..
    [0, 0, 1]
              [0, 1, 0]
     ..  ..   [0, 0, 1]   ..  ..
              [1, 0, 0]
                        [0, 0, 1]
     ..  ..    ..  ..   [0, 1, 0]
                        [0, 1, 0]
    where ".." are all zeros. The procedure is repeated in depth times.
    :param sparse_mats:
    :param depth: the depth of each sparse matrix, which is orthogonal to the planar operation above
    :return:
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


def expand_sparse_mats(sparse_mat, exp_step):
    """
    Make a stack of the same sparse matrix to a global one on its diagonal.
    e.g.,
    on one depth of sparse_mat,
    [0, 1, 0]    [0, 1, 0]
    [1, 0, 0]    [1, 0, 0]  ..  ..   ..  ..
    [0, 0, 1]    [0, 0, 1]
                          [0, 1, 0]
              to  ..  ..  [1, 0, 0]  ..  ..
                          [0, 0, 1]
                                   [0, 1, 0]
                  ..  ..   ..  ..  [1, 0, 0]
                                   [0, 0, 1]

    where ".." are all zeros, exp_step is 3. The procedure is repeated in len(sparse_mat) times.
    :param sparse_mat: of type tf.SparseTensorValue
    :param exp_step:
    :return:
    """
    global_sp_mat = []
    depth = len(sparse_mat)
    for d in range(depth):
        row_idx, col_idx, data = [[]] * 3
        shape = 0
        base = 0
        for i in range(exp_step):
            indices = sparse_mat[d].indices.transpose()
            row_idx.append(np.squeeze(np.asarray(indices[0, :]) + base))
            col_idx.append(np.squeeze(np.asarray(indices[1, :]) + base))
            data.append(sparse_mat[d].values)
            shape += sparse_mat[d].dense_shape[0]
            base += sparse_mat[d].dense_shape[0]
        row_idx = np.hstack(row_idx)
        col_idx = np.hstack(col_idx)
        data = np.hstack(data)
        indices = np.mat([row_idx, col_idx]).transpose()
        global_sp_mat.append(tf.SparseTensorValue(indices, data, (shape, shape)))
    return global_sp_mat


def merge_and_extend_sparse_mats(sparse_mats):
    """
    TODO: what is the difference between this and the first func?
    :param sparse_mats:
    :return:
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
