"""
This module defines the Graph Neural Network, which is the first part of the agent.
    - GraphCNN is used to get embedding features of all stages.
    - GraphSNN is used to get the job level and global level embedding summarizations.
"""
import tensorflow as tf
import numpy as np


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
        self.max_depth = max_depth       # maximum depth of root-leaf message passing
        self.activate_fn = activate_fn
        self.scope = scope

        # args.max_depth tensors
        self.adj_mats = [tf.sparse_placeholder(tf.float32, [None, None]) for _ in range(self.max_depth)]
        self.masks = [tf.placeholder(tf.float32, [None, 1]) for _ in range(self.max_depth)]

        # h: x  -->  x'
        self.prep_weights, self.prep_bias = init(self.input_dim, self.hidden_dims, self.output_dim, self.scope)
        # f: x' -->  e
        self.proc_weights, self.proc_bias = init(self.output_dim, self.hidden_dims, self.output_dim, self.scope)
        # g: e  -->  e
        self.agg_weights, self.agg_bias = init(self.output_dim, self.hidden_dims, self.output_dim, self.scope)

        self.outputs = self.forward()

    def forward(self):
        """
        Get embedding features of each node for all jobs simultaneously and use masks to filter non-runnable nodes.
        """
        # message passing among nodes
        # the information is flowing from leaves to roots
        x = self.inputs
        # raise x into higher dimension
        for layer in range(len(self.prep_weights)):
            x = tf.matmul(x, self.prep_weights[layer])
            x += self.prep_bias[layer]
            x = self.activate_fn(x)

        for d in range(self.max_depth):
            # work flow: index_select -> f -> masked assemble via adj_mat -> g
            y = x
            # process the features on the nodes
            for layer in range(len(self.proc_weights)):
                y = tf.matmul(y, self.proc_weights[layer])
                y += self.proc_bias[layer]
                y = self.activate_fn(y)

            # message passing
            y = tf.sparse_tensor_dense_matmul(self.adj_mats[d], y)

            # aggregate child features
            for layer in range(len(self.agg_weights)):
                y = tf.matmul(y, self.agg_weights[layer])
                y += self.agg_bias[layer]
                y = self.activate_fn(y)

            # remove the artifact from the bias term in g
            y = y * self.masks[d]
            # assemble neighboring information
            x = x + y
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

        self.summ_levels = 2
        self.summ_mats = [tf.sparse_placeholder(tf.float32, [None, None]) for _ in range(self.summ_levels)]
        self.job_summ_weights, self.job_summ_bias = init(self.input_dim, self.hidden_dims, self.output_dim, self.scope)
        self.global_summ_weights, self.global_summ_bias = init(self.output_dim, self.hidden_dims, self.output_dim, self.scope)

        self.summaries = self.forward()

    def forward(self):
        """
        Get the job level and global level summary.
        """
        # summarize information in each hierarchy
        x = self.inputs

        summaries = []
        # DAG level summary
        s = x
        for i in range(len(self.job_summ_weights)):
            s = tf.matmul(s, self.job_summ_weights[i])
            s += self.job_summ_bias[i]
            s = self.activate_fn(s)

        s = tf.sparse_tensor_dense_matmul(self.summ_mats[0], s)
        summaries.append(s)

        # global level summary
        for i in range(len(self.global_summ_weights)):
            s = tf.matmul(s, self.global_summ_weights[i])
            s += self.global_summ_bias[i]
            s = self.activate_fn(s)

        s = tf.sparse_tensor_dense_matmul(self.summ_mats[1], s)
        summaries.append(s)
        return summaries


def init(input_dim, hidden_dims, output_dim, scope):
    """
    GNN parameter initialization.
    :return list of weights (tf tensors) and list of biases (tf tensors)
    """
    weights, bias = [], []
    cur_in_dim = input_dim

    # hidden layers param init
    for hid_dim in hidden_dims:
        weights.append(glorot_init(shape=[cur_in_dim, hid_dim], scope=scope))
        bias.append(zeros(shape=[hid_dim], scope=scope))
        cur_in_dim = hid_dim

    # output layer param init
    weights.append(glorot_init(shape=[cur_in_dim, output_dim], scope=scope))
    bias.append(zeros(shape=[output_dim], scope=scope))

    return weights, bias


def glorot_init(shape, dtype=tf.float32, scope='default'):
    """
    The initialization method proposed by Xavier Glorot & Yoshua Bengio in AISTATS '10.
    """
    with tf.variable_scope(scope):
        init_range = np.sqrt(6. / (shape[0] + shape[1]))
        return tf.Variable(tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=dtype))


def zeros(shape, dtype=tf.float32, scope='default'):
    """
    Used to init bias.
    """
    with tf.variable_scope(scope):
        return tf.Variable(tf.zeros(shape, dtype=dtype))
