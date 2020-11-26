"""
This module defines the REINFORCE agent.
This agent consists of a graph neural network and a policy network, and is trained through REINFORCE algorithm.
Implemented with tensorflow 1.14.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
import numpy as np
import bisect
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from params import args
from agent import Agent
from env import Job, Stage
from algo.tf import assist, graph_nn
import utils


class ReinforceAgent(Agent):
    def __init__(self, sess, stage_input_dim, job_input_dim, hidden_dims, output_dim, max_depth, executor_levels, activate_fn,
                 eps=1e-6, optimizer=tf.train.AdamOptimizer, scope='reinforce_agent'):
        """
        Initialization the agent. In this init, we define the basic parameters such as stage and job raw feature dim.
        We also define the computation graph for tf vars.
        :param sess:
        :param stage_input_dim:
        :param job_input_dim:
        :param hidden_dims:
        :param output_dim:
        :param max_depth:
        :param executor_levels:
        :param activate_fn:
        :param eps:
        :param optimizer:
        :param scope:
        """
        super(ReinforceAgent, self).__init__()
        self.sess = sess
        self.stage_input_dim = stage_input_dim     # TODO; is fixed as 5
        self.job_input_dim = job_input_dim         # TODO: is fixed as 3
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
        self.stage_inputs = tf.placeholder(tf.float32, [None, self.stage_input_dim])           # raw input feature for stages
        # dim 0 = total_num_jobs
        self.job_inputs = tf.placeholder(tf.float32, [None, self.job_input_dim])               # raw input feature for jobs
        self.gcn = graph_nn.GraphCNN(
            self.stage_inputs, self.stage_input_dim, self.hidden_dims, self.output_dim,
            self.max_depth, self.activate_fn, self.scope
        )
        self.gsn = graph_nn.GraphSNN(
            tf.concat([self.stage_inputs, self.gcn.outputs], axis=1),
            self.stage_input_dim + self.output_dim, self.hidden_dims, self.output_dim,
            self.activate_fn, self.scope
        )
        # valid mask for stage action, of shape (batch_size, total_num_stages)
        self.stage_valid_mask = tf.placeholder(tf.float32, [None, None])                       # raw input to indicate stage validation
        # valid mask for executor limit on jobs, of shape (batch_size, num_jobs * num_exec_limits)
        self.job_valid_mask = tf.placeholder(tf.float32, [None, None])                         # raw input to indicate job validation
        # map back the job summarization to each stage, of shape (total_num_stages, num_jobs)
        self.job_summ_backward_map = tf.placeholder(tf.float32, [None, None])                  # raw input to indicate the job each stage belongs to

        # ======= the following part defines the forwarding computation graph =======
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

        # use entropy to promote exploration (decay over tm)
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

        # ======== claim params, gradients, optimizer, and model saver ========
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

        # ====== param init ======
        self.sess.run(tf.global_variables_initializer())
        if args.saved_model is not None:
            self.saver.restore(self.sess, args.saved_model)

    # the following 2 funcs are called in init
    def actor_network(self, stage_inputs, gcn_outputs, job_inputs, gsn_job_summary, gsn_global_summary,
                      stage_valid_mask, job_valid_mask, gsn_summ_backward_map, activate_fn):
        """
        This is the forwarding computation of the policy network.
        Called when init.
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
        """
        Called when init.
        """
        input_params = []
        for param in self.params:
            input_params.append(tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def invoke_model(self, obs):
        stage_inputs, job_inputs, jobs, src_job, num_src_exec, frontier_stages, exec_limits, \
            exec_commit, moving_executors, exec_map, action_map = self.translate_state(obs)

        # get msg passing path (with cache)
        gcn_mats, gcn_masks, job_summ_backward_map, running_jobs_mat, jobs_changed = self.msg_passing.get_msg_path(jobs)
        # get valid masks
        stage_valid_mask, job_valid_mask = self.get_valid_masks(jobs, frontier_stages, src_job, num_src_exec, exec_map, action_map)
        # get summ path which ignores the finished stages
        summ_mats = ReinforceAgent.get_unfinished_stages_summ_mat(jobs)

        # invoke learning model
        stage_act_probs, job_act_probs, stage_acts, job_acts = self.predict(stage_inputs, job_inputs, stage_valid_mask,
                                                                            job_valid_mask, gcn_mats, gcn_masks, summ_mats,
                                                                            running_jobs_mat, job_summ_backward_map)

        return stage_acts, job_acts, stage_act_probs, job_act_probs, stage_inputs, job_inputs, stage_valid_mask, \
            job_valid_mask, gcn_mats, gcn_masks, summ_mats, running_jobs_mat, job_summ_backward_map, exec_map, jobs_changed

    # the following 4 funcs are called in invoke_model()
    def translate_state(self, obs):
        """
        Translate the observation from Schedule.observe() into tf tensor format.
        This func gives the design of raw (feature) input.
        """
        jobs, src_job, num_src_exec, frontier_stages, exec_limits, exec_commit, moving_executors, action_map = obs
        total_num_stages = sum([job.num_stages for job in jobs])

        # set stage_inputs and job_inputs
        stage_inputs = np.zeros([total_num_stages, self.stage_input_dim])
        job_inputs = np.zeros([len(jobs), self.job_input_dim])

        exec_map = {}         # {job: num of executors allocated to it}
        for job in jobs:
            exec_map[job] = len(job.executors)
        # count the moving executors in
        for stage in moving_executors.moving_executors.values():
            exec_map[stage.job] += 1
        # count exec_commit in
        for src in exec_commit.commit:
            job = None
            if isinstance(src, Job):
                job = src
            elif isinstance(src, Stage):
                job = src.job
            elif src is None:
                job = None
            else:
                print('source', src, 'unknown!')
                exit(1)
            for stage in exec_commit.commit[src]:
                if stage is not None and stage.job != job:
                    exec_map[stage.job] += exec_commit.commit[src][stage]

        # gather job level inputs (thw following demonstrates the raw feature design)
        job_idx = 0
        for job in jobs:
            job_inputs[job_idx, 0] = exec_map[job] / 20.                   # dim0: num executors
            job_inputs[job_idx, 1] = 2 if job is src_job else -2           # dim1: cur exec belongs to this job or not
            job_inputs[job_idx, 2] = num_src_exec / 20.                    # dim2: num of src execs
            job_idx += 1
        # gather stage level inputs
        stage_idx = 0
        job_idx = 0
        for job in jobs:
            for stage in job.stages:
                stage_inputs[stage_idx, :3] = job_inputs[job_idx, :3]
                stage_inputs[stage_idx, 3] = (stage.num_tasks - stage.next_task_idx) * stage.tasks[-1].duration / 100000.  # remaining task execution tm
                stage_inputs[stage_idx, 4] = (stage.num_tasks - stage.next_task_idx) / 200.                                # num of remaining tasks
                stage_idx += 1
            job_idx += 1

        return stage_inputs, job_inputs, jobs, src_job, num_src_exec, frontier_stages, exec_limits, \
            exec_commit, moving_executors, exec_map, action_map

    def get_valid_masks(self, jobs, frontier_stages, src_job, num_src_exec, exec_map, action_map):
        job_valid_mask = np.zeros([1, len(jobs) * len(self.executor_levels)])
        job_valid = {}       # {job: True or False}

        base = 0
        for job in jobs:
            # new executor level depends on the src exec
            if job is src_job:
                # + 1 because we want at least one exec for this job
                least_exec_amount = exec_map[job] - num_src_exec + 1
            else:
                least_exec_amount = exec_map[job] + 1
            assert 0 < least_exec_amount <= self.executor_levels[-1] + 1

            # find the idx of the first valid executor limit
            exec_level_idx = bisect.bisect_left(self.executor_levels, least_exec_amount)
            if exec_level_idx >= len(self.executor_levels):
                job_valid[job] = False
            else:
                job_valid[job] = True

            # jobs behind exec_level_idx are valid
            for lvl in range(exec_level_idx, len(self.executor_levels)):
                job_valid_mask[0, base + lvl] = 1
            base += self.executor_levels[-1]

        total_num_stages = sum([job.num_stages for job in jobs])
        stage_valid_mask = np.zeros([1, total_num_stages])
        for stage in frontier_stages:
            if job_valid[stage.job]:
                act = action_map.inverse_map[stage]
                stage_valid_mask[0, act] = 1

        return job_valid_mask, stage_valid_mask

    @staticmethod
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

    def predict(self, stage_inputs, job_inputs, stage_valid_mask, job_valid_mask, gcn_mats, gcn_masks, summ_mats,
                running_jobs_mat, job_summ_backward_map):
        return self.sess.run(
            [self.stage_act_probs, self.job_act_probs, self.stage_acts, self.job_acts],
            feed_dict={
                i: d for i, d in zip(
                    [self.stage_inputs] + [self.job_inputs] + [self.stage_valid_mask] + [self.job_valid_mask] +
                    self.gcn.adj_mats + self.gcn.masks + self.gsn.summary_mats + [self.job_summ_backward_map],

                    [stage_inputs] + [job_inputs] + [stage_valid_mask] + [job_valid_mask] +
                    gcn_mats + gcn_masks + [summ_mats, running_jobs_mat] + [job_summ_backward_map]
                )
            }
        )

    # the following 4 funcs (return sess.run()) are called in model training
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

    def get_gradients(self, stage_inputs, job_inputs, stage_valid_mask, job_valid_mask, gcn_mats, gcn_masks, summ_mats,
                      running_jobs_mat, job_summ_backward_map, stage_act_vec, job_act_vec, adv, entropy_weight):
        return self.sess.run(
            [self.act_gradients, [self.adv_loss, self.entropy_loss]],
            feed_dict={
                i: d for i, d in zip(
                    [self.stage_inputs] + [self.job_inputs] + [self.stage_valid_mask] + [self.job_valid_mask] +
                    self.gcn.adj_mats + self.gcn.masks + self.gsn.summary_mats + [self.job_summ_backward_map] +
                    [self.stage_act_vec] + [self.job_act_vec] + [self.adv] + [self.entropy_weight],

                    [stage_inputs] + [job_inputs] + [stage_valid_mask] + [job_valid_mask] +
                    gcn_mats + gcn_masks + [summ_mats, running_jobs_mat] + [job_summ_backward_map] +
                    [stage_act_vec] + [job_act_vec] + [adv] + [entropy_weight]
                )
            }
        )

    def set_params(self, input_params):
        self.sess.run(
            self.set_params_op,
            feed_dict={
                i: d for i, d in zip(self.input_params, input_params)
            }
        )

    def get_action(self, obs):
        """
        Get the next-to-schedule stage and the exec limits to it by parsing the output of the agent.
        """
        jobs, src_job, num_src_exec, frontier_stages, exec_limits, exec_commit, moving_executors, action_map = obs
        if len(frontier_stages) == 0:      # no act
            return None, num_src_exec
        stage_acts, job_acts, stage_act_probs, job_act_probs, stage_inputs, job_inputs, stage_valid_mask, job_valid_mask, \
            gcn_mats, gcn_masks, summ_mats, running_jobs_mat, job_summ_backward_map, exec_map, jobs_changed = self.invoke_model(obs)
        if sum(stage_valid_mask[0, :]) == 0:        # no valid stage to assign
            return None, num_src_exec

        assert stage_valid_mask[0, stage_acts[0]] == 1
        # parse stage action, get the to-be-scheduled stage
        stage = action_map[stage_acts[0]]
        # find the corresponding job
        job_idx = jobs.index(stage.job)
        assert job_valid_mask[0, job_acts[0, job_idx] + len(self.executor_levels) * job_idx] == 1
        # parse exec limit action
        if stage.job is src_job:
            agent_exec_act = self.executor_levels[job_acts[0, job_idx]] - exec_map[stage.job] + num_src_exec
        else:
            agent_exec_act = self.executor_levels[job_acts[0, job_idx]] - exec_map[stage.job]
        use_exec = min(
            stage.num_tasks - stage.next_task_idx - exec_commit.stage_commit[stage] - moving_executors.count(stage),
            agent_exec_act,
            num_src_exec
        )

        return stage, use_exec


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
            msg_mats = assist.merge_sparse_mats(msg_mats, args.max_depth)
            msg_masks = assist.merge_masks(msg_masks)
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
            sp_mat = assist.SparseMat(dtype=np.float32, shape=(num_stages, num_stages))
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
            msg_mats.append(assist.SparseMat(dtype=np.float32, shape=(num_stages, num_stages)))
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
    TODO: when to call?
    :param a: of shape (batch_size, num_stages)
    :param b: of shape (batch_size, num_exec_limit * num_jobs)
    :param mask:
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
