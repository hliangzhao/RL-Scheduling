"""
This module defines the funcs to multi-process training the reinforce agent.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
import tensorflow as tf
import os
import utils
import time
import multiprocessing as mp
from schedule import Schedule
from algo.tf.reinforce_agent import *


# only show the errors and the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_worker(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    """
    Define how one reinforce agent is trained.
    :param agent_id:
    :param param_queue:
    :param reward_queue:
    :param adv_queue:
    :param gradient_queue:
    :return:
    """
    # config tf device settings
    tf.set_random_seed(agent_id)
    config = tf.ConfigProto(
        device_count={'GPU": args.worker_num_gpu'},
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.worker_gpu_fraction)
    )
    sess = tf.Session(config=config)

    # set up schedule event and worker agent
    schedule = Schedule()
    agent = ReinforceAgent(sess, args.stage_input_dim, args.job_input_dim, args.hidden_dims, args.output_dim,
                           args.max_depth, range(1, args.exec_cap + 1), activate_fn=leaky_relu)

    while True:
        agent_params, seed, max_time, entropy_weight = param_queue.get()
        # synchronize model
        agent.set_params(input_params=agent_params)
        # reset scheduling event
        schedule.reset(max_time)
        # setup experience pool
        experience_pool = {
            'stage_inputs': [],
            'job_inputs': [],
            'summ_mats': [],
            'running_jobs_mat': [],
            'stage_act_vec': [],
            'job_act_vec': [],
            'stage_valid_mask': [],
            'job_valid_mask': [],
            'jobs_changed': [],
            'gcn_mats': [],
            'gcn_masks': [],
            'job_summ_backward_map': [],

            'reward': [],
            'cur_time': []
        }

        try:
            # the masks (stage_valid_mask & job_valid_mask) have some probability to generate masked-out action,
            # when it happens (maybe in every few thousand iterations), the assert in agent.get_action() will lead to error
            # we manually catch the error and throw out the rollout
            obs = schedule.observe()
            done = False
            experience_pool['cur_time'].append(schedule.time_horizon.cur_time)

            while not done:
                stage, use_exec = invoke(agent, obs, experience_pool)
                # one step forward
                obs, reward, done = schedule.step(stage, use_exec)
                if stage is not None:
                    # this action is valid, store the reward and tm
                    experience_pool['reward'].append(reward)
                    experience_pool['cur_time'].append(schedule.time_horizon.cur_time)
                elif len(experience_pool['reward']) > 0:
                    # if we skip the reward when no legal stage is chosen, the agent will exhaustively pick all stages
                    # in one scheduling event to avoid negative reward
                    experience_pool['reward'][-1] += reward
                    experience_pool['cur_time'][-1] = schedule.time_horizon.cur_time
            assert len(experience_pool['stage_inputs']) == len(experience_pool['reward'])
            reward_queue.put(
                [experience_pool['reward']],
                [experience_pool['cur_time']],
                len(schedule.finished_jobs),
                np.mean([job.finish_time - job.start_time for job in schedule.finished_jobs]),
                schedule.time_horizon.cur_time >= schedule.max_time
            )

            # get advantage term from master
            batch_adv = adv_queue.get()
            if batch_adv is None:
                # TODO: some other agents panic for the try and the main thread throw out the rollout, reset and try again now
                continue

            # compute gradients
            agent_gradients, loss = compute_agent_gradients(agent, experience_pool, batch_adv, entropy_weight)
            # report gradients to master
            gradient_queue.put([agent_gradients, loss])

        except AssertionError:
            reward_queue.put(None)
            adv_queue.get()


def train_master():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # creat result and model folder
    utils.create_folder(args.result_folder)
    utils.create_folder(args.model_folder)

    # initialize communication queues
    param_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    gradients_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # setup training agents
    agents = []
    for i in range(args.num_agents):
        agents.append(
            mp.Process(target=train_worker, args=(i, param_queues[i], reward_queues[i], adv_queues[i], gradients_queues[i]))
        )
    # start training
    for i in range(args.num_agents):
        agents[i].start()

    # GPU config
    config = tf.ConfigProto(
        device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.master_gpu_fraction)
    )

    # start the session
    sess = tf.Session(config=config)

    # setup master agent
    agent = ReinforceAgent(sess, args.stage_input_dim, args.job_input_dim, args.hidden_dims, args.output_dim,
                           args.max_depth, range(1, args.exec_cap + 1), activate_fn=leaky_relu)
    # setup tf logger
    tf_logger = assist.Logger(
        sess,
        var_list=['actor_loss', 'entropy', 'value_loss', 'episode_length', 'avg_reward_per_sec', 'sum_reward',
                  'reset_prob', 'num_jobs', 'reset_hit', 'avg_job_duration', 'entropy_weight']
    )

    # TODO: add avg_reward_calculator
    pass


def invoke(reinforce_agent, obs, experience_pool):
    """
    Feed the agent with observation, and parse the action returned from it.
    Further, record this (s, a) into experience pool for training.
    """
    jobs, src_job, num_src_exec, frontier_stages, exec_limits, exec_commit, moving_executors, action_map = obs
    if len(frontier_stages) == 0:  # no act
        return None, num_src_exec
    stage_acts, job_acts, stage_act_probs, job_act_probs, stage_inputs, job_inputs, stage_valid_mask, job_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, running_jobs_mat, job_summ_backward_map, exec_map, jobs_changed = reinforce_agent.invoke_model(obs)
    if sum(stage_valid_mask[0, :]) == 0:  # no valid stage to assign
        return None, num_src_exec

    assert stage_valid_mask[0, stage_acts[0]] == 1
    # parse stage action, get the to-be-scheduled stage
    stage = action_map[stage_acts[0]]
    # find the corresponding job
    job_idx = jobs.index(stage.job)
    assert job_valid_mask[0, job_acts[0, job_idx] + len(reinforce_agent.executor_levels) * job_idx] == 1
    # parse exec limit action
    if stage.job is src_job:
        agent_exec_act = reinforce_agent.executor_levels[job_acts[0, job_idx]] - exec_map[stage.job] + num_src_exec
    else:
        agent_exec_act = reinforce_agent.executor_levels[job_acts[0, job_idx]] - exec_map[stage.job]
    use_exec = min(
        stage.num_tasks - stage.next_task_idx - exec_commit.stage_commit[stage] - moving_executors.count(stage),
        agent_exec_act,
        num_src_exec
    )

    # store the action vec into experience
    stage_act_vec = np.zeros(stage_act_probs.shape)
    stage_act_vec[0, stage_acts[0]] = 1
    job_act_vec = np.zeros(job_act_probs.shape)
    job_act_vec[0, job_idx, job_acts[0, job_idx]] = 1

    # store experience into pool
    experience_pool['stage_inputs'].append(stage_inputs)
    experience_pool['job_inputs'].append(job_inputs)
    experience_pool['summ_mats'].append(summ_mats)
    experience_pool['running_jobs_mat'].append(running_jobs_mat)
    experience_pool['stage_act_vec'].append(stage_act_vec)
    experience_pool['job_act_vec'].append(job_act_vec)
    experience_pool['stage_valid_mask'].append(stage_valid_mask)
    experience_pool['job_valid_mask'].append(job_valid_mask)
    experience_pool['jobs_changed'].append(jobs_changed)
    if jobs_changed:
        experience_pool['gcn_mats'].append(gcn_mats)
        experience_pool['gcn_masks'].append(gcn_masks)
        experience_pool['job_summ_backward_map'].append(job_summ_backward_map)

    return stage, use_exec


def compute_agent_gradients(reinforce_agent, experience_pool, batch_adv, entropy_weight):
    batch_points = assist.truncate_experiences(experience_pool['jobs_changed'])
    all_gradients = []
    all_loss = [[], [], 0]

    for bt in range(len(batch_points) - 1):
        bt_start = batch_points[bt]
        bt_end = batch_points[bt + 1]

        # use one piece of experience
        stage_inputs = np.vstack(experience_pool['stage_inputs'][bt_start: bt_end])
        job_inputs = np.vstack(experience_pool['job_inputs'][bt_start: bt_end])
        stage_act_vec = np.vstack(experience_pool['stage_act_vec'][bt_start: bt_end])
        job_act_vec = np.vstack(experience_pool['job_act_vec'][bt_start: bt_end])
        stage_valid_mask = np.vstack(experience_pool['stage_valid_mask'][bt_start: bt_end])
        job_valid_mask = np.vstack(experience_pool['job_valid_mask'][bt_start: bt_end])
        summ_mats = experience_pool['summ_mats'][bt_start: bt_end]
        running_job_mats = experience_pool['running_jobs_mat'][bt_start: bt_end]
        adv = batch_adv[bt_start: bt_end, :]
        gcn_mats = experience_pool['gcn_mats'][bt]
        gcn_masks = experience_pool['gcn_masks'][bt]
        summ_backward_map = experience_pool['job_summ_backward_map'][bt]

        batch_size = stage_act_vec.shape[0]
        extended_gcn_mats = assist.expand_sparse_mats(gcn_mats, batch_size)
        extended_gcn_masks = [np.tile(m, (batch_size, 1)) for m in gcn_masks]
        extended_summ_mats = assist.merge_and_extend_sparse_mats(summ_mats)
        extended_running_job_mats = assist.merge_and_extend_sparse_mats(running_job_mats)

        # compute gradients
        act_gradients, loss = reinforce_agent.get_gradients(stage_inputs, job_inputs, stage_valid_mask, job_valid_mask,
                                                            extended_gcn_mats, extended_gcn_masks, extended_summ_mats,
                                                            extended_running_job_mats, summ_backward_map, stage_act_vec,
                                                            job_act_vec, adv, entropy_weight)
        all_gradients.append(act_gradients)
        all_loss[0].append(loss[0])
        all_loss[1].append(loss[1])
    all_loss[0] = np.sum(all_loss[0])
    all_loss[1] = np.sum(all_loss[1])
    all_loss[2] = np.sum(batch_adv ** 2)          # tm-based baseline loss

    # aggregate all gradients from the batch
    gradients = assist.aggregate_gradients(all_gradients)

    return gradients, all_loss


class AvgRewardPerStep:
    """
    TODO: how to understand time?
    """
    def __init__(self, size):
        self.size = size
        self.count = 0
        self.reward_record, self.time_record = [], []
        self.reward_sum, self.time_sum = 0, 0

    def add(self, reward, tm):
        if self.count >= self.size:
            popped_reward = self.reward_record.pop(0)
            popped_time = self.time_record.pop(0)
            self.reward_sum -= popped_reward
            self.time_sum -= popped_time
        else:
            self.count += 1
        self.reward_record.append(reward)
        self.time_record.append(tm)
        self.reward_sum += reward
        self.time_sum += tm

    def add_multi(self, reward_list, time_list):
        assert len(reward_list) == len(time_list)
        for i in range(len(reward_list)):
            self.add(reward_list[i], time_list[i])

    def add_multi_filter_zeros(self, reward_list, time_list):
        assert len(reward_list) == len(time_list)
        for i in range(len(reward_list)):
            if time_list[i] != 0:
                self.add(reward_list[i], time_list[i])
            else:
                assert reward_list[i] == 0

    def get_avg_reward_per_step(self):
        return float(self.reward_sum) / float(self.time_sum)
