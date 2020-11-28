"""
This module defines the funcs to multi-process training the reinforce agent.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
import tensorflow as tf
import os
import time
import multiprocessing as mp
import numpy as np
from schedule import Schedule
from algo.learn.tf.reinforce_agent import ReinforceAgent, leaky_relu
from algo.learn.tf import assist
import algo.learn.baselines as bl
from params import args
import utils


# only show the errors and the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# model training entry
def train_master():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # creat model folder
    utils.create_folder(args.model_folder)

    # initialize communication queues
    param_queues = [mp.Queue(1) for _ in range(args.num_worker_agents)]
    reward_queues = [mp.Queue(1) for _ in range(args.num_worker_agents)]
    adv_queues = [mp.Queue(1) for _ in range(args.num_worker_agents)]
    gradients_queues = [mp.Queue(1) for _ in range(args.num_worker_agents)]

    # setup training agents
    worker_agents = []
    for i in range(args.num_worker_agents):
        worker_agents.append(
            mp.Process(target=train_worker,
                       args=(i, param_queues[i], reward_queues[i], adv_queues[i], gradients_queues[i]))
        )
    # start training
    for i in range(args.num_worker_agents):
        worker_agents[i].start()

    # GPU config
    config = tf.ConfigProto(
        device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.master_gpu_fraction)
    )

    # start the session
    sess = tf.Session(config=config)

    # setup master agent
    master_agent = ReinforceAgent(sess, args.stage_input_dim, args.job_input_dim, args.hidden_dims, args.output_dim,
                                  args.max_depth, range(1, args.exec_cap + 1), activate_fn=leaky_relu, eps=args.eps)
    # setup tf logger
    tf_logger = assist.Logger(
        sess,
        var_list=['actor_loss', 'entropy', 'value_loss', 'episode_length', 'avg_reward_per_sec', 'sum_reward',
                  'reset_prob', 'num_jobs', 'reset_hit', 'avg_job_duration', 'entropy_weight']
    )

    # more init
    avg_reward_calculator = AvgRewardPerStep(args.average_reward_storage_size)
    entropy_weight = args.entropy_weight_init
    episode_reset_prob = args.reset_prob

    # start training
    for ep in range(1, args.num_epochs):
        print('training epoch', ep)
        # synchronize master's params to each worker agent
        master_params = master_agent.get_params()
        # generate max time stochastically based on the reset prob
        max_time = generate_coin_flips(episode_reset_prob)
        for i in range(args.num_worker_agents):
            param_queues[i].put(
                [master_params, args.seed + ep, max_time, entropy_weight]
            )
        # store for advantage computation
        all_rewards, all_diff_times, all_times, all_num_finished_jobs, all_avg_job_duration, all_reset_hit = [[]] * 5

        t1 = time.time()

        # get reward from worker
        any_agent_panic = False
        for i in range(args.num_worker_agents):
            result = reward_queues[i].get()
            if result is None:
                any_agent_panic = True
                continue
            else:
                batch_reward, batch_time, num_finished_jobs, avg_job_duration, reset_hit = result
            diff_time = np.array(batch_time[1:]) - np.array(batch_time[:-1])

            # TODO: if some agent panic, what we actually store is the result of the last agent?
            all_rewards.append(batch_reward)
            all_diff_times.append(diff_time)
            all_times.append(batch_time[1:])
            all_num_finished_jobs.append(num_finished_jobs)
            all_avg_job_duration.append(avg_job_duration)
            all_reset_hit.append(reset_hit)

            avg_reward_calculator.add_multi_filter_zeros(batch_reward, diff_time)

        t2 = time.time()
        print('Got reward from workers in', t2 - t1, 'secs')

        if any_agent_panic:
            # the try condition in train_worker() breaks, and throw out this rollout
            for i in range(args.num_worker_agents):       # TODO: log this event
                adv_queues[i].put(None)
            continue                       # TODO: why add continue?

        # compute differential reward
        all_cum_reward = []
        avg_reward_per_step = avg_reward_calculator.get_avg_reward_per_step()
        for i in range(args.num_worker_agents):
            if args.diff_reward_enabled:
                # differential reward
                rewards = np.array(
                    [r - avg_reward_per_step * t for (r, t) in zip(all_rewards[i], all_diff_times[i])]
                )
            else:
                # regular reward
                rewards = np.array(
                    [r for (r, t) in zip(all_rewards[i], all_diff_times[i])]
                )
            cum_reward = discount(rewards, args.gamma)
            all_cum_reward.append(cum_reward)

        # compute baseline
        baselines = bl.get_piecewise_linear_fit_bl(all_cum_reward, all_times)

        # send adv to workers
        for i in range(args.num_worker_agents):
            batch_adv = all_cum_reward[i] - baselines[i]
            adv_queues[i].put(np.reshape(batch_adv, [len(batch_adv), 1]))

        t3 = time.time()
        print('Advantage ready in', t3 - t2, 'secs')

        # need-to-log values
        all_action_loss, all_entropy, all_value_loss = [[]] * 3

        # collect gradients from workers
        master_gradient = []
        for i in range(args.num_worker_agents):
            worker_gradient, loss = gradients_queues[i].get()
            master_gradient.append(worker_gradient)
            all_action_loss.append(loss[0])
            all_entropy.append(-loss[1] / float(all_cum_reward[i].shape[0]))
            all_value_loss.append(loss[2])

        t4 = time.time()
        print('Master collected gradients from workers in', t4 - t3, 'secs')
        master_agent.apply_gradients(aggregate_gradients(master_gradient), args.lr)

        t5 = time.time()
        print('Updated master\'s gradient in', t5 - t4, 'secs')

        # logging
        tf_logger.log(
            ep,
            [
                np.mean(all_action_loss),
                np.mean(all_entropy),
                np.mean(all_value_loss),
                np.mean([len(b) for b in baselines]),
                avg_reward_per_step * args.reward_scale,
                np.mean([cr[0] for cr in all_cum_reward]),
                episode_reset_prob,
                np.mean(all_num_finished_jobs),
                np.mean(all_reset_hit),
                np.mean(all_avg_job_duration),
                entropy_weight
            ]
        )

        # var decaying
        entropy_weight = decrease_var(entropy_weight, args.entropy_weight_min, args.entropy_weight_decay)
        episode_reset_prob = decrease_var(episode_reset_prob, args.reset_prob_min, args.reset_prob_decay)

        if ep % args.model_save_interval == 0:
            master_agent.save_model(args.model_folder + 'model_of_epoch_' + str(ep))

        # utils.progress_bar(count=ep, total=args.num_epochs)
    sess.close()


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
    sess = tf.Session(config=config)    # TODO: no sess close found

    # set up schedule event and worker agent
    schedule = Schedule()
    agent = ReinforceAgent(sess, args.stage_input_dim, args.job_input_dim, args.hidden_dims, args.output_dim,
                           args.max_depth, range(1, args.exec_cap + 1), activate_fn=leaky_relu, eps=args.eps)

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
                    # this action is valid, store the reward and time
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
                # TODO: some other agents panic for the try and the main thread throw out the rollout, in this case, reset and try again now
                continue

            # compute gradients
            agent_gradients, loss = compute_agent_gradients(agent, experience_pool, batch_adv, entropy_weight)
            # report gradients to master
            gradient_queue.put([agent_gradients, loss])

        except AssertionError:
            reward_queue.put(None)
            adv_queue.get()


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
    batch_points = truncate_experiences(experience_pool['jobs_changed'])
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
        act_gradients, loss = reinforce_agent.get_gradients(
            stage_inputs, job_inputs, stage_valid_mask, job_valid_mask,
            extended_gcn_mats, extended_gcn_masks, extended_summ_mats,
            extended_running_job_mats, summ_backward_map, stage_act_vec,
            job_act_vec, adv, entropy_weight)

        all_gradients.append(act_gradients)
        all_loss[0].append(loss[0])
        all_loss[1].append(loss[1])
    all_loss[0] = np.sum(all_loss[0])
    all_loss[1] = np.sum(all_loss[1])
    all_loss[2] = np.sum(batch_adv ** 2)          # time-based baseline loss

    # aggregate all gradients from the batch
    gradients = aggregate_gradients(all_gradients)

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

    def add(self, reward, time):
        if self.count >= self.size:
            popped_reward = self.reward_record.pop(0)
            popped_time = self.time_record.pop(0)
            self.reward_sum -= popped_reward
            self.time_sum -= popped_time
        else:
            self.count += 1
        self.reward_record.append(reward)
        self.time_record.append(time)
        self.reward_sum += reward
        self.time_sum += time

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


# ======= the following are auxiliary funcs for model training =======
def aggregate_gradients(gradients):
    ground_gradients = [np.zeros(g.shape) for g in gradients[0]]
    for grad in gradients:
        for i in range(len(ground_gradients)):
            ground_gradients[i] += grad[i]
    return ground_gradients


def increase_var(var, max_var, increase_rate):
    if var + increase_rate <= max_var:
        var += increase_rate
    else:
        var = max_var
    return var


def decrease_var(var, min_var, decay_rate):
    if var - decay_rate >= min_var:
        var -= decay_rate
    else:
        var = min_var
    return var


def truncate_experiences(bool_list):
    """
    Truncate experience.
    Example: bool_list = [True, False, True], return [0, 2, 3]
    """
    batch_points = [idx for idx, bool_v in enumerate(bool_list) if bool_v]
    batch_points.append(len(bool_list))
    return batch_points


def generate_coin_flips(prob):
    """
    Flip coins until the first head appears. Return the times of flip.
    Obviously, the returned value follows geometric distribution.
    """
    if prob == 0:
        return np.inf
    return np.random.geometric(prob)


def discount(x, gamma):
    """
    Used for decaying cumulate reward.
    It's a linear filter to input x. This func is equal to
    scipy.signal.lfilter([1], [1, -gamma], x[:-1], axis=0)[:-1].
    """
    out = np.zeros(x.shape)
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    return out
