"""
This module defines the funcs to multi-process training the reinforce agent.
"""
import tensorflow as tf
import os
import time
import multiprocessing as mp
import numpy as np
from algo.learn.schedule import Schedule
from algo.learn.reinforce_agent import ReinforceAgent, leaky_relu
from algo.learn.logger import Logger
import algo.learn.baselines as bl
from algo.learn.avg_reward import AvgPerStepReward
from algo.learn import sparse_op
from params import args
import utils

# only show the errors and the warnings in tf logs
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
            mp.Process(target=train_worker, args=(i, param_queues[i], reward_queues[i], adv_queues[i], gradients_queues[i]))
        )
    # start training
    for i in range(args.num_worker_agents):
        worker_agents[i].start()

    # start the session
    sess = tf.Session(
        config=tf.ConfigProto(
            # GPU config
            device_count={'GPU': args.master_num_gpu},
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.master_gpu_fraction)
        )
    )

    # setup master agent
    master_agent = ReinforceAgent(sess, args.stage_input_dim, args.job_input_dim, args.hidden_dims, args.output_dim,
                                  args.max_depth, range(1, args.exec_cap + 1), activate_fn=leaky_relu, eps=args.eps)
    # setup tf logger
    tf_logger = Logger(
        sess,
        var_list=['total_loss', 'entropy', 'value_loss', 'episode_length', 'avg_reward_per_sec', 'sum_reward',
                  'reset_prob', 'num_jobs', 'reset_hit', 'avg_job_duration', 'entropy_weight']
    )

    # more init
    avg_reward_calculator = AvgPerStepReward(args.average_reward_storage_size)
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
        all_rewards, all_diff_times, all_times, all_num_finished_jobs, all_avg_job_duration, all_reset_hit = [], [], [], [], [], []

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

            avg_reward_calculator.add_list_filter_zero(batch_reward, diff_time)

        t2 = time.time()
        print('Got reward from workers in', t2 - t1, 'secs')

        if any_agent_panic:
            # the try condition in train_worker() breaks, and throw out this rollout
            for i in range(args.num_worker_agents):       # TODO: log this event
                adv_queues[i].put(None)
            continue                       # goto next epoch directly

        # compute differential reward
        all_cum_reward = []
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_worker_agents):
            if args.diff_reward_enabled:
                # differential reward
                rewards = np.array(
                    [r - avg_per_step_reward * t for (r, t) in zip(all_rewards[i], all_diff_times[i])]
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
        all_action_loss, all_entropy, all_value_loss = [], [], []

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
            epoch_idx=ep,
            values=[
                np.mean(all_action_loss),
                np.mean(all_entropy),
                np.mean(all_value_loss),
                np.mean([len(b) for b in baselines]),
                avg_per_step_reward * args.reward_scale,
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
            master_agent.save_model(args.model_folder + 'model_epoch_' + str(ep))

        # utils.progress_bar(count=ep, total=args.num_epochs)
    sess.close()


def train_worker(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    """
    Define how one reinforce agent is trained.
    """
    # config tf device settings
    tf.set_random_seed(agent_id)
    sess = tf.Session(
        config=tf.ConfigProto(
            device_count={'GPU': args.worker_num_gpu},
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.worker_gpu_fraction)
        )
    )

    # set up schedule event and worker agent
    schedule = Schedule()
    worker_agent = ReinforceAgent(sess, args.stage_input_dim, args.job_input_dim, args.hidden_dims, args.output_dim,
                                  args.max_depth, range(1, args.exec_cap + 1), activate_fn=leaky_relu, eps=args.eps)

    while True:
        agent_params, seed, max_time, entropy_weight = param_queue.get()
        # synchronize model
        worker_agent.set_params(input_params=agent_params)
        # reset scheduling event
        schedule.seed(seed)
        schedule.reset(max_time)
        # setup experience pool
        exp_pool = {
            'stage_inputs': [], 'job_inputs': [],
            'gcn_mats': [], 'gcn_masks': [],
            'summ_mats': [], 'running_jobs_mat': [],
            'job_summ_backward_map': [],
            'stage_act_vec': [], 'job_act_vec': [],
            'stage_valid_mask': [], 'job_valid_mask': [],
            'reward': [], 'cur_time': [],
            'jobs_changed': [],
        }

        try:
            # the masks (stage_valid_mask & job_valid_mask) have some probability to generate masked-out action,
            # when it happens (maybe in every few thousand iterations), the assert in agent.get_action() will lead to error
            # we manually catch the error and throw out the rollout
            obs = schedule.observe()
            done = False
            exp_pool['cur_time'].append(schedule.wall_time.cur_time)

            while not done:
                stage, use_exec = invoke(worker_agent, obs, exp_pool)
                # one step forward
                obs, reward, done = schedule.step(stage, use_exec)
                if stage is not None:
                    # this action is valid, store the reward and time
                    exp_pool['reward'].append(reward)
                    exp_pool['cur_time'].append(schedule.wall_time.cur_time)
                elif len(exp_pool['reward']) > 0:
                    # if we skip the reward when no legal stage is chosen, the agent will exhaustively pick all stages
                    # in one scheduling event to avoid negative reward
                    exp_pool['reward'][-1] += reward
                    exp_pool['cur_time'][-1] = schedule.wall_time.cur_time
            assert len(exp_pool['stage_inputs']) == len(exp_pool['reward'])
            reward_queue.put([
                exp_pool['reward'],
                exp_pool['cur_time'],
                len(schedule.finished_jobs),
                np.mean([job.finish_time - job.start_time for job in schedule.finished_jobs]),
                schedule.wall_time.cur_time >= schedule.max_time
            ])

            # get advantage term from master
            batch_adv = adv_queue.get()
            if batch_adv is None:
                # TODO: some other agents panic for the try and the main thread throw out the rollout, in this case, reset and try again now
                continue

            # compute gradients
            worker_gradient, loss = compute_agent_gradients(worker_agent, exp_pool, batch_adv, entropy_weight)
            # report gradients to master
            gradient_queue.put([worker_gradient, loss])

        except AssertionError:
            reward_queue.put(None)
            adv_queue.get()


def invoke(reinforce_agent, obs, exp_pool):
    """
    Feed the reinforce agent with observation, and parse the actions it returns.
    Further, record this (s, a) into experience pool for the following training.
    """
    jobs, src_job, num_src_exec, frontier_stages, exec_limits, exec_commit, moving_executors, action_map = obs
    if len(frontier_stages) == 0:  # no act
        return None, num_src_exec
    stage_act, job_act, stage_act_probs, job_act_probs, stage_inputs, job_inputs, stage_valid_mask, job_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, running_jobs_mat, job_summ_backward_map, exec_map, jobs_changed = reinforce_agent.invoke_model(obs)
    if sum(stage_valid_mask[0, :]) == 0:  # no valid stage to assign
        return None, num_src_exec

    assert stage_valid_mask[0, stage_act[0]] == 1
    # parse stage action, get the to-be-scheduled stage
    stage = action_map[stage_act[0]]
    # find the corresponding job
    job_idx = jobs.index(stage.job)
    assert job_valid_mask[0, job_act[0, job_idx] + len(reinforce_agent.executor_levels) * job_idx] == 1
    # parse exec limit action
    if stage.job is src_job:
        agent_exec_act = reinforce_agent.executor_levels[job_act[0, job_idx]] - exec_map[stage.job] + num_src_exec
    else:
        agent_exec_act = reinforce_agent.executor_levels[job_act[0, job_idx]] - exec_map[stage.job]
    use_exec = min(
        stage.num_tasks - stage.next_task_idx - exec_commit.stage_commit[stage] - moving_executors.count(stage),
        agent_exec_act,
        num_src_exec
    )

    # store the action vec into experience
    stage_act_vec = np.zeros(stage_act_probs.shape)
    stage_act_vec[0, stage_act[0]] = 1
    job_act_vec = np.zeros(job_act_probs.shape)
    job_act_vec[0, job_idx, job_act[0, job_idx]] = 1

    # store experience into pool
    exp_pool['stage_inputs'].append(stage_inputs)
    exp_pool['job_inputs'].append(job_inputs)
    exp_pool['summ_mats'].append(summ_mats)
    exp_pool['running_jobs_mat'].append(running_jobs_mat)
    exp_pool['stage_act_vec'].append(stage_act_vec)
    exp_pool['job_act_vec'].append(job_act_vec)
    exp_pool['stage_valid_mask'].append(stage_valid_mask)
    exp_pool['job_valid_mask'].append(job_valid_mask)
    exp_pool['jobs_changed'].append(jobs_changed)
    if jobs_changed:
        exp_pool['gcn_mats'].append(gcn_mats)
        exp_pool['gcn_masks'].append(gcn_masks)
        exp_pool['job_summ_backward_map'].append(job_summ_backward_map)

    return stage, use_exec


def compute_agent_gradients(reinforce_agent, exp_pool, batch_adv, entropy_weight):
    batch_points = truncate_experiences(exp_pool['jobs_changed'])
    all_gradients = []
    all_loss = [[], [], 0]

    for bt in range(len(batch_points) - 1):
        bt_start = batch_points[bt]
        bt_end = batch_points[bt + 1]

        # use one piece of experience
        stage_inputs = np.vstack(exp_pool['stage_inputs'][bt_start: bt_end])
        job_inputs = np.vstack(exp_pool['job_inputs'][bt_start: bt_end])
        stage_act_vec = np.vstack(exp_pool['stage_act_vec'][bt_start: bt_end])
        job_act_vec = np.vstack(exp_pool['job_act_vec'][bt_start: bt_end])
        stage_valid_mask = np.vstack(exp_pool['stage_valid_mask'][bt_start: bt_end])
        job_valid_mask = np.vstack(exp_pool['job_valid_mask'][bt_start: bt_end])
        summ_mats = exp_pool['summ_mats'][bt_start: bt_end]
        running_job_mats = exp_pool['running_jobs_mat'][bt_start: bt_end]
        adv = batch_adv[bt_start: bt_end, :]
        gcn_mats = exp_pool['gcn_mats'][bt]
        gcn_masks = exp_pool['gcn_masks'][bt]
        summ_backward_map = exp_pool['job_summ_backward_map'][bt]

        # given an episode of experience (advantage computed from baseline)
        batch_size = stage_act_vec.shape[0]
        # expand sparse adj_mats
        extended_gcn_mats = sparse_op.expand_sparse_mats(gcn_mats, batch_size)
        # extended masks
        # (on the dimension according to extended adj_mat)
        extended_gcn_masks = [np.tile(m, (batch_size, 1)) for m in gcn_masks]
        # expand sparse summ_mats
        extended_summ_mats = sparse_op.merge_and_extend_sparse_mats(summ_mats)
        # expand sparse running_dag_mats
        extended_running_job_mats = sparse_op.merge_and_extend_sparse_mats(running_job_mats)

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


# call the train func
train_master()
