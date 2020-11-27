"""
This module tests the three algorithms, ReinforceAgent, FIFOAGent, and DynamicAgent.
    Author: Hailiang Zhao (adapted from https://github.com/hongzimao/decima-sim)
"""
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from algo.learn.tf.reinforce_agent import ReinforceAgent, leaky_relu
from algo.dynamic.dynamic_agent import DynamicAgent
from algo.fifo.fifo_agent import FIFOAgent
from params import args
import utils
from schedule import Schedule
import visualization


matplotlib.use('agg')


def test():
    # creat test result folder if non-exists
    utils.create_folder(args.result_folder)

    tf.set_random_seed(args.seed)
    schedule = Schedule()
    agents = {}
    for scheme in args.test_schemes:
        if scheme == 'reinforce':
            sess = tf.Session()
            agents[scheme] = ReinforceAgent(
                sess, args.stage_input_dim, args.job_input_dim, args.hidden_dims,
                args.output_dim, args.max_depth, range(1, args.exec_cap + 1), activate_fn=leaky_relu
            )
        elif scheme == 'dynamic':
            agents[scheme] = DynamicAgent()
        elif scheme == 'fifo':
            agents[scheme] = FIFOAgent(exec_cap=args.exec_cap)
        else:
            print('Scheme ' + str(scheme) + ' unknown!')
            exit(1)

    all_total_reward = {scheme: [] for scheme in args.test_schemes}
    for exp in range(args.num_exp):
        for scheme in args.test_schemes:
            schedule.seed(args.num_exp + exp)
            schedule.reset()
            agent = agents[scheme]
            obs = schedule.observe()

            total_reward, done = 0, False
            while not done:
                stage, use_exec = agent.get_action(obs)
                obs, reward, done = schedule.step(stage, use_exec)
                total_reward += reward
            # TODO: should append 'reward' inside the while loop?
            all_total_reward[scheme].append(total_reward)

            # visualize
            visualization.show_job_time(
                schedule.finished_jobs,
                schedule.executors,
                save_path=args.result_folder + 'visualization_exp_' + str(exp) + '_scheme_' + scheme + '.png'
            )
            visualization.show_exec_usage(
                schedule.finished_jobs,
                save_path=args.result_folder + 'visualization_exp_' + str(exp) + '_scheme_' + scheme + '.png'
            )

        # plot CDF
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for scheme in args.test_schemes:
            tm, cum_reward = utils.compute_cdf(all_total_reward[scheme])
            ax.plot(tm, cum_reward)
        plt.xlabel('Total reward')
        plt.ylabel('cdf')
        plt.legend(args.test_schemes)
        fig.savefig(args.result_folder, 'total_reward_cdf.png')
        plt.close(fig)
