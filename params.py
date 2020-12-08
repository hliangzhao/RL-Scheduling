import argparse

parser = argparse.ArgumentParser(description='Scheduling')

# -- Basic --
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
# TODO: why add eps to loss?
parser.add_argument('--eps', type=float, default=1e-6, help='epsilon (default: 1e-6)')
# num_proc is used in multi-resource environment, not used temporarily
parser.add_argument('--num_proc', type=int, default=1, help='number of processors (default: 1)')
parser.add_argument('--num_exp', type=int, default=10, help='number of experiments (default: 10)')
parser.add_argument('--query_type', type=str, default='tpch', help='job type, "tpch" or "alibaba" (default: TPC-H)')
parser.add_argument('--job_folder', type=str, default='/data/tpch-queries/',
                    help='the data path where job info read from, "data/tpch-queries/" or "data/alibaba-cluster-trace/" (default: data/tpch-queries/)')
parser.add_argument('--result_folder', type=str, default='results/', help='the folder path where test results saved (default: results)')
parser.add_argument('--model_folder', type=str, default='models/stream_200_job_diff_reward_reset_5e-7_5e-8/', help='the folder path where trained models saved (default: models)')

# -- Environment --
parser.add_argument('--exec_cap', type=int, default=100, help='number of total executors (default: 100)')
parser.add_argument('--num_init_jobs', type=int, default=10, help='number of initial jobs in system (default: 10)')
parser.add_argument('--num_stream_jobs', type=int, default=200, help='number of all jobs generated in future (default: 200)')
parser.add_argument('--stream_interval', type=int, default=25000, help='inter job average arrival time in milliseconds (default: 25000)')
parser.add_argument('--executor_data_point', type=int, default=[5, 10, 20, 40, 50, 60, 80, 100], nargs='+', help='number of executors used in tpch-queries collection')
parser.add_argument('--reward_scale', type=float, default=100000.0, help='scale the reward to some normal values (default: 100000.0)')
parser.add_argument('--var_num_jobs', type=bool, default=False, help='vary number of jobs in batch (default: False)')
parser.add_argument('--moving_delay', type=int, default=2000, help='the delay of moving an executor to another job (milliseconds), this is the content switch cost (default: 2000)')
parser.add_argument('--warmup_delay', type=int, default=1000, help='executor warming up delay (milliseconds) (default: 1000)')
parser.add_argument('--diff_reward_enabled', type=int, default=1, help='enable differential reward (default: 0)')

# -- Multi resource environment (not available till now) --
parser.add_argument('--exec_group_num', type=int, default=[50, 50], nargs='+', help='number of executors in each type group (default: [50, 50])')
parser.add_argument('--exec_cpus', type=float, default=[1.0, 1.0], nargs='+', help='amount of CPU in each type group (default: [1.0, 1.0])')
parser.add_argument('--exec_mems', type=float, default=[1.0, 0.5], nargs='+', help='amount of memory in each type group (default: [1.0, 0.5])')

# -- Evaluation --
parser.add_argument('--test_schemes', type=str, default=['reinforce', 'fifo', 'dynamic'], nargs='+', help='schemes for testing the performance')

# -- TPC-H --
parser.add_argument('--tpch_size', type=str, default=['2g', '5g', '10g', '20g', '50g', '80g', '100g'], nargs='+',
                    help='job input data size (GB) (default: [2g, 5g, 10g, 20g, 50g, 80g, 100g])')
parser.add_argument('--tpch_num', type=int, default=22, help='number of TPC-H queries (default: 22)')

# -- Visualization --
parser.add_argument('--canvas_base', type=int, default=-10, help='Canvas color scale bottom (default: -10)')

# -- Learning --
parser.add_argument('--stage_input_dim', type=int, default=5, help='stage input dimensions to graph embedding (default: 5)')
parser.add_argument('--job_input_dim', type=int, default=3, help='job input dimensions to graph embedding (default: 3)')
parser.add_argument('--hidden_dims', type=int, default=[16, 8], nargs='+', help='hidden dimensions throughout graph embedding (default: [16, 8])')
parser.add_argument('--output_dim', type=int, default=8, help='output dimensions throughout graph embedding (default: 8)')
parser.add_argument('--max_depth', type=int, default=8, help='maximum depth of root-leaf message passing (default: 8)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
parser.add_argument('--gamma', type=float, default=1, help='cumulative reward discount factor (default: 1)')
parser.add_argument('--entropy_weight_init', type=float, default=1, help='initial exploration entropy weight (default: 1)')
parser.add_argument('--entropy_weight_min', type=float, default=0.0001, help='final minimum entropy weight (default: 0.0001)')
parser.add_argument('--entropy_weight_decay', type=float, default=1e-3, help='entropy weight decay rate (default: 1e-3)')
parser.add_argument('--master_num_gpu', type=int, default=1, help='number of GPU cores used in master (default: 0)')
parser.add_argument('--worker_num_gpu', type=int, default=1, help='number of GPU cores used in worker (default: 0)')
parser.add_argument('--master_gpu_fraction', type=float, default=0.25, help='fraction of memory master uses in GPU (default: 0.5)')
parser.add_argument('--worker_gpu_fraction', type=float, default=0.1, help='fraction of memory worker uses in GPU (default: 0.1)')
parser.add_argument('--num_worker_agents', type=int, default=6, help='number of parallel agents (default: 10)')

parser.add_argument('--average_reward_storage_size', type=int, default=100000, help='storage size for computing average reward (default: 100000)')
parser.add_argument('--reset_prob', type=float, default=5e-7, help='probability for episode to reset (after x seconds) (default: 0)')
parser.add_argument('--reset_prob_decay', type=float, default=4e-10, help='decay rate of reset probability (default: 0)')
parser.add_argument('--reset_prob_min', type=float, default=5e-8, help='minimum of decay probability (default: 0)')
parser.add_argument('--num_epochs', type=int, default=10000000, help='number of training epochs (default: 10000000)')
parser.add_argument('--learn_obj', type=str, default='mean', help='learning objective, chosen between "mean" and "makespan" (default: mean)')
parser.add_argument('--saved_model', type=str, default=None, help='path to the saved trained model (default: None)')
parser.add_argument('--model_save_interval', type=int, default=100000, help='interval for saving PyTorch model (default: 100000)')
parser.add_argument('--num_saved_models', type=int, default=1000, help='number of models to keep (default: 1000)')

# -- Spark interface (not implemented) --
parser.add_argument('--scheduler_for_spark', type=str, default='reinforce', help='type of scheduling algorithm (default: reinforce)')

args = parser.parse_args()
