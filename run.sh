# prepare environment
conda create --name rl-scheduling python=3.7
conda activate rl-scheduling

conda install pytorch torchvision torchaudio -c pytorch
conda install tensorflow=1.14
conda install networkx
echo "Environment prepared!"

# model train
echo "Start training reinforce agent ..."
python algo/learn/tf/train.py --exec_cap 50 --num_init_jobs 1 --num_stream_jobs 200 --reset_prob 5e-7 --reset_prob_min 5e-8 --reset_prob_decay 4e-10 --diff_reward_enabled 1 --num_worker_agents 16 --model_save_interval 100 --model_folder ./models/stream_200_job_diff_reward_reset_5e-7_5e-8/

# model test
echo "Start testing ..."
python algo/test.py --exec_cap 50 --num_init_jobs 1 --num_stream_jobs 5000 --num_exp 1 --saved_model ./models/stream_200_job_diff_reward_reset_5e-7_5e-8/model_ep_10000
