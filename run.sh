# prepare environment
conda create --name rl-scheduling python=3
conda activate rl-scheduling

conda install pytorch -c pytorch
conda install tensorflow-gpu==1.15
conda install networkx
echo "Environment prepared!"

# model train
cd RL-Scheduling
echo "Start training reinforce agent ..."
python algo/learn/tf/train.py

# model test
echo "Start testing ..."
python algo/test.py --exec_cap 50 --num_init_jobs 1 --num_stream_jobs 5000 --num_exp 1 --saved_model models/stream_500_job_diff_reward_reset_5e-7_5e-8/model_epoch_10000
