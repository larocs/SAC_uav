#!/bin/bash

# REWARD_FUNCTION=Normal
REWARD_FUNCTION=Reward_24

SEED=42
CUDA=True
PREFIX=Vibration2

env_reset_mode=Discretized_Uniform

STATE=New_action


log_interval=10

BATCH_SIZE=4000
buffer_size=1000000

net_size_value=64
net_size_policy=64
num_steps_until_train=1
num_trains_per_step=1
min_num_steps_before_training=0
learning_rate=0.0001
max_episodes=100000
ACT_FUNCTION=tanh

eval_interval=100
experiment_name=${PREFIX}_state_${STATE}_Reward_${REWARD_FUNCTION}_clipaction_${CLIP_ACTION}_lr_${learning_rate}_bat_${batch_size}_net_${net_size_value}_netpol_${net_size_policy}_ati_${ACT_FUNCTION}_buff_${buffer_size}_numsteps_${num_steps_until_train}_numtrainperstep_${num_trains_per_step}_before_${min_num_steps_before_training}_reset_${env_reset_mode}
experiment_name=teste

## SAVING MODEL
SAVED_POLICY=saved_policies/sac_optimal_policy.pt
SAVE_INTERVAL=10

python main2.py --save_path=${experiment_name} --replay_buffer_size=${buffer_size} --restore_path=${SAVED_POLICY} \
--log_interval=${log_interval} --env_reset_mode=${env_reset_mode} --batch-size=${BATCH_SIZE}     \
--net_size_value=${net_size_value} --net_size_policy=${net_size_policy} --num_steps_until_train=${num_steps_until_train} --num_trains_per_step=${num_trains_per_step} --min_num_steps_before_training=${min_num_steps_before_training} \
--use_cuda=${CUDA} --seed=${SEED} --eval_interval=100 --reward_function=${REWARD_FUNCTION} --max_episodes=${max_episodes} \
--learning_rate=${learning_rate} --use_double=True --state=${STATE} \
--activation_function=${ACT_FUNCTION} --save_interval=${SAVE_INTERVAL}
