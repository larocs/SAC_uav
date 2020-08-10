#!/bin/bash
SAVED_POLICY=saved_policies/sac_optimal_policy.pt
REWARD_FUNCTION=Reward_24
env_reset_mode=Discretized_Uniform
STATE=New_action

python evaluate.py --headless=False --reward_function=${REWARD_FUNCTION} --state=${STATE} --file=${SAVED_POLICY} \
--env_reset_mode=${env_reset_mode}



# --save_path=${experiment_name} --replay_buffer_size=${buffer_size} --restore_path=${experiment_name}/state.pt --log_interval=${log_interval} --env_reset_mode=${env_reset_mode} \
# --net_size_value=${net_size_value} --net_size_policy=${net_size_policy} --num_steps_until_train=${num_steps_until_train} --num_trains_per_step=${num_trains_per_step} --min_num_steps_before_training=${min_num_steps_before_training} \
# --use_cuda=${CUDA} --seed=${SEED} --eval_interval=100 --reward_function=${REWARD_FUNCTION} --max_episodes=${max_episodes} \
# --learning_rate=${learning_rate} --use_double=True --state=${STATE} --threshold=${THRESHOLD} --clip-action=${CLIP_ACTION} \
# --activation_function=${ACT_FUNCTION} --save_interval=${SAVE_INTERVAL}
