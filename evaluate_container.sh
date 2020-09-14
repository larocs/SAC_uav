#!/bin/bash

## Path to the trained policy
# SAVED_POLICY=saved_policies/sac_optimal_policy.pt
SAVED_POLICY=saved_policies/sac_optimal_policy_2.pt

## Reward function
REWARD_FUNCTION=Reward_24
## Initial position probability distribution (\rho_{o})
env_reset_mode=Discretized_Uniform
## State-Space
STATE=New_action


python3 evaluate_container.py --headless=True --reward_function=${REWARD_FUNCTION} --state=${STATE} --file=${SAVED_POLICY} \
--env_reset_mode=${env_reset_mode}

