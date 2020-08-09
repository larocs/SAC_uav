import pandas as pd
import numpy as np
import os, time
import argparse
import dill

import pyrep.backend.sim  as sim

from larocs_sim.envs.drone_env import DroneEnv

from torch import nn
from torch import optim
import torch
import csv
import random
import torch.nn.functional as F
import copy

from torch.distributions import Normal, Beta
import random
from collections import OrderedDict 



def terminate():
    try:
        env.shutdown();import sys; sys.exit(0)
    except:
        import sys; sys.exit(0)

seed = 42
# Setting seed
torch.manual_seed(seed)
random.seed(a = seed)
np.random.seed(seed=seed)


def argparser():
   
   
    parser = argparse.ArgumentParser("Benching envs")
    parser.add_argument(
        '--net_size_policy', help='Size of the neural networks hidden layers', type=int, default=256)
    parser.add_argument(
        '--activation_function', help='activation function for policy', type=str, default='relu', choices = ['relu','tanh'])
       
    parser.add_argument(
        '--net_size_value', help='Size of the neural networks hidden layers', type=int, default=256)
    parser.add_argument(
        '--max_episodes', help='Number of epochs in the training', type=int, default=int(10000))
    parser.add_argument(
        '--replay_buffer_size', help='Size of the replay buffer', type=int, default=int(1e6))
    parser.add_argument(
        '--num_steps_until_train', help='How many steps we sample with current policy', type=int, default=1)
    parser.add_argument(
        '--num_trains_per_step', help='Number of timesteps in each step', type=int, default=1)
    parser.add_argument(
        '--min_num_steps_before_training', help='Number of timesteps using random (uniform) policy to fill \
    the Replay buffer in the beggining of the training'\
                        , type=int, default=100)
    parser.add_argument( \
        '--batch_size', help='Batch size in each epoch', type=int, default=256)
    # parser.add_argument(
    #     '--use_automatic_entropy_tuning', help='Set True to automatically discover best alpha (temperature)', type=bool, default=True)    
    parser.add_argument(
        '--env_reset_mode', help='How to sample the starting position of the agent', choices=('Uniform', 'Gaussian','False', 'Discretized_Uniform'), default='False')
    parser.add_argument(
        '--use_cuda', help='If the device is a GPU or CPU', default='False', choices=['True','False'])
    parser.add_argument(
        '--restore_path', help='Filename of the policy being loaded', type=str, default=None)
    parser.add_argument(
        '--save_path', help='Filename to save the current training', type=str)
    parser.add_argument(
        '--log_interval', help='Frequency of logging', type=int, default=100)
    parser.add_argument(
        '--save-interval', help='Frequency of saving', type=int, default=100)
    parser.add_argument(
        '--eval_interval', help='Frequency for evaluating deterministic policy', type=int, default=None)
    parser.add_argument(
        '--use_double', help='Flag to use float64',  type=str, default=None)
    parser.add_argument(
        '--learning_rate', help='Learning rate',  type=float, default=3e-4)
    parser.add_argument(
        '--reward_function', help='What reward function to use', default='Normal',type=str)
    parser.add_argument(
        '--seed', help='Global seed', default=42,type=int)
    parser.add_argument(
        '--state', help='Global seed', default='Old',type=str)
    parser.add_argument(
        '--same-norm', help='same_norm', default=False)
    parser.add_argument(
        '--threshold', help='Clipping the difference between action vectors', default=4.0,type=float)
    parser.add_argument(
        '--clip-action', help='Clipping the difference between action vectors', default=100,type=int)
    parser.add_argument(
        '--save_interval', help='Frequency of saving', type=int, default=100)
    return parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2,
                                                     activation_function = F.relu):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.activation_function = activation_function



    def forward(self, state,):

        x = self.activation_function(self.linear1(state))

        x = self.activation_function(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample() ## Add reparam trick?
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon) ##  -  np.log(self.action_range) See gist https://github.com/quantumiracle/SOTA-RL-Algorithms/blob/master/sac_v2.py
        # log_prob = normal.log_prob(z) - torch.log(torch.clamp(1 - action.pow(2), min=0,max=1) + epsilon) # nao precisa por causa do squase tanh


        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std


    def get_action(self, state):
        if args.use_double:
            state = torch.DoubleTensor(state).unsqueeze(0).to(device)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z      = normal.sample()
        # z      = normal.sample().detach()
        
        action = torch.tanh(z)

        action  = action.detach().cpu().numpy()
        return action[0]




    def deterministic_action(self, state):
        if args.use_double:
            state = torch.DoubleTensor(state).unsqueeze(0).to(device)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        mean, log_std = self.forward(state)
        action = torch.tanh(mean)

        action  = action.detach().cpu().numpy()
        return action[0]


def check_dir(file_name):
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def soft_q_update(batch_size,
           gamma=0.99,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=1e-2,
          ):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    if args.use_double:
        state      = torch.DoubleTensor(state).to(device)
        next_state = torch.DoubleTensor(next_state).to(device)
        action     = torch.DoubleTensor(action).to(device)
        reward     = torch.DoubleTensor(reward).unsqueeze(1).to(device)
        done       = torch.DoubleTensor(np.float64(done)).unsqueeze(1).to(device)

    else:
        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    expected_q_value = soft_q_net(state, action)
    expected_value   = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

    target_value = target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

    expected_new_q_value = soft_q_net(state, new_action)
    next_value = expected_new_q_value - log_prob
    value_loss = value_criterion(expected_value, next_value.detach())

    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss  = std_lambda  * log_std.pow(2).mean()
    z_loss    = z_lambda    * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    soft_q_optimizer.zero_grad()
    q_value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)

    soft_q_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)

    value_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)

    policy_optimizer.step()


    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    
    return(q_value_loss.item(), policy_loss.item(), value_loss.item())

# def calc_metrics_epi(env_info):
#     radius = env_info['radius']/env.weight_dict['radius']
#     rel_or = env.drone_orientation
#     roll = rel_or[0]
#     pitch = rel_or[1]
#     yaw = rel_or[2]
    
#     return(radius, abs(roll), abs(pitch), abs(yaw))

def calc_metrics_epi(info):
    
    df_components = pd.DataFrame(info, index=[0]) 
    
    weight = pd.DataFrame(env.weight_dict, index=[0])
    cols = df_components.columns
    
    cols_metrics = [x for x in df_components.columns if x not in ['death','r_alive']]
    df_metrics = df_components[cols]/weight[cols]
    df_metrics = df_metrics[cols_metrics]
    df_metrics.columns = ['Episode_metrics_' + x for x in df_metrics.columns]
    df_components.columns = ['Episode_components_' + x for x in df_components.columns]
    
    return list(df_metrics.columns), df_metrics.values.flatten().tolist(), \
           list(df_components.columns), df_components.values.flatten().tolist()


## novo calc_metrics
def calc_metrics(df):

    metrics = pd.DataFrame()
    grouper = df.groupby('rollout')[['action_magnitude']].agg(['mean','std'])
    grouper.columns = ["_".join(x) for x in grouper.columns.ravel()]
    metrics = pd.concat([metrics,grouper.mean()], axis=0, sort=False)
    cols = ['dist_to_target'] + [col for col in df.columns if 'component' in col]
    grouper = df.groupby('rollout')[cols].agg(['mean','std'])
    grouper.columns = ["_".join(x) for x in grouper.columns.ravel()]
    metrics = pd.concat([metrics,grouper.mean()], axis=0, sort=False)
    vel_acc_cols = [col for col in df.columns if  ('angular' in col )]
    grouper = df.groupby('rollout')[vel_acc_cols].agg(['mean','std'])
    grouper.columns = ["_".join(x) for x in grouper.columns.ravel()]
    metrics = pd.concat([metrics,grouper.mean()], axis=0, sort=False)
    grouper = df.groupby('rollout')[['Average_Returns']].agg(['mean','std','min','max'])
    grouper.columns = ["_".join(x) for x in grouper.columns.ravel()]
    metrics = pd.concat([metrics,grouper.mean()], axis=0, sort=False)
    metrics.columns = ['Result']
    metrics.index.name = 'Metric'
    return metrics

def num_success(set_tau):

    list_of_dataframes = []

    for k,x in enumerate(set_tau['rewards']):
        df = pd.DataFrame(x,columns=['Rewards'])
        df['rollout'] = k
        list_of_dataframes.append(df)

    df = pd.concat(list_of_dataframes, axis = 0, ignore_index=True)

    df_results = df.groupby('rollout')[['Rewards']].sum()
    df_results['successfully_finished'] = 0
    index_list = df_results[df_results['Rewards'] > 800].index
    df_results.loc[index_list, 'successfully_finished'] = 1


    print("Percentage of complete tests = {0:.2f} %.".format(df_results['successfully_finished'].mean() * 100))
    return (df_results['successfully_finished'].mean() * 100)


def pyrep_state(env_pyrep, use_vision_sensor = False):
    state = OrderedDict()
    #state['position'] = env_pyrep.agent.get_position()
    state['position_x'] = env_pyrep.agent.get_position()[0]
    state['position_y'] = env_pyrep.agent.get_position()[1]
    state['position_z'] = env_pyrep.agent.get_position()[2]
    state['dist_to_target'] = np.sum(np.array(env_pyrep.agent.get_position(relative_to=env.target))**2)
    
    #state['orientation'] = env_pyrep.agent.get_orientation()
    state['orientation_roll'] = env_pyrep.agent.get_orientation()[0]
    state['orientation_pitch'] = env_pyrep.agent.get_orientation()[1]
    state['orientation_yaw'] = env_pyrep.agent.get_orientation()[2]
    
    #state['linear_velocity'] = env_pyrep.agent.get_velocity()[0]
    state['linear_velocity_x'] = env_pyrep.agent.get_velocity()[0][0]
    state['linear_velocity_y'] = env_pyrep.agent.get_velocity()[0][1]
    state['linear_velocity_z'] = env_pyrep.agent.get_velocity()[0][2]
        
    #state['angular_velocity'] = env_pyrep.agent.get_velocity()[1]
    state['angular_velocity_x'] = env_pyrep.agent.get_velocity()[1][0]
    state['angular_velocity_y'] = env_pyrep.agent.get_velocity()[1][1]
    state['angular_velocity_z'] = env_pyrep.agent.get_velocity()[1][2]
    

    return state

def clipped_increment(a,b,threshold, same_norm = False):
    
    b_minus_a = b-a
    norm_b_minus_a = np.linalg.norm(b_minus_a, ord =2)
    if same_norm:
        prev_norm_b = np.linalg.norm(b, ord =2)
    if (norm_b_minus_a) <= threshold:
        pass
    else:
        b_minus_a_unit = np.divide(b_minus_a, norm_b_minus_a)
        b_minus_a_increment = np.multiply(b_minus_a_unit,  threshold)
        b = np.add(b_minus_a_increment,a)
        if same_norm:
            b = np.divide(b, np.linalg.norm(b, ord =2))
            b = np.multiply(b,prev_norm_b)

    return b



def get_summarized_metrics(set_tau):
    
    df = create_dataframe(set_tau)


    metrics = calc_metrics(df)
    last_15_metrics = calc_metrics(df.groupby('rollout').apply(lambda x : x.iloc[-15:,:]).reset_index(drop=True))


    metrics_index = list(metrics.index)
    metrics_values = list(metrics.values)
    last15_index = ['last15_'+x for x in last_15_metrics.index]
    last15_values = list(last_15_metrics.values)
    index = metrics_index + last15_index
    values = metrics_values + last15_values
    values = [x[0] for x in values]

    index = ['Eval_'+x for x in index]
    return index,values 


def create_dataframe(set_tau):
    
    list_of_dataframes = []
    for k, (list_1, list_2, list_3, list_4) in enumerate(zip(set_tau['big_states'],\
                                 set_tau['rewards'], set_tau['infos'], set_tau['actions'])):

        df = pd.DataFrame(list_1)
        df['Average_Returns'] = list_2
        df = df[[column for column in df.columns if "Vision" not in column]]
        df['rollout'] = k

        df3 = pd.DataFrame(list_3)
        ## cols with standard deviation bigger than zero
        cols_std_bigger_than_zero = list((df3.std(axis=0)[(df3.std(axis=0) != 0)].index))
        df3 = df3[cols_std_bigger_than_zero]
        df3.columns = ['component_{0}'.format(col) for col in df3.columns]

        df4 = pd.DataFrame(list_4)
        df4.columns = ['action_{0}'.format(ele) for ele in df4.columns]
        df4['action_magnitude'] = df4.apply(np.linalg.norm,axis=1, args = [2])


        df = pd.concat([df,df3, df4], axis=1,sort=False)
        list_of_dataframes.append(df)
    df = pd.concat(list_of_dataframes, axis = 0, ignore_index=True,sort=False)
    df = df[['rollout', 'Average_Returns'] + list(set(df.columns) - set(['rollout', 'Average_Returns']))]

    df_acc = df[[col for col in df.columns if 'velocity' in col]]
    df_acc_shift = df_acc.shift(1)
    df_acc_shift.iloc[0, :] = df_acc_shift.iloc[1,:]
    df_acc = df_acc.subtract(df_acc_shift, fill_value=None).div(env.dt)


    df_acc.columns = [col.replace('velocity','acceleration') for col in df_acc.columns]

    df = pd.concat([df,df_acc], axis=1)
    
    return df

def rollouts(env, policy, action_range,clip=False, threshold = 5, same_norm=False,clip_action=100, \
             max_timesteps = 1000,  time_horizon=250):

    count = 0
    dones = False
    set_of_obs,set_of_next_obs,set_of_rewards,set_of_actions,set_of_dones,set_of_infos,set_of_big_states  = [], [],[],[],[],[],[]


    rollout=-1

    while True:
        mb_obs, mb_next_obs, mb_rewards, mb_actions, mb_dones, mb_infos, mb_big_states = [], [], [], [], [], [], []

        # sim.simRemoveBanner(sim.sim_handle_all)
        rollout+=1

       
        # sim.simAddBanner(label = "Rollout = {0}".format(rollout).encode('ascii'),\
        #      size = 0.2,\
        #      options =  1,
        #      positionAndEulerAngles=[0,0,2.5,1.3,0,0],
        #      parentObjectHandle = -1)
        
        obs0 = env.reset()
        try:
            a1, agent_info = policy.deterministic_action(obs0)
        except:
            a1 = policy.deterministic_action(obs0)
                
        new_next_action = a1  
        first_action = True
        
        
        for j in range(time_horizon):
            dones = False
            if count == max_timesteps:
                set_tau = {'obs' : set_of_obs,
                'next_obs' : set_of_next_obs,
                'rewards' : set_of_rewards,
                'actions' : set_of_actions,
                'dones' : set_of_dones,
                'infos' : set_of_infos,
                'big_states' : set_of_big_states}
                return set_tau
            
            if first_action==True:
            
                first_action = False
            elif first_action == False:
                
                try:
                    a2, agent_info = policy.deterministic_action(obs0)
                except:
                    a2= policy.deterministic_action(obs0)
                if clip:
                    new_next_action = ( clipped_increment(a1, a2 , threshold,same_norm=same_norm))
                else:
                    new_next_action = a2
            
            if env.agent.use_vision_sensor:
                state = pyrep_state(env, use_vision_sensor=True)
            else:
                state = pyrep_state(env)
            
            ## Getting a state bigger then the one being used for Reinf. Lear. calculations

            # Take actions in env and look the results
            # Infos contains a ton of useful informations

            obs1, rewards, dones, infos = env.step(np.clip(new_next_action*action_range[1],\
                                       a_min=-clip_action, a_max=+clip_action))

            obs0 = obs1
            a1 = new_next_action
            
            ## Append on the experience buffers
            # mb_obs.append(obs.copy())
            mb_obs.append(obs0)
            mb_next_obs.append(obs1)
            mb_actions.append(a1)
            mb_dones.append(dones)
            mb_rewards.append(rewards)
            mb_infos.append(infos)
            mb_big_states.append(state)

            count += 1
            if dones == True:
                break

                set_of_obs.append(mb_obs)
        set_of_next_obs.append(mb_next_obs)
        set_of_rewards.append(mb_rewards)
        set_of_actions.append(mb_actions)
        set_of_dones.append(mb_dones)
        set_of_infos.append(mb_infos)
        set_of_big_states.append(mb_big_states)




args = argparser()
if (args.env_reset_mode) == 'False':
    args.env_reset_mode = False

if (args.use_cuda) == 'False':
    args.use_cuda = False
else:
    args.use_cuda = True
if args.activation_function == 'relu':
    args.activation_function = F.relu
else:
    args.activation_function = F.tanh


use_cuda = torch.cuda.is_available()

if use_cuda and (args.use_cuda == True):
    device=torch.device("cuda")
else:
    device=torch.device("cpu")


# Set environment

env = DroneEnv(random=args.env_reset_mode, headless=True, seed=args.seed,\
 reward_function_name=args.reward_function,state=args.state)


# Set save/restore paths
save_path = os.path.join('./checkpoint/', args.save_path) +'/'


restore_path = save_path

report_folder = save_path #save_path.split('/')[0] + '/'

reward_file = 'mean_reward_10_eps.csv'
# reward_file = 'mean_reward_100_eps.csv'


# Check if they exist
check_dir(save_path)
if restore_path:
    check_dir(restore_path)
    print("Restore path = ", restore_path)
check_dir(report_folder)



action_dim = env.action_space.shape[0]
state_dim = env.observation_space[0]
hidden_dim = 256
hidden_dim = args.net_size_value
action_range = [env.agent.action_space.low.min(), env.agent.action_space.high.max()]



policy_net = PolicyNetwork(state_dim, action_dim, args.net_size_policy).to(device).type(torch.double)

set_tau = rollouts(env, policy_net,action_range,\
            clip=True, threshold = 4, same_norm = True, clip_action=100, \
            max_timesteps = 250,  time_horizon=250)



index_cols, values = get_summarized_metrics(set_tau)




## Getting the reward componenet columns!
obs0 = env.reset()
observation, reward, done, info = env.step(np.array([1,1,1,1]))
metrics_index, metrics_values, components_index, components_values = calc_metrics_epi(info)




if not os.path.isfile(os.path.join(report_folder,reward_file)):
    print('There is no csv there')
    with open(os.path.join(report_folder,reward_file), 'w') as outcsv:
        writer = csv.writer(outcsv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Episode", "Total time (s)", "Frame",
                                                 "Buffer_size",
                                                  "Mean Reward 10 eps",
                                               "Last_Eval_Return",
                                               "Last_Eval_Return_Std",
                                                "value_loss", "q_value_loss", "policy_loss", \
                                                "lenght_mean", "lenght_min", "lenght_max", "lenght_std", \
                                                ] \
                                                + metrics_index \
                                                + components_index \
                                                + index_cols
                                                )


# Networks instantiation
replay_buffer_size = args.replay_buffer_size
replay_buffer = ReplayBuffer(replay_buffer_size)

if args.use_double:
    value_net = ValueNetwork(state_dim, hidden_dim).to(device).type(torch.double)
    target_value_net = ValueNetwork(state_dim, hidden_dim).to(device).type(torch.double)
    soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device).type(torch.double)
    policy_net = PolicyNetwork(state_dim, action_dim, args.net_size_policy).to(device).type(torch.double)
    replay_buffer.buffer = np.asarray(replay_buffer.buffer).astype(np.float64).tolist() 
else:
    value_net = ValueNetwork(state_dim, hidden_dim).to(device)
    target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)
    soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, args.net_size_policy).to(device)
    



eval_policy = copy.deepcopy(policy_net)



for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    
# Types of losses
value_criterion  = nn.MSELoss()
soft_q_criterion = nn.MSELoss()


# Learning rates
value_lr  = args.learning_rate
soft_q_lr = args.learning_rate
policy_lr = args.learning_rate



value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)




episode = 0
frame_count = 0
max_episodes = args.max_episodes

time_horizon = 250
batch_size = args.batch_size




# Load parameters if necessary
if restore_path is not None:
    try:
        # checkpoint = torch.load(restore_path+'/state.pt')
        checkpoint = torch.load(restore_path)

        # Episode and frames
        episode = checkpoint['episode']
        frame_count = checkpoint['frame_count']
        # Models
        value_net.load_state_dict(checkpoint['value_net'])
        target_value_net.load_state_dict(checkpoint['target_value_net'])
        soft_q_net.load_state_dict(checkpoint['soft_q_net'])
        policy_net.load_state_dict(checkpoint['policy_net'])
        # Optimizers
        value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        soft_q_optimizer.load_state_dict(checkpoint['soft_q_optimizer'])
        policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        replay_buffer = checkpoint['replay_buffer']
    except:
        print('Não foi possível carregar um modelo pré-existente')






begin = time.time()
# Run algorithm
rewards = []
print('env_reset_mode = ', args.env_reset_mode)
percentage_completed = []

eval = False
if args.eval_interval:
    eval = True

count =0



print('esquece que eu ')
same_norm = args.same_norm
threshold = args.threshold
clip_action = args.clip_action





episode_metrics = []
episode_components = []
list_eval=[]
list_eval_std=[]

episode_last_epi_metrics = []
episode_lenght = []
eval_metrics_radius = []
while episode < max_episodes:
    
    if (episode % 5 == 0): ## Hack because of PyRep set position bug
        env.restart=True
        env.reset()
        env.restart=False


    if (episode % 100 == 0) & (episode > -1) and (eval==True):
        # env,restart=True
        begin2=time.time()
        print("Eval Phase")
        set_tau = rollouts(env, policy_net,action_range,\
                    clip=True, threshold = threshold, same_norm = same_norm, clip_action=clip_action, \
                   max_timesteps = 2000,  time_horizon=250)

        print('Reward médio = {0:.2f}'.format(np.mean(([np.mean(reward) for reward in set_tau['rewards']]))))
        average_returns = np.mean(([np.sum(reward) for reward in set_tau['rewards']]))
        std_returns = np.std(([np.sum(reward) for reward in set_tau['rewards']]))

        print('Soma média = {0:.2f}'.format(average_returns))

        list_eval.append(average_returns)
        list_eval_std.append(std_returns)

        env.restart=False

        _, values = get_summarized_metrics(set_tau)


        print()

    else:
        list_eval.append(None)
        list_eval_std.append(None)

    
    if episode % 20 == 0:
        env.restart=True
        env.reset()
        env.restart=False

    
    state = env.reset()
    episode_reward = 0
    epi_metrics = []
    epi_components = []

    qvalue_loss_history = []
    value_loss_history = []
    policy_loss_history = []


    for step in range(time_horizon):
        if frame_count > args.min_num_steps_before_training:
            action = policy_net.get_action(state)# .detach()
            next_state, reward, done, env_info = env.step(action*action_range[1])
        else:
            action = np.random.sample(action_dim)
            next_state, reward, done, env_info = env.step(action*action_range[1])

        replay_buffer.push(state, action, reward, next_state, done)
        # metrics = calc_metrics_epi(env_info)
        _, metrics_values, _, components_values = calc_metrics_epi(env_info)

        if len(replay_buffer) > batch_size:
            if (episode % args.num_steps_until_train )== 0:
                for i in range(args.num_trains_per_step):
                    q_value_loss, policy_loss, value_loss = soft_q_update(batch_size)
        
              
        if done:
            break
        
        state = next_state
        episode_reward += reward
        frame_count += 1
        # epi_metrics.append(metrics)
        epi_metrics.append(metrics_values)
        epi_components.append(components_values)



    episode_lenght.append(step)    

    epi_metrics = np.asarray(epi_metrics)
    epi_components = np.asarray(epi_components)

    last_epi_metrics = epi_metrics[-args.log_interval:]
    last_epi_components = epi_components[-args.log_interval:]



    if len(replay_buffer) <= batch_size:
        q_value_loss = None
        value_loss = None
        policy_loss = None


    qvalue_loss_history.append(q_value_loss)
    value_loss_history.append(value_loss)
    policy_loss_history.append(policy_loss)
        
    episode_metrics.append(np.array(epi_metrics).mean(axis=0))
    episode_components.append(np.array(epi_components).mean(axis=0))

    episode_last_epi_metrics.append(last_epi_metrics)
    

    if (episode % args.save_interval == 0):# and (episode > 0):
            save_state = {
                'episode': episode,
                'frame_count': frame_count,
                'value_net': value_net.state_dict(),
                'target_value_net': target_value_net.state_dict(),
                'soft_q_net': soft_q_net.state_dict(),
                'policy_net': policy_net.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'soft_q_optimizer':soft_q_optimizer.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'replay_buffer': replay_buffer
            }
            print('save_path = ',save_path)
            torch.save(save_state, save_path+'/state.pt')
            torch.save(save_state['policy_net'], save_path+'/state.pt'[:-3]+'_policy_{0}.pt'.format(episode))
            torch.save(save_state['policy_net'], save_path+'/state.pt'[:-3]+'_policy_last.pt')


            #plot(episode, rewards)

    rewards.append(episode_reward)
    if (episode % 100) == 0:
        print('Time elapsed so far = {0:.2f} seconds.'.format(time.time()-begin))

    if (episode % args.log_interval == 0) and (episode >=10):
        mean_radius =  np.mean([elem[0] for elem in episode_metrics])
        mean_roll =  np.mean([elem[1] for elem in episode_metrics])
        mean_pitch =  np.mean([elem[2] for elem in episode_metrics])
        mean_yaw =  np.mean([elem[3] for elem in episode_metrics])

        lenght_mean =  np.mean(episode_lenght)
        lenght_min =  np.min(episode_lenght)
        lenght_max =  np.max(episode_lenght)
        lenght_std =  np.std(episode_lenght)


        print()
        print()

        episode_lenght = episode_lenght[-args.log_interval:]
        episode_metrics = episode_metrics[-args.log_interval:]
        episode_components = episode_components[-args.log_interval:]
        episode_last_epi_metrics = episode_last_epi_metrics[-args.log_interval:]

        
        time_elapsed = time.time()-begin


        if  (episode % args.save_interval != 0):
            values = [None] * len(values)
        # Save mean reward
        with open(os.path.join(report_folder,reward_file), 'a') as csvfile:
                rew_writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                rew_writer.writerow([episode,time_elapsed , frame_count, len(replay_buffer), np.mean(rewards),
                                    list_eval[count], list_eval_std[count],
                                    value_loss, q_value_loss, policy_loss ,\
                                    lenght_mean, lenght_min, lenght_max, lenght_std, \
                                ] + \
                                (np.mean(np.array(episode_metrics), axis=0).tolist()) + \
                                (np.mean(np.array(episode_components), axis=0).tolist()) + \
                                 values)


        # Print information in terminal
        print('Episódio: {}'.format(episode))

        print('Média {0} Eps: {1}'.format(args.log_interval, np.mean(rewards[-args.log_interval:])))
        print()
        print('Value Loss = ',value_loss)
        print('Q-Value Loss = ',q_value_loss)
        print('Policy Loss = ',policy_loss)
        print("Replay Buffer len = ", len(replay_buffer))


        
        value_loss_history.append(value_loss)
        policy_loss_history.append(policy_loss)

        rewards = rewards[-args.log_interval:]
        last_eval_element =  [i for i in list_eval if i != None][-1]
        # last_percent_element =  [i for i in percentage_completed if i != None][-1]

        print()
        print(env.reward_info)
        


    episode += 1
    count += 1
env.shutdown()
