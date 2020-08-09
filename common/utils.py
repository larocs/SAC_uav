import torch
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import os


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

def check_dir(file_name):
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

def restore_data(restore_path):
    try:
        checkpoint = torch.load(restore_path+'/state.pt')
        # checkpoint = torch.load(restore_path)

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




def terminate():
    try:
        env.shutdown();import sys; sys.exit(0)
    except:
        import sys; sys.exit(0)


