import torch
import gym
import random
import numpy as np
# import matplotlib.pyplot as plt
import os

from networks.structures import PolicyNetwork, ValueNetwork, SoftQNetwork


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ReplayBuffer:
    """
    A experience buffer used to store and replay data

    Parameters
    ----------
        capacity : [int]
            The max size of the buffer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Add data to the buffer
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    # def sample(self, batch_size):
    #     """
    #     Sample a random batch of memmory data from the buffer
    #     ----------
    #     batch_size : [int]
    #         The size of the batch

    #     Returns
    #     -------
    #     [list]
    #         A batch of rollout experience
    #     """
    #     batch = random.sample(self.buffer, batch_size)
    #     state, action, reward, next_state, done = map(np.stack, zip(*batch))
    #     return state, action, reward, next_state, done

    def sample(self, batch_size):  # this version is significantly faster
        """
        Sample a random batch of memmory data from the buffer
        ----------
        batch_size : [int]
            The size of the batch

        Returns
        -------
        [list]
            A batch of rollout experience
        """

        batch = random.sample(self.buffer, batch_size)
        state = np.array([elem[0] for elem in batch], dtype=np.double)
        action = np.array([elem[1] for elem in batch], dtype=np.double)
        reward = np.array([elem[2] for elem in batch], dtype=np.double)
        next_state = np.array([elem[3] for elem in batch], dtype=np.double)
        done = np.array([elem[4] for elem in batch], dtype=np.double)

        return state, action, reward, next_state, done

    def __len__(self):
        """
        Size of the buffer
        """
        return len(self.buffer)


def check_dir(file_name):
    """
    Checking if directory path exists
    """
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)


def restore_data(restore_path):
    """
    Restore data to re-load training

    Parameters
    ----------
    restore_path : [str]
        File path of the saved data
    """
    try:
        checkpoint = torch.load(restore_path + '/state.pt')
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
    except BaseException:
        print('Não foi possível carregar um modelo pré-existente')


def terminate():
    """
    Helper function to proper close the process and the Coppelia Simulator
    """
    try:
        env.shutdown()
        import sys
        sys.exit(0)
    except BaseException:
        import sys
        sys.exit(0)
