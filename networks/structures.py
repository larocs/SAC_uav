import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Beta

# from common.utils import *


class ValueNetwork(nn.Module):
    """
    A Value V(s) network

    Parameters
        ----------
        state_dim : [int]
            The observation_space of the environment
        hidden_dim : [int]
            The latent dimension in the hidden-layers
        init_w : [float], optional
            Initial weights for the neural network, by default 3e-3
    """

    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """
        Forward-pass of the value net

        Parameters
        ----------
        state : [torch.Tensor]
            The input state

        Returns
        -------

            The value_hat from being in each state
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    """
    A Q(s,a)-function network

    Parameters
    ----------
    state_dim : [int]
        The observation_space of the environment
    action_dim : [int]
        The action of the environment
    hidden_dim : [int]
        The latent dimension in the hidden-layers
    init_w : [float], optional
        Initial weights for the neural network, by default 3e-3
    """

    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """
        Forward-pass of the q-value net

        Parameters
        ----------
        action : [torch.Tensor]
            The input action
        state : [torch.Tensor]
            The input state
        Returns
        -------
            Q(s,v) q-value
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    """
    The policy network for implementing SAC

    Parameters
        ----------
        state_dim : [int]
            The observation_space of the environment
        action_dim : [int]
            The action of the environment
        hidden_dim : [int]
            The latent dimension in the hidden-layers
        init_w : [float], optional
            Initial weights for the neural network, by default 3e-3
        log_std_min : int, optional
            Min possible value for policy log_std, by default -20
        log_std_max : int, optional
            Max possible value for policy log_std, by default 2
        activation_function : , optional
            Name of the activation function

    """

    def __init__(
            self,
            state_dim,
            num_actions,
            hidden_size,
            init_w=3e-3,
            log_std_min=-20,
            log_std_max=2,
            activation_function=F.relu):

        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.activation_function = activation_function

    def forward(self, state,):
        """
        Policy forward-pass

        Parameters
        ----------
        state : [torch.Tensor]
            The input state

        Returns
            [torch.Tensor] - action to be taken
        -------
        """
        x = self.activation_function(self.linear1(state))

        x = self.activation_function(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        """
        Calculates log_prob and squashes the action

        Parameters
        ----------
        state : [torch.Tensor]
            The input state

        Returns
        -------
            squashed_action, log_prob, raw_action, policy_mean, policy_log_std
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()  # Add reparam trick?
        action = torch.tanh(z)

        # -  np.log(self.action_range) See gist https://github.com/quantumiracle/SOTA-RL-Algorithms/blob/master/sac_v2.py
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        # log_prob = normal.log_prob(z) - torch.log(torch.clamp(1 - action.pow(2),
        # min=0,max=1) + epsilon) # nao precisa por causa do squase tanh

        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        """
        Return stochastic action, without calculating log_prob

        Parameters
        ----------
        state : [torch.Tensor]
            The input state

        Returns
        -------
        squashed_action:
            Action after tanh
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()

        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]

    def deterministic_action(self, state):
        """
        Return deterministic action, without calculating log_prob

        Parameters
        ----------
        state : [torch.Tensor]
            The input state

        Returns
        -------
        squashed_action:
            Action after tanh
        """
        mean, log_std = self.forward(state)
        action = torch.tanh(mean)

        action = action.detach().cpu().numpy()
        return action[0]
