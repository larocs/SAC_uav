import argparse
import os
import time
import copy
import csv

import numpy as np

from torch import nn
from torch import optim
import torch

from sim_framework.envs.drone_env import DroneEnv

from common import utils
from networks.structures import PolicyNetwork, ValueNetwork, SoftQNetwork


def argparser():

    parser = argparse.ArgumentParser("Benching envs")
    parser.add_argument(
        '--net_size_policy',
        help='Size of the neural networks hidden layers',
        type=int,
        default=64)
    parser.add_argument(
        '--activation_function',
        help='activation function for policy',
        type=str,
        default='relu',
        choices=[
            'relu',
            'tanh'])

    parser.add_argument(
        '--net_size_value',
        help='Size of the neural networks hidden layers',
        type=int,
        default=256)
    parser.add_argument(
        '--max_episodes',
        help='Number of epochs in the training',
        type=int,
        default=int(10000))
    parser.add_argument(
        '--replay_buffer_size',
        help='Size of the replay buffer',
        type=int,
        default=int(1e6))
    parser.add_argument(
        '--num_steps_until_train',
        help='How many steps we sample with current policy',
        type=int,
        default=1)
    parser.add_argument(
        '--num_trains_per_step',
        help='Number of timesteps in each step',
        type=int,
        default=1)
    parser.add_argument(
        '--min_num_steps_before_training',
        help='Number of timesteps using random (uniform) policy to fill \
    the Replay buffer in the beggining of the training',
        type=int,
        default=100)
    parser.add_argument(
        '--batch-size', help='Batch size in each epoch', type=int, default=256)
    # parser.add_argument(
    #     '--use_automatic_entropy_tuning', help='Set True to automatically discover best alpha (temperature)', type=bool, default=True)
    parser.add_argument(
        '--env_reset_mode',
        help='How to sample the starting position of the agent',
        choices=(
            'Uniform',
            'Gaussian',
            'False',
            'Discretized_Uniform'),
        default='False')
    parser.add_argument(
        '--use_cuda',
        help='If the device is a GPU or CPU',
        default='False',
        choices=[
            'True',
            'False'])
    parser.add_argument(
        '--restore_path',
        help='Filename of the policy being loaded',
        type=str,
        default=None)
    parser.add_argument(
        '--save_path', help='Filename to save the current training', type=str)
    parser.add_argument(
        '--log_interval', help='Frequency of logging', type=int, default=100)
    parser.add_argument(
        '--save-interval', help='Frequency of saving', type=int, default=100)
    parser.add_argument(
        '--eval_interval',
        help='Frequency for evaluating deterministic policy',
        type=int,
        default=None)
    parser.add_argument(
        '--use_double', help='Flag to use float64', type=str, default=None)
    parser.add_argument(
        '--learning_rate', help='Learning rate', type=float, default=3e-4)
    parser.add_argument(
        '--reward_function',
        help='What reward function to use',
        default='Normal',
        type=str)
    parser.add_argument(
        '--seed', help='Global seed', default=42, type=int)
    parser.add_argument(
        '--state', help='Global seed', default='Old', type=str)
    parser.add_argument(
        '--same-norm', help='same_norm', default=False)
    parser.add_argument(
        '--threshold',
        help='Clipping the difference between action vectors',
        default=4.0,
        type=float)
    parser.add_argument(
        '--clip-action',
        help='Clipping the difference between action vectors',
        default=100,
        type=int)
    parser.add_argument(
        '--save_interval', help='Frequency of saving', type=int, default=100)
    return parser.parse_args()

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


class SAC():

    def __init__(
        self,
        env,
        replay_buffer_size,
        hidden_dim,
        restore_path,
        device,
        max_episodes,
        save_path,
        learning_rate=3e-4,
        use_double=True,
        min_num_steps_before_training=0,
        save_interval=100,
    ):
        """
        Soft Actor-Critic algorithm


        Parameters
        ----------
        env :
            The environment to be used
        replay_buffer_size : [int]
            Replay-buffer size
        hidden_dim : [int]
            Size of the hidden-layers in Q and V functions[description]
        restore_path : [str]
            File path to restore training
        device : [str or torch.device]
            'cpu' or 'gpu
        max_episodes : [int]
            Max number of episodes to train the policy
        save_path : [str]
            File path to save the networks
        learning_rate : [float], optional
            The learning rate for gradient based optimization, by default 3e-4
        use_double : bool, optional
            Use Float Tensor or Double Tensor, by default True
        min_num_steps_before_training : int, optional
            Number of steps to randomly sample before start acting, by default 0
        save_interval : int, optional
            The interval in epochs to save the models, by default 100
        """
        self.env = env

        self.save_path = save_path
        self.save_interval = save_interval
        self.min_num_steps_before_training = min_num_steps_before_training
        self.restore_path = restore_path
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_double = use_double
        # Network and env parameters
        self.action_dim = self.env.action_space.shape[0]
        try:
            self.state_dim = self.env.observation_space.shape[0]
        except BaseException:
            self.state_dim = self.env.observation_space
        # hidden_dim = args.net_size_value
        self.action_range = [self.env.agent.action_space.low.min(
        ), self.env.agent.action_space.high.max()]

        self._creating_models(replay_buffer_size, self.state_dim,
                              self.action_dim, self.device, self.hidden_dim)

        # Copying the data to the target networks
        for target_param, param in zip(
                self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        # Types of losses
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        # Learning rates
        self.value_lr = learning_rate
        self.soft_q_lr = learning_rate
        self.policy_lr = learning_rate
        # Optimizers
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=self.value_lr)
        self.soft_q_optimizer = optim.Adam(
            self.soft_q_net.parameters(), lr=self.soft_q_lr)
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.policy_lr)

        if self.use_double:
            self.state_to_tensor = lambda x: torch.DoubleTensor(
                x).unsqueeze(0).to(device)
        else:
            self.state_to_tensor = lambda x: torch.FloatTensor(
                x).unsqueeze(0).to(device)

    def soft_q_update(self,
                      batch_size,
                      gamma=0.99,
                      mean_lambda=1e-3,
                      std_lambda=1e-3,
                      z_lambda=0.0,
                      soft_tau=1e-2,
                      ):
        """
        SAC train update (Soft-Q)

        Parameters
        ----------
        batch_size : [int]
            Batch size
        gamma : float, optional
            Discount factor, by default 0.99
        mean_lambda : [float], optional
            coefficient for penalty on policy mean magnitude, by default 1e-3
        std_lambda : [float], optional
            coefficient for penalty on policy variance, by default 1e-3
        z_lambda : float, optional
            coefficient for penalty on policy mean before been squashed by tanh, by default 0
        soft_tau : [float], optional
            Soft coefficient to update target networks, by default 1e-2
        """
        # Sampling memmory batch
        state, action, reward, next_state, done = self.replay_buffer.sample(
            batch_size)

        # Broadcast
        if self.use_double:
            state = torch.DoubleTensor(state).to(self.device)
            next_state = torch.DoubleTensor(next_state).to(self.device)
            action = torch.DoubleTensor(action).to(self.device)
            reward = torch.DoubleTensor(reward).unsqueeze(1).to(self.device)
            done = torch.DoubleTensor(np.float64(
                done)).unsqueeze(1).to(self.device)

        else:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            done = torch.FloatTensor(np.float32(
                done)).unsqueeze(1).to(self.device)

        # Net forward-passes
        expected_q_value = self.soft_q_net(state, action)
        expected_value = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(
            state)

        ## Qf - loss
        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(
            expected_q_value, next_q_value.detach())

        ## Vf - loss
        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        # Policy Loss
        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()
        policy_loss += mean_loss + std_loss + z_loss

        # NN updates
        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.policy_optimizer.step()

        # Updating the target networks
        for target_param, param in zip(
                self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return(q_value_loss.item(), policy_loss.item(), value_loss.item())

    def __write_csv(
            self,
            episode,
            time_elapsed,
            frame_count,
            len_replay_buffer,
            episode_reward,
            value_loss,
            q_value_loss,
            policy_loss,
            step):
        """
        Writes data in csv
        """

        with open(os.path.join(self.save_path, 'progress.csv'), 'a') as csvfile:
            rew_writer = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            rew_writer.writerow([episode,
                                 time_elapsed,
                                 frame_count,
                                 len(self.replay_buffer),
                                 episode_reward,
                                 value_loss,
                                 q_value_loss,
                                 policy_loss,
                                 step])

    def __save_model(self, episode, frame_count):
        """
        Saves model pickles

        Parameters
        ----------
        episode : [int]
            Current episode
        frame_count : [int]
            Current timestep
        """
        save_state = {
            'episode': episode,
            'frame_count': frame_count,
            'value_net': self.value_net.state_dict(),
            'target_value_net': self.target_value_net.state_dict(),
            'soft_q_net': self.soft_q_net.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'soft_q_optimizer': self.soft_q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer
        }
        torch.save(save_state, self.save_path + '/state.pt')
        torch.save(save_state['policy_net'],
                   self.save_path + '/state.pt'[:-3] + '_policy.pt')
        print('saving model at = ', self.save_path)

    def _creating_models(
            self,
            buffer_size,
            state_dim,
            action_dim,
            device,
            hidden_dim):
        """
        Istantiating the networks and buffer
        """

        self.policy_net = PolicyNetwork(
            state_dim,
            action_dim,
            args.net_size_policy).to(device).type(
            torch.double)
        self.eval_policy = copy.deepcopy(self.policy_net)
        self.replay_buffer = utils.ReplayBuffer(buffer_size)
        self.value_net = ValueNetwork(
            state_dim, hidden_dim).to(device).type(torch.double)
        self.target_value_net = ValueNetwork(
            state_dim, hidden_dim).to(device).type(torch.double)
        self.soft_q_net = SoftQNetwork(
            state_dim, action_dim, hidden_dim).to(device).type(torch.double)

        if self.use_double:
            self.policy_net = self.policy_net.type(torch.double)
            self.target_value_net = self.target_value_net.type(torch.double)
            self.value_net = self.value_net.type(torch.double)
            self.replay_buffer.buffer = np.asarray(
                self.replay_buffer.buffer).astype(np.float64).tolist()
            self.soft_q_net = self.soft_q_net.type(torch.double)

    def train(self):
        """
        Trains a continuous policy by means of SAC
        """
        # Starting data
        episode = 0
        frame_count = 0
        max_episodes = args.max_episodes
        time_horizon = 250
        batch_size = args.batch_size
        # Load parameters from previous training if available
        utils.restore_data(self.restore_path)

        begin = time.time()
        while episode < max_episodes:

            if (episode % 50 == 0):  # Hack because of PyRep set position bug
                self.env.restart = True
                self.env.reset()
                self.env.restart = False

            state = self.env.reset()
            episode_reward = 0

            for step in range(time_horizon):
                if frame_count > self.min_num_steps_before_training:

                    action = self.policy_net.get_action(
                        self.state_to_tensor(state))  # .detach()
                    next_state, reward, done, self.env_info = self.env.step(
                        action * self.action_range[1])

                else:
                    action = np.random.sample(self.action_dim)
                    next_state, reward, done, self.env_info = self.env.step(
                        action * self.action_range[1])

                self.replay_buffer.push(
                    state, action, reward, next_state, done)

                if len(self.replay_buffer) > batch_size:
                    if (episode % args.num_steps_until_train) == 0:
                        for i in range(args.num_trains_per_step):
                            q_value_loss, policy_loss, value_loss = self.soft_q_update(
                                batch_size)

                state = next_state
                episode_reward += reward
                frame_count += 1

                if done:
                    break

            print("Episode = {0} | Reward = {1:.2f} | Lenght = {2:.2f}".format(
                episode, episode_reward, step))
            episode += 1

            # Saving
            if (episode % self.save_interval == 0) and (episode > 0):
                self.__save_model(episode, frame_count)

            time_elapsed = time.time() - begin
            if (episode % 100 == 0) and (episode > 0):
                print('Time elapsed so far = {0:.2f} seconds.'.format(
                    time_elapsed))

            # Logging
            if len(self.replay_buffer) > batch_size:
                self.__write_csv(episode,
                                 time_elapsed,
                                 frame_count,
                                 len(self.replay_buffer),
                                 episode_reward,
                                 value_loss,
                                 q_value_loss,
                                 policy_loss,
                                 step)


def main(args):

    env = DroneEnv(random=args.env_reset_mode, headless=True, seed=args.seed,
                   reward_function_name=args.reward_function, state=args.state)

    use_cuda = torch.cuda.is_available()

    if use_cuda and (args.use_cuda):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Set save/restore paths
    save_path = os.path.join('./checkpoint/', args.save_path) + '/'

    restore_path = args.restore_path or save_path
    report_folder = save_path  # save_path.split('/')[0] + '/'

    # Check if they exist
    utils.check_dir(save_path)
    if restore_path:
        utils.check_dir(restore_path)
    utils.check_dir(report_folder)

    # Preparing log csv
    if not os.path.isfile(os.path.join(report_folder, 'progress.csv')):
        print('There is no csv there')
        with open(os.path.join(report_folder, 'progress.csv'), 'w') as outcsv:
            writer = csv.writer(outcsv, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Episode", "Total time (s)", "Frame",
                             "Buffer_size",
                             "Mean_Reward",
                             "value_loss", "q_value_loss", "policy_loss",
                             "episode_lenght"])

        # Network and env parameters
    action_dim = env.action_space.shape[0]
    try:
        state_dim = env.observation_space.shape[0]
    except BaseException:
        state_dim = env.observation_space
    hidden_dim = args.net_size_value
    action_range = [env.agent.action_space.low.min(
    ), env.agent.action_space.high.max()]

    sac = SAC(
        env=env,
        replay_buffer_size=args.replay_buffer_size,
        hidden_dim=hidden_dim,
        restore_path=restore_path,
        device=device,
        save_path=save_path,
        learning_rate=args.learning_rate,
        max_episodes=args.max_episodes,
        use_double=args.use_double,
        save_interval=args.save_interval)

    sac.train()


if __name__ == "__main__":
    args = argparser()

    # Setting seed
    torch.manual_seed(args.seed)
    # random.seed(a = seed)
    np.random.seed(seed=args.seed)

    main(args)
