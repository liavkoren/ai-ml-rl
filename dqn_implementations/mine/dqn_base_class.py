from copy import copy
from datetime import datetime
import json
import os
import pickle

import attr
import gym
import numpy as np


date_formater = '%Y-%m-%d__%H--%M--%S'  # eg: '2018-05-10__13--57--47'


@attr.s
class AgentBaseClass:
    env_name = attr.ib()

    memory_size = attr.ib(default=int(1e5))
    batch_size = attr.ib(default=64)
    save_every = attr.ib(default=500)
    render_every = attr.ib(default=10)
    num_episodes = attr.ib(default=1000)
    update_target_every = attr.ib(default=100)
    report_every = attr.ib(default=10)
    max_episode_len = attr.ib(default=700)
    discount_rate = attr.ib(default=0.99)  # aka gamma, discount for future rewards.
    epsilon_max = attr.ib(default=1.0)  # probability that we take an exploratory, off-policy action
    epsilon_min = attr.ib(default=.01)
    annealing_const = attr.ib(default=.001)  # aka lambda, how quickly epsilon anneals down to min value
    data_directory = attr.ib('data')
    random_init_steps = attr.ib(default=1000)
    MemoryClass = attr.ib()
    online_net = attr.ib()
    target_net = attr.ib()

    def __attrs_post_init__(self):
        self.episode_rewards = []
        self.episode_traces = []
        self.target_q_history = []
        self.online_q_history = []
        self.mean_td_errors = []
        self.transitions_per_episode = []
        self.steps = 0
        self.env = gym.make(self.env_name)
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n
        self.target_net.set_weights(self.online_net.get_weights())
        self.save_dir = None
        self.init_memory()
        if self.save_every:
            self._create_save_dir()
            self._save_self()
            now = datetime.now()
            self.data_file_name = self.env_name + now.strftime(date_formater) + 'history.data'

    def init_memory(self):
        self.memory = self.MemoryClass(self.memory_size, self.batch_size)
        if not self.random_init_steps:
            return

    def Q_val_one(self, net, state):
        """ Given a net and a single state, predict values for the state. """
        return net.predict(state.reshape((1, self.state_shape[0]))).flatten()

    def _create_save_dir(self):
        now = datetime.now()
        data_dir = os.path.join(self.data_directory, now.strftime(date_formater))
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        self.save_dir = os.path.join(data_dir, self.env_name)

    def _save_self(self):
        """ Saves a copy of the Agent at startup for posterity. """
        data = copy(self.__dict__)
        data.pop('memory')
        data.pop('env')

        file_name = os.path.join(self.save_dir, 'agent.params')
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
        print(f'Saved network to {file_name}')

    def _save_data(self):
        """ Save networks, episode_rewards, episode_traces, target_q_history, online_q_history. """
        self.online_net.save(os.path.join(self.save_dir, 'online-net.ht'))
        self.target_net.save(os.path.join(self.save_dir, 'target-net.ht'))
        file_name = os.path.join(self.save_dir, 'history.data')
        with open(file_name, 'w+') as file:
            data = {
                'episode_rewards': np.array(self.episode_rewards).tolist(),
                'episode_traces': np.array(self.episode_traces).tolist(),
                'target_q_history': np.array(self.target_q_history).tolist(),
                'online_q_history': np.array(self.online_q_history).tolist(),
                'mean_td_errors': np.array(self.mean_td_errors).tolist(),
                'episode_lengths': np.array(self.episode_lengths).tolist(),
                'time': datetime.now().strftime(date_formater)
            }
            json.dump(data, file)
        print(f'Saved data to {file_name}')
