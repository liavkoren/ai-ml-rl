"""
Leaf nodes store the transition priorities and the internal nodes are
intermediate sums, with the parent node containing the sum over all priorities,
p_total. This provides a efficient way of calculating the cumulative sum of
priorities, allowing O(log N ) updates and sampling. To sample a minibatch of
size k, the range [0, p_total] is divided equally into k ranges. Next, a value
is uniformly sampled from each range. Finally the transitions that correspond to
each of these sampled values are retrieved from the tree. Overhead is similar to
rank-based prioritization.
"""

from copy import copy
from datetime import datetime
import json
import math
import os
import pickle
import random

import attr
import keras
import gym
import numpy as np

from SumTree import SumTree


date_formater = '%Y-%m-%d__%H--%M--%S'  # eg: '2018-05-10__13--57--47'


@attr.s
class Memory:
    size = attr.ib()
    batch_size = attr.ib(default=64)
    epsilon = attr.ib(default=0.01)
    alpha = attr.ib(default=0.6)

    def __attrs_post_init__(self):
        self.sumtree = SumTree(capacity=self.size)

    def _td_error_to_priority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def store(self, td_error, transition):
        priority = self._td_error_to_priority(td_error)
        self.sumtree.add(priority, transition)

    def sample(self):
        # https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py#L102
        batch = []
        segment_size = self.sumtree.total() // self.batch_size

        for index in range(self.batch_size):
            a = segment_size * index
            b = segment_size * (index + 1)
            s = random.uniform(a, b)
            idx, p, data = self.sumtree.get(s)
            batch.append((idx, data))
        return batch

    def update(self, idx, td_error):
        priority = self._td_error_to_priority(td_error)
        self.sumtree.update(idx, priority)


def make_network(state_shape, action_shape):
    q_network = keras.Sequential()
    q_network.add(keras.layers.Dense(64, activation='relu', input_shape=state_shape))
    q_network.add(keras.layers.Dense(action_shape, activation='linear'))
    q_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    return q_network


@attr.s
class PrioritizedExperienceReplayDDQN:
    env_name = attr.ib()
    memory_size = attr.ib(default=int(1e5))
    save_every = attr.ib(default=500)
    render_every = attr.ib(default=10)
    num_episodes = attr.ib(default=1000)
    update_target_every = attr.ib(default=100)
    report_every = attr.ib(default=10)
    max_episode_len = attr.ib(default=700)
    batch_size = attr.ib(default=64)
    discount_rate = attr.ib(default=0.99)  # aka gamma, discount for future rewards.
    epsilon_max = attr.ib(default=1.0)  # probability that we take an exploratory, off-policy action
    epsilon_min = attr.ib(default=.01)
    annealing_const = attr.ib(default=.001)  # aka lambda, how quickly epsilon anneals down to min value
    data_directory = attr.ib('data')
    random_init_steps = attr.ib(default=1000)  # How many random transitions to observe before starting to learn.

    def __attrs_post_init__(self):
        self.episode_rewards = []
        self.episode_traces = []
        self.target_q_history = []
        self.online_q_history = []
        self.mean_td_errors = []
        self.episode_lengths = []
        self.steps = 0
        self.env = gym.make(self.env_name)
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n
        self.online_net = make_network(self.state_shape, self.action_shape)
        self.target_net = make_network(self.state_shape, self.action_shape)
        self.target_net.set_weights(self.online_net.get_weights())
        self.save_dir = None
        self.init_memory()
        if self.save_every:
            self._save_self()
            now = datetime.now()
            self.data_file_name = self.env_name + now.strftime(date_formater) + 'history.data'

    def _save_self(self):
        """ Saves a copy of the Agent at startup for posterity. """
        now = datetime.now()
        data_dir = os.path.join(self.data_directory, now.strftime(date_formater), 'PER')
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        self.save_dir = data_dir
        data = copy(self.__dict__)
        data.pop('memory')
        data.pop('env')

        file_name = os.path.join(self.save_dir, self.env_name + '_agent.params')
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
        print(f'Saved network to {file_name}')

    def _save_data(self):
        """ Save networks, episode_rewards, episode_traces, target_q_history, online_q_history. """
        self.online_net.save(os.path.join(self.save_dir, self.env_name + '_online-net.ht'))
        self.target_net.save(os.path.join(self.save_dir, self.env_name + '_target-net.ht'))
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

    def init_memory(self):
        self.memory = Memory(size=self.memory_size, batch_size=self.batch_size)
        if not self.random_init_steps:
            return
        print('Starting to initializing memory with random agent.')
        state = self.env.reset()
        for _ in range(self.random_init_steps):
            action = random.randint(0, self.action_shape-1)
            state_, reward, episode_done, _ = self.env.step(action)

            # See https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
            # for discussion of random init and rewards as naive error signal
            self.memory.store(reward, (state, action, abs(reward), state_, episode_done))
            if episode_done:
                state = self.env.reset()
            else:
                state = state_
        print('Finished initializing memory with random agent.')

    def replay(self, verbose=0):
        batch = self.memory.sample()  # A list of (idx, observation) tuples
        observations = [b[1] for b in batch]

        # unpack all the replay memories into arrays:
        # transition is: (state, action, reward, state', is_done)
        states = np.array([transition[0] for transition in observations])  # (observations x state-size)
        transition_actions = np.array([transition[1] for transition in observations])
        rewards = np.array([transition[2] for transition in observations])  # (observations x 1)
        terminal_mask = np.array([True if transition[3] is None else False for transition in observations])  # (observations x 1)
        terminal_state = np.zeros(self.state_shape)
        states_ = np.array([transition[3] if transition[3] is not None else terminal_state for transition in observations])  # (observations x state-size)

        # set y = r for terminal states:
        terminal_state_actions = transition_actions[terminal_mask]
        y = self.target_net.predict(states)  # (observations x num-actions)
        y[terminal_mask, terminal_state_actions] = rewards[terminal_mask]

        # DDQN update:
        # set y = r + gamma * Q_hat(s', argmax Q(s', a')). Remember that y_ is the output of Q_hat, aka target_net.
        non_terminal_mask = ~terminal_mask
        online_predicted_actions_ = self.online_net.predict(states_).argmax(axis=1)  # observations x num-action
        best_actions = online_predicted_actions_[non_terminal_mask]
        non_terminal_actions = transition_actions[non_terminal_mask]
        y_ = self.target_net.predict(states_)  # (observations x num-actions)
        y[non_terminal_mask, non_terminal_actions] = rewards[non_terminal_mask] + self.discount_rate * y_[non_terminal_mask, best_actions]
        self.online_net.fit(states, y, batch_size=self.memory.batch_size, epochs=1, verbose=verbose)  # REMEBER, Q is a func from (state, action) pairs to values.

        memory_indices = [b[0] for b in batch]
        target_predictions = self.target_net.predict(states_)[range(self.batch_size), online_predicted_actions_]
        online_predictions = self.online_net.predict(states)[range(self.batch_size), transition_actions]

        td_errors = rewards + self.discount_rate * target_predictions - online_predictions  # should be (batch x 1)
        self.mean_td_errors.append(td_errors.mean())
        td_errors = np.abs(td_errors)
        for idx, error in zip(memory_indices, td_errors):
            self.memory.update(idx, error)

    def Q_val_one(self, net, state):
        """ Given a net and a single state, predict values for the state. """
        return net.predict(state.reshape((1, self.state_shape[0]))).flatten()

    def run(self):
        for episode_count in range(self.num_episodes):
            episode_done = False
            episode_reward = 0
            state = self.env.reset()
            self.target_q_history.append(self.Q_val_one(self.target_net, state).max())
            self.online_q_history.append(self.Q_val_one(self.online_net, state).max())

            if self.save_every and self.steps % self.save_every == 0:
                self._save_data()

            episode_len = 0
            while not episode_done:
                episode_len += 1
                EPSILON = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.annealing_const * self.steps)
                self.steps += 1

                if random.random() < EPSILON:
                    action = random.randint(0, self.action_shape-1)
                else:
                    action = self.Q_val_one(self.online_net, state).argmax()

                state_, reward, episode_done, _ = self.env.step(action)
                episode_reward += reward

                if self.render_every and episode_count % self.render_every == 0:
                    self.env.render()
                if episode_done:
                    state_ = None

                self.memory.store(abs(reward), (state, action, reward, state_, episode_done))
                self.replay()
                state = state_

                if self.steps % self.update_target_every == 0:
                    self.target_net.set_weights(self.online_net.get_weights())
                if episode_len > self.max_episode_len:
                    episode_done = True

                if episode_done:
                    if self.report_every and episode_count % self.report_every == 0:
                        mean_10 = sum(self.episode_rewards[-10:]) / 10
                        mean_100 = sum(self.episode_rewards[-100:]) / 100
                        print(f'Episode: {episode_count}, steps: {self.steps}, reward: {episode_reward}, 10-episode mean reward: {mean_10}, 100-episode mean reward: {mean_100} ', end='\r', flush=True)
                    self.episode_rewards.append(episode_reward)
        if self.save_every:
            self._save_data()


if __name__ == '__main__':
    ddqn = PrioritizedExperienceReplayDDQN(env_name='CartPole-v1')
    ddqn.run()
