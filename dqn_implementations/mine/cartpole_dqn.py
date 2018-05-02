"""
based on https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
https://github.com/jaara/AI-blog/blob/master/CartPole-basic.py
"""

from collections import deque
import json
import random

import gym
import keras
import numpy as np


class Brain:
    def __init__(self):
        self.model = self._create_model()

    def _create_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(output_dim=64, activation='relu', input_dim=STATE_SIZE))
        model.add(keras.layers.Dense(output_dim=ACTION_COUNT, activation='linear'))
        opt = keras.optimizers.Adam()
        model.compile(loss='mse', optimizer=opt)
        return model

    def predict(self, state):
        # Returns an array of ACTION_SIZE, with the Q-val of each action.
        return self.model.predict(state)

    def predictOne(self, state):
        return self.predict(state.reshape(1, STATE_SIZE)).flatten()

    def train(self, x, y):
        self.model.fit(x, y, batch_size=64, nb_epoch=1, verbose=0)
        # is batch_size diff from len(x)/len(y) ?
        # YEP


class Memory(deque):
    def __init__(self, size):
        self.size = size

    def add(self, thing):
        if len(self) > self.size - 1:
            self.popleft()
        return super().append(thing)

    def sample(self, batch_size):
        batch_size = min(len(self), batch_size)
        return random.sample(self, batch_size)


MEMORY_SIZE = 1e5
BATCH_SIZE = 64
GAMMA = .99
EPSILON_MAX = 1.0
EPSILON_MIN = .01
EPSILON = EPSILON_MAX
STATE_SIZE = None
ACTION_COUNT = None
LAMBDA = .001


class Agent:
    def __init__(self):
        self.brain = Brain()
        self.memory = Memory(MEMORY_SIZE)
        self.time = 0
        self.epsilon = EPSILON_MAX

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_COUNT-1)
        else:
            return np.argmax(self.brain.predictOne(state))

    def observe(self, sample):
        self.memory.add(sample)
        self.epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * 2.718 ** (-LAMBDA * self.time)
        self.time += 1

    def replay(self):
            batch = self.memory.sample(BATCH_SIZE)
            terminal_state = np.zeros(STATE_SIZE)

            states = np.array([tup[0] for tup in batch])
            predict = self.brain.predict(states)  # (batch_size x action_count)

            states_ = np.array([tup[3] if tup[3] is not None else terminal_state for tup in batch])
            predict_ = self.brain.predict(states_)  # (batch_size x action_count)

            # why... SHOULD a Q Network converge at all???
            x = np.zeros((len(batch), STATE_SIZE))
            # does y have to be a vector, or can it be just the max q-val?
            y = np.zeros((len(batch), ACTION_COUNT))
            for idx, sample in enumerate(batch):
                state, action, reward, state_, is_done = sample
                target = predict[idx]
                if state_ is None:
                    target[action] = reward
                else:
                    target[action] = reward + GAMMA * np.amax(predict_[idx])
                x[idx] = state
                y[idx] = target

            self.brain.train(x, y)


class Environment:
    def __init__(self, name):
        self.env = gym.make(name)
        global STATE_SIZE, ACTION_COUNT
        STATE_SIZE = self.env.observation_space.shape[0]
        ACTION_COUNT = self.env.action_space.n

    def run_episode(self, agent, render=True):
        state = self.env.reset()
        total_reward = 0
        while True:
            if render:
                self.env.render()
            action = agent.act(state)
            state_, reward, is_done, info = self.env.step(action)
            if is_done:
                state_ = None
            agent.observe((state, action, reward, state_, is_done))
            agent.replay()
            state = state_
            total_reward += reward
            if is_done:
                print('reward for episode: %d' % total_reward)
                return total_reward


if __name__ == '__main__':
    PROBLEM = 'CartPole-v0'
    env = Environment(PROBLEM)
    agent = Agent()
    time = 0
    reward_history = []
    while True:
        reward_history.append(env.run_episode(agent, render=True))
        time += 1
        if time % 10 == 0:
            with open('reward_hisory', 'w') as file:
                file.write(json.dumps(reward_history))
        if time % 10 == 0:
            print(time)
