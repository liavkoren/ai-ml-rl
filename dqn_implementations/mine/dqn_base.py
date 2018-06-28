from collections import deque
import random

import attr


@attr.s
class Memory(deque):
    size = attr.ib()
    minibatch_size = attr.ib()

    def append(self, thing):
        if len(self) > self.size - 1:
            self.popleft()
        return super().append(thing)

    def sample(self):
        batch_size = min(len(self), self.minibatch_size)
        return random.sample(self, batch_size)


@attr.s
class DqnBaseClass:
    env = attr.ib()
    discount_rate = attr.ib(default=0.99)
    epsilon_max = attr.ib(default=1.0)
    epsilon_min = attr.ib(default=0.01)
    annealing_const = attr.ib(default=.001)  # aka Lambda
    minibatch_size = attr.ib(default=64)
    memory_size = attr.ib(default=int(1e6))
    num_episodes = attr.ib(default=1000)  # num of episodes in a training epoch
    num_hidden_units = attr.ib(default=64)
    render_every = attr.ib(default=20)  # set to zero to turn off rendering
    update_target_every = attr.ib(default=200)
    reward_clip_ceiling = attr.ib(None)
    reward_clip_floor = attr.ib(None)

    def __attrs_post_init__(self):
        self.steps = 0
        self.reset_memory()
        self.reset_data_recorders()
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n
        self.online_net = make_network(self.state_shape, self.action_shape, self.num_hidden_units)

    def reset_memory(self):
        self.memory = Memory(self.memory_size, self.minibatch_size)

    def reset_data_recorders(self):
        self.episode_rewards = []
        self.episode_losses = []
        self.td_errors = []
        self.online_net_q_values = []
        self.target_net_q_values = []
        self.w1_gradient = []
        self.w2_gradient = []

    def q_value_one(self, net, state):
        return net.predict(state.reshape((1, self.state_shape[0]))).flatten()

    def training_rewards_string(self, episode):
        last_ep = self.episode_rewards[-1]
        ten_ep_mean = sum(self.episode_rewards[-10:])/len(self.episode_rewards[-10:])
        hundred_ep_mean = sum(self.episode_rewards[-100:])/len(self.episode_rewards[-100:])
        return f'Ep: {episode} // steps: {self.steps} // last ep reward: {last_ep:.2f} // {min(10, len(self.episode_rewards[-10:]))}-ep mean: {ten_ep_mean:.2f} // {min(100, len(self.episode_rewards[-100:]))}-ep mean: {hundred_ep_mean:.2f}'

    def render(self, episode):
        if self.render_every and episode % self.render_every == 0:
            self.env.render()

    # TODO: rename to train
    def run(self):
        raise NotImplementedError

    def test(self, num_episodes=500):
        episode_done = False
        self.reset_data_recorders()
        self.reward_clip_ceiling = None
        self.reward_clip_floor = None
        for episode in range(num_episodes):
            episode_td_errors = 0
            episode_reward = 0
            episode_done = False
            state = self.env.reset()
            self.target_net_q_values.append(self.q_value_one(state))
            while not episode_done:
                action = self.q_value_one(self.online_net, state).argmax()
                self.render(episode)
                state_, reward, episode_done, _ = self.env.step(action)
                episode_reward += reward
                episode_td_errors += self.replay(target_net=self.online_net, online_net=self.online_net)
                state = state_
                if episode_done:
                    self.episode_rewards.append(episode_reward)
                    self.td_errors.append(episode_td_errors)
                    print(self.training_rewards_string(episode), end='\r', flush=True)
        self.env.close()
