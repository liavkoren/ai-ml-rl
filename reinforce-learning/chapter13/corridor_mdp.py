import attr
from matplotlib import pyplot as plt
import numpy as np


LEFT = 0
RIGHT = 1
EPISODES_PER_TRIAL = 1000

@attr.s
class Corridor:
    def __attrs_post_init__(self):
        self.state = 0
        self.is_done = False

    def reset(self):
        self.__attrs_post_init__()

    def __repr__(self):
        return f'Corridor(state={self.state})'

    def step(self, action):
        assert action in [LEFT, RIGHT], 'Invalid action'
        assert not self.is_done, 'Reset the environment'
        self.is_done = False
        reward = -1
        is_start_state = self.state == 0
        is_switched_action_state = self.state == 1

        if is_start_state:
            if action == RIGHT:
                self.state = 1
        elif is_switched_action_state:
            # switchero state:
            if action == RIGHT:
                self.state = 0
            else:
                self.state = 2
        else:
            if action == RIGHT:
                self.state = 3
                reward = 1
                self.is_done = True
            else:
                self.state = 1

        return reward, self.is_done


@attr.s
class EpsilonAgent:
    direction = attr.ib(default=LEFT)

    def __attrs_post_init__(self):
        self.rewards = {}
        self.epsilon = 0.1
        self.env = Corridor()
        self.max = float('-inf')
        self.argmax = -1

    def get_action(self):
        # 1 - epsilon is the prob of going left
        if np.random.random() < self.epsilon:
            return np.random.choice([LEFT, RIGHT], 1)[0]
        return self.direction

    def run_episode(self):
        episode_reward = 0
        self.env.reset()
        is_done = False
        while not is_done:
            action = self.get_action()
            reward, is_done = self.env.step(action)
            episode_reward += reward
        return episode_reward

    def get_episode_trace(self):
        states = []
        actions = []
        rewards = []
        self.env.reset()
        is_done = False
        while not is_done:
            action = self.get_action()
            reward, is_done = self.env.step(action)
            states.append(self.env.state)
            actions.append(action)
            rewards.append(reward)
            if len(states) == 100:
                is_done = True
                rewards[-1] = -10
        return states, actions, rewards

    def trials(self):
        for idx in range(5, 96):
            self.epsilon = idx / 100
            cumulative_rewards = 0
            if idx > 50:
                self.epsilon = (100 - idx) / 100
                self.direction = RIGHT
            for episode in range(EPISODES_PER_TRIAL):
                cumulative_rewards += self.run_episode()
                if cumulative_rewards / EPISODES_PER_TRIAL > self.max:
                    self.max = cumulative_rewards / EPISODES_PER_TRIAL
                    self.argmax = idx
                print(f'idx: {idx / 100}, eps: {self.epsilon}, episode: {episode}, cum reward: {cumulative_rewards}, av reward: {cumulative_rewards / EPISODES_PER_TRIAL}', end='\r', flush=True)
            self.rewards[idx / 100] = cumulative_rewards / EPISODES_PER_TRIAL
        return self.rewards


def discounted_return(rewards, discount_rate=0.9):
    total = 0
    for idx, reward in enumerate(rewards):
        total += discount_rate ** idx * reward
    return total


def reinforce(policy, step_size=0.0001, num_episodes=1000, discount_rate=0.9):
    for episode in range(num_episodes):
        states, actions, rewards = policy.get_episode_trace()
        episode_reward = sum(rewards)
        for idx, action in enumerate(actions):
            policy.epsilon += step_size * (discount_rate ** idx) * episode_reward
    return policy


def plot_dynamics():
    agent = EpsilonAgent()
    data = agent.trials()
    print(data)
    print(agent.max, agent.argmax)
    x = data.keys()
    y = data.values()
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    raw_policy = EpsilonAgent()
    raw_policy.epsilon = np.random.random()

    good_policy = reinforce(raw_policy)
    print(good_policy.epsilon)
