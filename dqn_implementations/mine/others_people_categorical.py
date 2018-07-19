"""
Three other people's implementations of Distributional algorithm

https://github.com/floringogianu/categorical-dqn/blob/master/policy_improvement/categorical_update.py
https://github.com/higgsfield/RL-Adventure/blob/master/6.categorical%20dqn.ipynb
https://github.com/hengyuan-hu/rainbow
"""

import copy

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


# https://github.com/floringogianu/categorical-dqn/blob/master/policy_improvement/categorical_update.py
class CategoricalPolicyImprovement(object):
    """ Deep Q-Learning training method. """

    def __init__(self, policy, target_policy, lr=0.00025, discount=0.95, v_min=-10, v_max=10, atoms_no=51, batch_size=32):
        self.name = 'Categorical-PI'
        self.policy = policy
        self.target_policy = target_policy
        self.lr = lr
        self.gamma = discount

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer.zero_grad()

        self.dtype = torch.float
        self.v_min, self.v_max = v_min, v_max
        self.atoms_no = atoms_no
        self.support = torch.linspace(v_min, v_max, atoms_no)
        self.support = self.support.type(self.dtype)
        self.delta_z = (v_max - v_min) / (atoms_no - 1)
        self.m = torch.zeros(batch_size, self.atoms_no).type(self.dtype)

    def accumulate_gradient(self, batch_sz, states, actions, rewards,
                            next_states, mask):
        """ Compute the difference between the return distributions of Q(s,a)
            and TQ(s_,a).
        """
        states = Variable(states)
        actions = Variable(actions)
        next_states = Variable(next_states, volatile=True)

        # Compute probabilities of Q(s,a*)
        q_probs = self.policy(states)
        actions = actions.view(batch_sz, 1, 1)
        action_mask = actions.expand(batch_sz, 1, self.atoms_no)
        qa_probs = q_probs.gather(1, action_mask).squeeze()

        # Compute distribution of Q(s_,a)
        target_qa_probs = self._get_categorical(next_states, rewards, mask)

        # Compute the cross-entropy of phi(TZ(x_,a)) || Z(x,a)
        qa_probs = qa_probs.clamp(min=1e-3)  # Tudor's trick for avoiding nans
        loss = - torch.sum(target_qa_probs * torch.log(qa_probs))

        # Accumulate gradients
        loss.backward()

    def update_model(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _get_categorical(self, next_states, rewards, mask):
        batch_sz = next_states.size(0)
        gamma = self.gamma

        # Compute probabilities p(x, a)
        probs = self.target_policy(next_states).data
        qs = torch.mul(probs, self.support.expand_as(probs))
        argmax_a = qs.sum(2).max(1)[1].unsqueeze(1).unsqueeze(1)
        action_mask = argmax_a.expand(batch_sz, 1, self.atoms_no)
        qa_probs = probs.gather(1, action_mask).squeeze()

        # Mask gamma and reshape it torgether with rewards to fit p(x,a).
        rewards = rewards.expand_as(qa_probs)
        gamma = (mask.float() * gamma).expand_as(qa_probs)

        # Compute projection of the application of the Bellman operator.
        bellman_op = rewards + gamma * self.support.unsqueeze(0).expand_as(rewards)
        bellman_op = torch.clamp(bellman_op, self.v_min, self.v_max)

        # Compute categorical indices for distributing the probability
        m = self.m.fill_(0)
        b = (bellman_op - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms_no - 1)) * (l == u)] += 1

        # Distribute probability
        """
        for i in range(batch_sz):
            for j in range(self.atoms_no):
                uidx = u[i][j]
                lidx = l[i][j]
                m[i][lidx] = m[i][lidx] + qa_probs[i][j] * (uidx - b[i][j])
                m[i][uidx] = m[i][uidx] + qa_probs[i][j] * (b[i][j] - lidx)
        for i in range(batch_sz):
            m[i].index_add_(0, l[i], qa_probs[i] * (u[i].float() - b[i]))
            m[i].index_add_(0, u[i], qa_probs[i] * (b[i] - l[i].float()))
        """
        # Optimized by https://github.com/tudor-berariu
        offset = torch.linspace(0, ((batch_sz - 1) * self.atoms_no), batch_sz)\
            .type(torch.long)\
            .unsqueeze(1).expand(batch_sz, self.atoms_no)

        m.view(-1).index_add_(0, (l + offset).view(-1),
                              (qa_probs * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1),
                              (qa_probs * (b - l.float())).view(-1))
        return Variable(m)

    def update_target_net(self):
        """ Update the target net with the parameters in the online model."""
        self.target_policy.load_state_dict(self.policy.state_dict())

    def get_model_stats(self):
        param_abs_mean = 0
        grad_abs_mean = 0
        t_param_abs_mean = 0
        n_params = 0
        for p in self.policy.parameters():
            param_abs_mean += p.data.abs().sum()
            grad_abs_mean += p.grad.data.abs().sum()
            n_params += p.data.nelement()
        for t in self.target_policy.parameters():
            t_param_abs_mean += t.data.abs().sum()

        print("Wm: %.9f | Gm: %.9f | Tm: %.9f" % (
            param_abs_mean / n_params,
            grad_abs_mean / n_params,
            t_param_abs_mean / n_params))

# ----------------------------------
# https://github.com/higgsfield/RL-Adventure/blob/master/6.categorical%20dqn.ipynb


# these are mini-batch sized tensors of nexts, rewards, dones:
def projection_distribution(target_model, next_state, rewards, dones, discount):
    batch_size = next_state.size(0)
    Vmax = target_model.vmax
    Vmin = target_model.vmin
    num_atoms = target_model.num_atoms
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms)

    next_dist = target_model(next_state).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist = next_dist.gather(1, next_action).squeeze(1)

    rewards = rewards.unsqueeze(1).expand_as(next_dist)
    dones = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    Tz = rewards + (1 - dones) * discount * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b = (Tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    offset = torch.linspace(
        0,  # start
        (batch_size - 1) * num_atoms,  # end
        batch_size  # steps
    ).long(  # cast to int
    ).unsqueeze(1  # "new tensor with a dim of size one inserted at the specified position."
                   # Basically turns a row vect into a col vect.
                   # (batch_size, 1): [[0], [51], [102], [153]]
    ).expand(batch_size, num_atoms)  # (batch_size, 1) -> (batch_size, num_atoms), copying values.
    proj_dist = torch.zeros(next_dist.size())
    proj_dist.view(-1
        ).index_add_(
        0,
        (l + offset).view(-1),
        (next_dist * (u.float() - b)).view(-1)
    )
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    return proj_dist  # this is `m` in the paper.


# ----------------------------------
# https://github.com/hengyuan-hu/rainbow
EPS = 1e-7


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def one_hot(x, n):
    assert x.dim() == 2
    # one_hot_x = torch.zeros(x.size(0), n).cuda()
    one_hot_x = torch.zeros(x.size(0), n)
    one_hot_x.scatter_(1, x, 1)
    return one_hot_x


class DQNAgent(object):
    def __init__(self, q_net, double_dqn, num_actions):
        self.online_q_net = q_net
        self.target_q_net = copy.deepcopy(q_net)
        self.double_dqn = double_dqn
        self.num_actions = num_actions

    def save_q_net(self, prefix):
        torch.save(self.online_q_net.state_dict(), prefix+'online_q_net.pth')

    def parameters(self):
        return self.online_q_net.parameters()

    def sync_target(self):
        self.target_q_net = copy.deepcopy(self.online_q_net)

    def target_q_values(self, states):
        q_vals = self.target_q_net(Variable(states, volatile=True)).data
        return q_vals

    def online_q_values(self, states):
        q_vals = self.online_q_net(Variable(states, volatile=True)).data
        return q_vals

    def compute_targets(self, rewards, next_states, non_ends, gamma):
        """Compute batch of targets for dqn
        params:
            rewards: Tensor [batch]
            next_states: Tensor [batch, channel, w, h]
            non_ends: Tensor [batch]
            gamma: float
        """
        next_q_vals = self.target_q_values(next_states)

        if self.double_dqn:
            next_actions = self.online_q_values(next_states).max(1, True)[1]
            next_actions = one_hot(next_actions, self.num_actions)
            next_qs = (next_q_vals * next_actions).sum(1)
        else:
            next_qs = next_q_vals.max(1)[0] # max returns a pair

        targets = rewards + gamma * next_qs * non_ends
        return targets

    def loss(self, states, actions, targets):
        """
        params:
            states: Variable [batch, channel, w, h]
            actions: Variable [batch, num_actions] one hot encoding
            targets: Variable [batch]
        """
        assert_eq(actions.size(1), self.num_actions)

        qs = self.online_q_net(states)
        preds = (qs * actions).sum(1)
        err = nn.functional.smooth_l1_loss(preds, targets)
        return err


class DistributionalDQNAgent(DQNAgent):
    def __init__(self, q_net, double_dqn, num_actions, num_atoms, vmin, vmax):
        super(DistributionalDQNAgent, self).__init__(q_net, double_dqn, num_actions)

        self.num_atoms = num_atoms
        self.vmin = float(vmin)
        self.vmax = float(vmax)

        self.delta_z = (self.vmax - self.vmin) / (num_atoms - 1)

        zpoints = np.linspace(vmin, vmax, num_atoms).astype(np.float32)
        # self.zpoints = Variable(torch.from_numpy(zpoints).unsqueeze(0)).cuda()
        self.zpoints = Variable(torch.from_numpy(zpoints).unsqueeze(0))

    def _q_values(self, q_net, states):
        """internal function to compute q_value
        params:
            q_net: self.online_q_net or self.target_q_net
            states: Variable [batch, channel, w, h]
        """
        probs = q_net(states) # [batch, num_actions, num_atoms]
        q_vals = (probs * self.zpoints).sum(2)
        return q_vals, probs

    def target_q_values(self, states):
        states = Variable(states, volatile=True)
        q_vals, _ = self._q_values(self.target_q_net, states)
        return q_vals.data

    def online_q_values(self, states):
        states = Variable(states, volatile=True)
        q_vals, _ = self._q_values(self.online_q_net, states)
        return q_vals.data

    def compute_targets(self, rewards, next_states, non_ends, gamma):
        """Compute batch of targets for distributional dqn
        params:
            rewards: Tensor [batch, 1]
            next_states: Tensor [batch, channel, w, h]
            non_ends: Tensor [batch, 1]
            gamma: float
        """
        assert not self.double_dqn, 'not supported yet'

        # get next distribution
        next_states = Variable(next_states, volatile=True)
        # [batch, num_actions], [batch, num_actions, num_atoms]
        next_q_vals, next_probs = self._q_values(self.target_q_net, next_states)
        next_actions = next_q_vals.data.max(1, True)[1]  # [batch, 1]
        next_actions = one_hot(next_actions, self.num_actions).unsqueeze(2)
        next_greedy_probs = (next_actions * next_probs.data).sum(1)

        # transform the distribution
        rewards = rewards.unsqueeze(1)
        non_ends = non_ends.unsqueeze(1)
        proj_zpoints = rewards + gamma * non_ends * self.zpoints.data
        proj_zpoints.clamp_(self.vmin, self.vmax)

        # project onto shared support
        b = (proj_zpoints - self.vmin) / self.delta_z
        lower = b.floor()
        upper = b.ceil()
        # handle corner case where b is integer
        eq = (upper == lower).float()
        lower -= eq
        lt0 = (lower < 0).float()
        lower += lt0
        upper += lt0

        # note: it's faster to do the following on cpu
        ml = (next_greedy_probs * (upper - b)).cpu().numpy()
        mu = (next_greedy_probs * (b - lower)).cpu().numpy()

        lower = lower.cpu().numpy().astype(np.int32)
        upper = upper.cpu().numpy().astype(np.int32)

        batch_size = rewards.size(0)
        mass = np.zeros((batch_size, self.num_atoms), dtype=np.float32)
        brange = range(batch_size)
        for i in range(self.num_atoms):
            mass[brange, lower[brange, i]] += ml[brange, i]
            mass[brange, upper[brange, i]] += mu[brange, i]

        return torch.from_numpy(mass)
        # return torch.from_numpy(mass).cuda()

    def loss(self, states, actions, targets):
        """
        params:
            states: Variable [batch, channel, w, h]
            actions: Variable [batch, num_actions] one hot encoding
            targets: Variable [batch, num_atoms]
        """
        assert_eq(actions.size(1), self.num_actions)

        actions = actions.unsqueeze(2)
        probs = self.online_q_net(states)  # [batch, num_actions, num_atoms]
        probs = (probs * actions).sum(1)  # [batch, num_atoms]
        xent = -(targets * torch.log(probs.clamp(min=EPS))).sum(1)
        xent = xent.mean(0)
        return xent
