import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as Categorical
from torch.distributions import Normal
import numpy as np


class DiscretPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_unit = [128, 64]):
        super(DiscretPolicy, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.model = nn.Sequential(
            nn.Linear(s_dim, hidden_unit[0]),
            nn.ReLU(),
            nn.Linear(hidden_unit[0], hidden_unit[1]),
            nn.ReLU(),
            nn.Linear(hidden_unit[1], a_dim),
            nn.Softmax()
        )

    def forward(self, s):
        logits = self.model(s)
        dist = Categorical.Categorical(logits)
        return dist, dist.sample()


class GaussianPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, device, action_space = None, hidden_units = [128, 64], std = None, log_std_min = -20, log_std_max = 2):
        super(GaussianPolicy, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.std = std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        self.PreProcessor = nn.Sequential(
            nn.Linear(s_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU()
        )
        self.mean_fc = nn.Linear(hidden_units[-1], a_dim)
        if self.std == None:
            self.log_std_fc = nn.Linear(hidden_units[-1], a_dim)
        else:
            self.log_std = np.log(self.std)
            assert self.log_std_min <= self.log_std <= self.log_std_max

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)
        self.action_bias = self.action_bias.to(self.device)
        self.action_scale = self.action_scale.to(self.device)

    def forward(self, s):
        preprocess = self.PreProcessor(s)
        mean = self.mean_fc(preprocess)
        if self.std == None:
            log_std = self.log_std_fc(preprocess)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
        else:
            std = self.std

        return mean, std

    def sample(self, s):
        mean, std = self.forward(s)
        dist = Normal(mean, std)
        x_t = dist.sample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = dist.log_prob(x_t)
        #enforcing action bound
        #log_prob -= torch.log(self.action_scale * (1 - torch.pow(y_t, 2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


class QFunction(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_units = [64, 32]):
        super(QFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], 1)
        )

    def forward(self, s, a):
        return self.model(torch.cat([s, a], dim=1))

class VFunction(nn.Module):
    def __init__(self, s_dim, hidden_units = [64, 32]):
        super(VFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], 1)
        )

    def forward(self, s):
        return self.model(s)


if __name__ == '__main__':
    policy = DiscretPolicy(32, 5)
    s = torch.ones([10, 32])
    print(policy(s))

    policy_gaussian = GaussianPolicy(32, 5, device=torch.device('cpu'), std=None)
    print(policy_gaussian(s), '\n', policy_gaussian.sample(s))
