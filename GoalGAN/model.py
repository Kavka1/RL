import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np


class GaussianPolicy(nn.Module):
    def __init__(self, o_dim, g_dim, a_dim, action_var, device):
        super(GaussianPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(o_dim + g_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, a_dim),
            nn.Tanh()
        )
        self.action_var = torch.full(size=[a_dim, ], fill_value=action_var).to(device)
        self.cov_mat = torch.diag(self.action_var)

    def forward(self, o_g_input):
        a_mean = self.model(o_g_input)
        dist = MultivariateNormal(loc=a_mean, covariance_matrix=self.cov_mat)
        return dist


class VFunction(nn.Module):
    def __init__(self, o_dim, g_dim):
        super(VFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(o_dim + g_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, o_g_input):
        return self.model(o_g_input)


class QFunction(nn.Module):
    def __init__(self, o_dim, g_dim, a_dim):
        super(QFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(o_dim + g_dim + a_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, o_g_a_input):
        return self.model(o_g_a_input)


class GoalGenerator(nn.Module):
    def __init__(self, noise_dim, g_dim):
        super(GoalGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, g_dim)
        )

    def forward(self, noise):
        return self.model(noise)


class GoalDiscriminator(nn.Module):
    def __init__(self, g_dim):
        super(GoalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(g_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, goal):
        return self.model(goal)