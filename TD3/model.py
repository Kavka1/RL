import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as Normal
import numpy as np

class QFunction(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_units = [128, 64]):
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

class DeterministicPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, device, hidden_units=[128, 64], action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.device = device
        self.action_space = action_space

        self.model = nn.Sequential(
            nn.Linear(s_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], a_dim),
            nn.Tanh()
        )

        if self.action_space is not None:
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.).to(self.device)
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(self.device)
        else:
            self.action_bias = torch.FloatTensor(0.).to(self.device)
            self.action_scale = torch.FloatTensor(1.).to(self.device)

    def sample(self, s):
        pi = self.model(s)
        return pi * self.action_scale + self.action_bias