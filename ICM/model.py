import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import activation


class StateEncoder(nn.Module):
    def __init__(self, s_dim, f_dim):
        super(StateEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, f_dim)
        )

    def forward(self, state):
        return self.model(state)


class ForwardModel(nn.Module):
    def __init__(self, f_dim, a_dim):
        super(ForwardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(f_dim + a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, f_dim)
        )

    def forward(self, state_encoded, action):
        x = torch.cat([state_encoded, action], dim=-1)
        x = self.model(x)
        return x


class InverseModel(nn.Module):
    def __init__(self, f_dim, a_dim):
        super(InverseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(f_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, a_dim)
        )

    def forward(self, state_encoded, next_state_encoded):
        x = torch.cat([state_encoded, next_state_encoded], dim=-1)
        action = self.model(x)
        return action


class Policy(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Policy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, a_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)


class QFunction(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(QFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim + a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        Q = self.model(x)
        return Q