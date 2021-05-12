import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Policy(nn.Module):
    def __init__(self, o_dim, a_dim, g_dim):
        super(Policy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(o_dim + g_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, a_dim),
            nn.Tanh()
        )

    def forward(self, o_g):
        return self.model(o_g)


class QFunction(nn.Module):
    def __init__(self, o_dim, a_dim, g_dim):
        super(QFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(o_dim + a_dim + g_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, o_a_g):
        return self.model(o_a_g)


if __name__ == '__main__':
    policy = Policy(5, 5, 1)
    Q = QFunction(5, 5, 1)
