import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class DeterministicPolicy(nn.Module):
    def __init__(self, o_dim, a_dim):
        super(DeterministicPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(o_dim),
            nn.Linear(o_dim, 128),
            nn.Tanh(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.LayerNorm(128),
            nn.Linear(128, a_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        action = self.model(obs)
        return action


class QFunction(nn.Module):
    def __init__(self, o_dim, a_dim):
        super(QFunction, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(o_dim + a_dim),
            nn.Linear(o_dim + a_dim, 400),
            nn.ELU(),
            nn.LayerNorm(400),
            nn.Linear(400, 300),
            nn.ELU(),
            nn.LayerNorm(300),
            nn.Linear(300, 1),
        )

    def forward(self, obs, action):
        input = torch.cat([obs, action], dim=-1)
        Q_value = self.model(input)
        return Q_value

class GaussianPolicy(nn.Module):
    def __init__(self, o_dim, a_dim, log_std_max=20, log_std_min=-2):
        super(GaussianPolicy, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim

        self.encoder = nn.Sequential(
            nn.Linear(o_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(256, a_dim),
            nn.Tanh()
        )
        self.log_std_layer = nn.Linear(256, a_dim)

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def forward(self, obs):
        X = self.encoder(obs)
        mean = self.mean_layer(X)
        log_std = self.log_std_layer(X)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        action = dist.rsample() #reparametrization trick
        log_prob = dist.log_prob(action)
        
        action = torch.tanh(action)
        log_prob -= torch.log(1- action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean

    
class TwinQFunction(nn.Module):
    def __init__(self, o_dim, a_dim):
        super(TwinQFunction, self).__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(o_dim + a_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(o_dim + a_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, obs, action):
        input = torch.cat([obs, action], dim=-1)
        
        Q1_value = self.Q1(input)
        Q2_value = self.Q2(input)
        
        return Q1_value, Q2_value