from typing import Tuple, Dict, List, Type, Union
import numpy as np
import torch
from torch import random
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self,config: Dict) -> None:
        super(Policy, self).__init__()
        
        o_dim, a_dim = config['o_dim'], config['a_dim']
        hidden_sizes = config['hidden_sizes']
        module_seq = []
        last_dim = o_dim
        for i in range(len(hidden_sizes)):
            module_seq += [nn.Linear(last_dim, hidden_sizes[i]), nn.ReLU()]
            last_dim = hidden_sizes[i]
        module_seq += [nn.Linear(last_dim, a_dim), nn.Tanh()]
        self.model = nn.Sequential(*module_seq)

    def forward(self, observation: torch.tensor):
        return self.model(observation)

class Q_Function(nn.Module):
    def __init__(self, config: Dict) -> None:
        super(Q_Function, self).__init__()

        o_dim, a_dim = config['o_dim'], config['a_dim']
        hidden_sizes = config['hidden_sizes']
        module_seq = []
        last_dim = o_dim + a_dim
        for i in range(len(hidden_sizes)):
            module_seq += [nn.Linear(last_dim, hidden_sizes[i]), nn.ReLU()]
            last_dim = hidden_sizes[i]
        module_seq += [nn.Linear(last_dim, 1)]
        self.model = nn.Sequential(*module_seq)
    
    def forward(self, obs: torch.tensor, action: torch.tensor) -> torch.tensor:
        value = self.model(torch.cat([obs, action], dim=-1))
        return value

class Twin_Q(nn.Module):
    def __init__(self,config: Dict) -> None:
        super(Twin_Q, self).__init__()
        self.Q_1 = Q_Function(config)
        self.Q_2 = Q_Function(config)
    
    def Q1_value(self, obs: torch.tensor, action: torch.tensor) -> torch.tensor:
        return self.Q_1(obs, action)
    
    def Q2_value(self, obs: torch.tensor, action: torch.tensor) -> torch.tensor:
        return self.Q_2(obs, action)

    def forward(self, obs: torch.tensor, action: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        return self.Q_1(obs, action), self.Q_2(obs, action)


if __name__ == "__main__":
    config = {'hidden_sizes': [256, 256, 256], 'o_dim': 10, 'a_dim': 2}
    actor = Policy(config=config)