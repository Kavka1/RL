from typing import Tuple, Dict, List, Type, Union
import numpy as np
import torch
from torch import random
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, o_dim: Union[int, np.int32], a_dim: Union[int, np.int32], config: Dict):
        super(Policy, self).__init__()
        
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

class Q_Function_Disc(nn.Module):
    def __init__(self, o_dim: Union[int, np.int32], a_dim: Union[int, np.int32], config: Dict):
        super(Q_Function_Disc, self).__init__()
        hidden_sizes = config['hidden_sizes']
        module_seq = []
        last_dim = o_dim
        for i in range(len(hidden_sizes)):
            module_seq += [nn.Linear(last_dim, hidden_sizes[i]), nn.ReLU()]
            last_dim = hidden_sizes[i]
        module_seq += [nn.Linear(last_dim, a_dim)]
        self.model = nn.Sequential(*module_seq)
    
    def forward(self, obs: torch.tensor, action: torch.tensor):
        # Todo: need fix
        values = self.model(obs)
        return values[action]

class Q_Function_Cont(nn.Module):
    def __init__(self, o_dim: Union[int, np.int32], a_dim: Union[int, np.int32], config: Dict):
        super(Q_Function_Cont, self).__init__()
        hidden_sizes = config['hidden_sizes']
        module_seq = []
        last_dim = o_dim + a_dim
        for i in range(len(hidden_sizes)):
            module_seq += [nn.Linear(last_dim, hidden_sizes[i]), nn.ReLU()]
            last_dim = hidden_sizes[i]
        module_seq += [nn.Linear(last_dim, 1)]
        self.model = nn.Sequential(*module_seq)
    
    def forward(self, obs: torch.tensor, action: torch.tensor):
        values = self.model(torch.cat([obs, action], dim=-1))
        return values

if __name__ == "__main__":
    config = {'hidden_sizes': [256, 256, 256]}
    actor = Policy(o_dim=10, a_dim=2, config=config)
    Q_d = Q_Function_Disc(o_dim=10, a_dim=2, config=config)
    Q_c = Q_Function_Cont(o_dim=10, a_dim=2, config=config)

    obs = torch.rand(size=(1, 10))
    print(actor(obs), Q_d(obs, a_d), Q_c(obs, a_c))