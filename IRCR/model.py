from math import log
from typing import List, Tuple, Dict, Type
import numpy as np
import torch
from torch._C import DictType
import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(self, model_config: Dict, device: torch.device) -> None:
        super(Policy, self).__init__()
        self.hidden_layers = model_config['policy_hidden_layers']
        self.o_dim = model_config['o_dim']
        self.a_dim = model_config['a_dim']
        self.logstd_min = model_config['logstd_min']
        self.logstd_max = model_config['logstd_max']
        self.action_min = model_config['action_low']
        self.action_max = model_config['action_high']
        self.device = device

        model = []
        hidden_layers = [self.o_dim] + self.hidden_layers
        for i in range(len(hidden_layers)-1):
            model.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            model.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*model)
        self.mean_head = nn.Sequential(nn.Linear(hidden_layers[-1], self.a_dim))
        self.logstd_head = nn.Sequential(nn.Linear(hidden_layers[-1], self.a_dim))

    def __call__(self, obs: np.array, evaluation: bool = False) -> np.array:
        obs = torch.from_numpy(obs).float().to(self.device)
        
        if evaluation:
            with torch.no_grad():
                preprocess = self.hidden_layers(obs)
                mean = self.mean_head(preprocess)
            action = torch.tanh(mean)
        else:
            with torch.no_grad():
                preprocess = self.hidden_layers(obs)
                mean = self.mean_head(preprocess)
                logstd = self.logstd_head(preprocess)
            logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
            std = torch.exp(logstd)
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.tanh(action)
        
        action = np.clip(action.cpu().numpy(), self.action_min, self.action_max)
        return action

    def forward_batch(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        preprocess = self.hidden_layers(obs)
        mean = self.mean_head(preprocess)
        logstd = self.logstd_head(preprocess)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        std = torch.exp(logstd)

        dist = Normal(mean, std)
        arctanh_action = dist.rsample()
        action = torch.tanh(arctanh_action)

        log_prob = dist.log_prob(arctanh_action) - torch.log(1- torch.tanh(arctanh_action).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        print(f"--------- Loaded model from {path} -----------")


class TwinQFuncion(nn.Module):
    def __init__(self, model_config: Dict, device: torch.device) -> None:
        super(TwinQFuncion, self).__init__()
        self.o_dim = model_config['o_dim']
        self.a_dim = model_config['a_dim']
        self.hidden_layers = model_config['value_hidden_layers']
        self.device = device

        model = []
        hidden_layers = [self.o_dim + self.a_dim] + self.hidden_layers
        for i in range(len(hidden_layers) - 1):
            model.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            model.append(nn.ReLU())
        model.append(nn.Linear(hidden_layers[-1], 1))

        self.Q1 = nn.Sequential(*model)
        self.Q2 = nn.Sequential(*model)
    
    def __call__(self, obs: torch.tensor, action: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        x = torch.cat([obs, action], dim=-1)
        q1_value = self.Q1(x)
        q2_value = self.Q2(x)
        return q1_value, q2_value

    def get_q1(self, obs: torch.tensor, action: torch.tensor) -> torch.tensor:
        x = torch.cat([obs, action], dim=-1)
        q1_value = self.Q1(x)
        return q1_value

    def get_q2(self, obs: torch.tensor, action: torch.tensor) -> torch.tensor:
        x = torch.cat([obs, action], dim=-1)
        q2_value = self.Q2(x)
        return q2_value