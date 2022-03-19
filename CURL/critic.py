from typing import Dict, List, Tuple
import torch
import torch.nn as nn
 
from RL.CURL.encoder import PixelEncoder
from RL.CURL.utils import weight_init


class QFunction(nn.Module):
    def __init__(
        self, 
        o_dim: int, 
        a_dim: int, 
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.trunk = nn.ModuleList(
            nn.Linear(o_dim + a_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def __forward__(self, obs: torch.tensor, a: torch.tensor) -> torch.tensor:
        return self.trunk(torch.cat([obs, a], dim=-1))


class Critic(nn.Module):
    def __init__(
        self,
        encoder: PixelEncoder,
        a_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.Q1 = QFunction(encoder.feature_dim, a_dim, hidden_size)
        self.Q2 = QFunction(encoder.feature_dim, a_dim, hidden_size)
        
        self.outputs = dict()
        self.apply(weight_init)

    def __forward__(self, obs: torch.tensor, a: torch.tensor, detach_encoder: bool = False) -> Tuple[torch.tensor, torch.tensor]:
        x = self.encoder(obs, detach_encoder)
        q1 = self.Q1(x, a)
        q2 = self.Q2(x, a)
        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def call_q1(self, obs: torch.tensor, a: torch.tensor, detach_encoder: bool = False) -> torch.tensor:
        x = self.encoder(obs, detach_encoder)
        q1 = self.Q1(x, a)
        self.outputs['q1'] = q1
        return q1

    def call_q2(self, obs: torch.tensor, a: torch.tensor, detach_encoder: bool = False) -> torch.tensor:
        x = self.encoder(obs, detach_encoder)
        q2 = self.Q2(x, a)
        self.outputs['q2'] = q2
        return q2