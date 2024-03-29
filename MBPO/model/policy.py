from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from RL.MBPO.model.base import call_mlp, Module


class SquashedGaussianPolicy(Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        hidden_layers: List[int],
        inner_nonlinear: str,
        log_std_min: float,
        log_std_max: float,
        initializer: str
    ) -> None:
        super().__init__()
        self.s_dim, self.a_dim = s_dim, a_dim
        self._model = call_mlp(
            s_dim,
            a_dim * 2,
            hidden_layers,
            inner_nonlinear,
            'Identity',
            initializer
        )
        self.log_std_min = nn.Parameter(torch.ones([a_dim]) * log_std_min, requires_grad=False)
        self.log_std_max = nn.Parameter(torch.ones([a_dim]) * log_std_max, requires_grad=False)

    def sample_action(self, state: torch.tensor, with_noise: bool) -> torch.tensor:
        with torch.no_grad():
            mix = self._model(state)
            mean, log_std = torch.chunk(mix, 2, dim=-1)
        if with_noise:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action = dist.sample()
        else:
            action = mean
        return torch.tanh(action)

    def forward(self, state: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.distributions.Distribution]:
        mix             =   self._model(state)
        mean, log_std   =   torch.chunk(mix, 2, dim=-1)
        log_std         =   torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std             =   torch.exp(log_std)

        dist                =   Normal(mean, std)
        arctanh_actions     =   dist.rsample()
        log_prob            =   dist.log_prob(arctanh_actions).sum(-1, keepdim=True)

        action              =   torch.tanh(arctanh_actions)
        squashed_correction =   torch.log(1 - action**2 + 1e-6).sum(-1, keepdim=True)
        log_prob            =   log_prob - squashed_correction

        return action, log_prob, dist