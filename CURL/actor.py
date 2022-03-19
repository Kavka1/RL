from typing import List, Tuple, Dict
import torch
import torch.nn as nn

from RL.CURL.encoder import PixelEncoder
from RL.CURL.utils import weight_init, reparameterize, compute_gaussian_logprob, squash


class Actor(nn.Module):
    def __init__(
        self,
        encoder: PixelEncoder,
        a_dim: int,
        hidden_dim: int,
        logstd_max: float,
        logstd_min: float,
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.logstd_max = logstd_max
        self.logstd_min = logstd_min

        self.trunk = nn.Sequential(
            nn.Linear(encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, a_dim * 2)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def __call__(self, obs: torch.tensor, detach_encoder: bool, compute_pi: bool , compute_log_prob: bool) -> Tuple:
        x = self.encoder(obs, detach_encoder)

        mu, arctanh_log_std = self.trunk(x).chunk(2, dim=-1)

        log_std = torch.tanh(arctanh_log_std)
        log_std = self.logstd_min + 0.5 * (self.logstd_max - self.logstd_min) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            pi, noise = reparameterize(mu, log_std.exp(), True)
        else:
            pi, noise = None, None

        if compute_log_prob:
            log_prob = compute_gaussian_logprob(noise, log_std)
        else:
            log_prob = None

        mu, pi, log_prob = squash(mu, pi, log_prob)

        return mu, pi, log_prob, log_std