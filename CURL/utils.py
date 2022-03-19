from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def tie_weights(trg: nn.Module, src: nn.Module) -> None:
    assert type(trg) == type(src)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def reparameterize(mu: torch.tensor, std: torch.tensor, return_noise: bool = True) -> Tuple[torch.tensor, torch.tensor]:
    noise = torch.rand_like(std)
    output = mu + std * noise
    if return_noise:
        return output, noise
    else:
        return output


def compute_gaussian_logprob(noise: torch.tensor, log_std: torch.tensor) -> torch.tensor:
    # noise denotes to the x = mu + "noise" * std
    return (- 0.5 * noise ** 2 - log_std).sum(-1, keepdim = True) - 0.5 * torch.log(2 * torch.pi) * noise.size(-1)


def squash(mu: torch.tensor, pi: torch.tensor, log_prob: torch.tensor) -> Tuple:
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_prob is not None:
        log_prob -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_prob