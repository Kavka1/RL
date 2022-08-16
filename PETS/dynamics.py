from typing import List, Tuple, Dict, Union
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from RL.PETS.common import init_weight, Module, Normalizer, call_mlp
from RL.PETS.buffer import Buffer


def unbatch_forward(batch_sequential: nn.ModuleList, input: torch.tensor, index: int) -> torch.tensor:
    for layer in batch_sequential:
        if isinstance(layer, BatchedLinear):
            input = F.linear(input, layer.weight[index], layer.bias[index])
        else:
            input = layer(input)
    return input


class BatchedLinear(nn.Module):
    def __init__(self, ensemble_size: int, in_dim: int, out_dim: int, initializer: str) -> None:
        super().__init__()
        self.ensemble_size      =       ensemble_size
        self.in_dim             =       in_dim
        self.out_dim            =       out_dim
        self.weight             =       nn.Parameter(torch.empty(ensemble_size, in_dim, out_dim))
        self.bias               =       nn.Parameter(torch.empty(ensemble_size, out_dim))
        # weight initialization
        init_weight(self.weight, initializer)
        init_weight(self.bias, initializer)

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert len(x.shape) == 3
        assert x.shape[0] == self.ensemble_size
        return torch.bmm(x, self.weight) + self.bias.unsqueeze(1)


class BatchGaussianEnsemble(Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        device: str,
        ensemble_size: int,
        trunk_hiddens: List[int],
        head_hiddens: List[int],
        inner_nonlinear: str,
        initializer: str,
        init_min_log_var: float,
        init_max_log_var: float,
        log_var_bound_weight: float,
        batch_size: int,
        learning_rate: float,
        optimizer_factory
    ) -> None:
        super().__init__()

        self.s_dim, self.a_dim  =   s_dim, a_dim
        self.ensemble_size      =   ensemble_size
        in_dim, out_dim         =   s_dim + a_dim, s_dim + 1
        self.device             =   torch.device(device)

        self.min_log_var = nn.Parameter(torch.full([out_dim], init_min_log_var, device=self.device).float())
        self.max_log_var = nn.Parameter(torch.full([out_dim], init_max_log_var, device=self.device).float())
        self.state_normalizer = Normalizer(s_dim)

        self.log_var_bound_weight   =   log_var_bound_weight
        self.batch_size             =   batch_size
        self.learning_rate          =   learning_rate

        layer_factory = lambda n_in, n_out: BatchedLinear(self.ensemble_size, n_in, n_out, initializer)

        self.trunk = call_mlp(
            in_dim, 
            trunk_hiddens[-1],
            trunk_hiddens[:-1],
            inner_nonlinear,
            inner_nonlinear,
            initializer,
            layer_factory
        )
        self.mean_head = call_mlp(
            trunk_hiddens[-1],
            out_dim,
            head_hiddens,
            inner_nonlinear,
            'Identity',
            initializer,
            layer_factory
        )
        self.log_var_head = call_mlp(
            trunk_hiddens[-1],
            out_dim,
            head_hiddens,
            inner_nonlinear,
            'Identity',
            initializer,
            layer_factory
        )

        self.to(device)
        
        self.optimizer = optimizer_factory([
            *self.trunk.parameters(),
            *self.mean_head.parameters(),
            *self.log_var_head.parameters(),
            self.min_log_var, self.max_log_var
        ], self.learning_rate)

    @property
    def total_batch_size(self,) -> float:
        return self.ensemble_size * self.batch_size

    def forward_single(self, state: torch.tensor, action: torch.tensor, model_index: int) -> Tuple:
        normalized_state = self.state_normalizer(state)
        inputs = torch.cat([normalized_state, action], dim=-1)
        batch_size = inputs.shape[0]
        # forward
        embed_feature   =   unbatch_forward(self.trunk, inputs, model_index)
        means           =   unbatch_forward(self.mean_head, embed_feature, model_index)
        log_vars        =   unbatch_forward(self.log_var_head, embed_feature, model_index)
        # pre next states = states + mean_diff[:-1], pred rewards = mean_diff[-1]
        means           =   means + torch.cat([state, torch.zeros([batch_size, 1], device=self.device)], dim=-1)
        log_vars        =   self.max_log_var - F.softplus(self.max_log_var - log_vars)
        log_vars        =   self.min_log_var + F.softplus(log_vars - self.min_log_var)
        return means, log_vars

    def forward_all(self, state: torch.tensor, action: torch.tensor) -> Tuple:
        normalized_state    =   self.state_normalizer(state)
        inputs              =   torch.cat([normalized_state, action], -1)
        batch_size          =   inputs.shape[1]
        # forward
        embed_feature       =   self.trunk(inputs)
        means               =   self.mean_head(embed_feature)
        log_vars            =   self.log_var_head(embed_feature)
        # pre next states = states + mean_diff[:-1], pred rewards = mean_diff[-1]
        means               =   means + torch.cat([state, torch.zeros([self.ensemble_size, batch_size, 1], device=self.device)], -1)
        log_vars            =   self.max_log_var - F.softplus(self.max_log_var - log_vars)
        log_vars            =   self.min_log_var + F.softplus(log_vars - self.min_log_var)
        return means, log_vars

    def _rebatch(self, x):
        total_batch_size = len(x)
        assert total_batch_size % self.ensemble_size == 0, f'{total_batch_size} not divisible by {self.ensemble_size}'
        batch_size = total_batch_size // self.ensemble_size
        remaining_dims = tuple(x.shape[1:])
        return x.reshape(self.ensemble_size, batch_size, *remaining_dims)

    def compute_loss(self, state: torch.tensor, action: torch.tensor, target: torch.tensor) -> torch.tensor:
        inputs = [state, action, target]
        total_batch_size = len(target)
        remainder = total_batch_size % self.ensemble_size
        if remainder != 0:
            nearest = total_batch_size - remainder
            inputs  = [x[:nearest] for x in inputs]

        state, action, target   =   [self._rebatch(x) for x in inputs]
        means, log_vars         =   self.forward_all(state, action)
        inv_vars                =   torch.exp(-log_vars)
        squared_errors          =   torch.sum((target - means) ** 2 * inv_vars, dim=-1)
        log_dets                =   torch.sum(log_vars, dim=-1)
        mle_loss                =   torch.mean(squared_errors + log_dets)
        log_var_bound_loss      =   self.max_log_var.sum() - self.min_log_var.sum()
        return mle_loss + self.log_var_bound_weight * log_var_bound_loss

    def train(self, buffer: Buffer, steps: int) -> List[float]:
        n = len(buffer)
        state, action, reward, done, delta_state = buffer.sample_all()
        # transfer to tensor
        state       = torch.tensor(state, device=self.device).float()
        action      = torch.tensor(action, device=self.device).float()
        reward      = torch.tensor(reward, device=self.device).unsqueeze(-1).float()
        delta_state = torch.tensor(delta_state, device=self.device).float()
        # fit the normalizer
        self.state_normalizer.fit(state)
        # targets: [next_state, reward]
        target      = torch.cat([delta_state, reward], -1)
        all_losses  = []
        for _ in range(steps):
            indices = torch.randint(n, [self.total_batch_size], device=self.device)
            loss    = self.compute_loss(state[indices], action[indices], target[indices])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_losses.append(loss.item())
        return all_losses

    def sample(self, state: torch.torch.tensor, action: torch.torch.tensor) -> Tuple:
        index = random.randrange(self.ensemble_size)
        means, log_vars = self.forward_single(state, action, index)
        stds = torch.exp(log_vars).sqrt()
        samples = means + stds * torch.randn_like(means)
        return samples[:,:-1], samples[:,-1]

    def means(self, state: torch.tensor, action: torch.tensor) -> Tuple:
        state = state.repeat(self.ensemble_size, 1, 1)
        action= action.repeat(self.ensemble_size, 1, 1)
        means, log_vars = self.forward_all(state, action)
        return means[:, :, :-1], means[:, :, -1]