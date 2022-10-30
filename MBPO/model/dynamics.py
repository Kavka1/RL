import itertools
from typing import List, Tuple, Dict, Union
import numpy as np
import random
from pandas import period_range
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL.MBPO.buffer import Buffer
from RL.MBPO.model.base import Module, call_mlp, BatchedLinear, unbatch_forward, Normalizer


class BatchGaussianEnsemble(Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
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
        param_reg_weight: float,
        optimizer_factory,
        elite_size: int,
        device: str,
        use_action_normalization: bool
    ) -> None:
        super().__init__()

        self.s_dim, self.a_dim  =   s_dim, a_dim
        self.ensemble_size      =   ensemble_size
        self.elite_size         =   elite_size
        in_dim, out_dim         =   s_dim + a_dim, s_dim + 1
        self.device             =   torch.device(device)
        self.use_action_norm    =   use_action_normalization

        self.state_normalizer = Normalizer(s_dim)
        if self.use_action_norm:
            self.action_normalizer= Normalizer(a_dim)
        
        self.min_log_var = nn.Parameter(torch.full([out_dim], init_min_log_var, device=self.device).float(), requires_grad=True)
        self.max_log_var = nn.Parameter(torch.full([out_dim], init_max_log_var, device=self.device).float(), requires_grad=True)
        
        self.batch_size             =   batch_size
        self.learning_rate          =   learning_rate
        self.log_var_bound_weight   =   log_var_bound_weight

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
        
        self.elite_model_indexes = list(range(self.elite_size))

        self.optimizer = optimizer_factory([
            *self.trunk.parameters(),
            *self.mean_head.parameters(),
            *self.log_var_head.parameters(),
            self.min_log_var, self.max_log_var
        ], self.learning_rate, weight_decay = param_reg_weight)

    @property
    def total_batch_size(self,) -> float:
        return self.ensemble_size * self.batch_size

    def forward_single(self, state: torch.tensor, action: torch.tensor, model_index: int) -> Tuple:
        normalized_state    = self.state_normalizer(state)
        if self.use_action_norm:
            action   = self.action_normalizer(action)
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
        if self.use_action_norm:
            action   = self.action_normalizer(action)
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

    def compute_loss(self, state: torch.tensor, action: torch.tensor, target: torch.tensor, factored: bool) -> torch.tensor:
        inputs = [state, action, target]
        total_batch_size = len(target)
        remainder = total_batch_size % self.ensemble_size
        if remainder != 0:
            nearest = total_batch_size - remainder
            inputs  = [x[:nearest] for x in inputs]

        state, action, target   =   [self._rebatch(x) for x in inputs]
        means, log_vars         =   self.forward_all(state, action)
        inv_vars                =   torch.exp(-log_vars)
        squared_errors          =   torch.mean((target - means) ** 2 * inv_vars, dim=-1)     # [ensemble_size, batch_size]
        log_dets                =   torch.mean(log_vars, dim=-1)                             # [ensemble_size, batch_size]
        log_var_bound_loss      =   self.max_log_var.sum() - self.min_log_var.sum()          # [None]

        if factored:
            mle_loss                =   torch.mean(squared_errors + log_dets, dim= -1)      # [ensemble_size]
        else:
            mle_loss                =   torch.mean(squared_errors + log_dets)               # [None]

        return mle_loss + self.log_var_bound_weight * log_var_bound_loss

    def train(self, buffer: Buffer, max_train_step_since_update: int, hold_ratio: float) -> List[float]:
        self._max_train_step_since_update   = max_train_step_since_update
        self._steps_since_update            = 0
        self._stats                         = {}
        self._snapshots                     = {i: (None, 1e10) for i in range(self.ensemble_size)}
        # obtain all samples and shuffle
        n = len(buffer)
        state, action, reward, done, next_state = buffer.sample_all()
        permutation = np.random.permutation(n)
        state       = state[permutation, :]
        action      = action[permutation, :]
        reward      = reward[permutation]
        next_state  = next_state[permutation, :]
        # transfer to tensor
        state       = torch.tensor(state, device=self.device).float()
        action      = torch.tensor(action, device=self.device).float()
        reward      = torch.tensor(reward, device=self.device).unsqueeze(-1).float()
        next_state  = torch.tensor(next_state, device=self.device).float()
        # targets: [next_state, reward]
        target      = torch.cat([next_state, reward], -1)
        # fit the normalizer
        self.state_normalizer.fit(state)
        if self.use_action_norm:
            self.action_normalizer.fit(action)
        # split the samples
        num_holdout = int(hold_ratio * n)
        train_state, train_action       =   state[num_holdout:, :], action[num_holdout:, :]
        train_target                    =   target[num_holdout:, :]

        holdout_state, holdout_action   =   state[:num_holdout, :], action[:num_holdout, :]
        holdout_target                  =   target[:num_holdout, :]

        for step in itertools.count():
            indices = torch.randperm(train_state.shape[0], device=self.device)
            for start_pos in range(0, train_state.shape[0], self.total_batch_size):
                if start_pos + self.total_batch_size >= train_state.shape[0]:
                    idxs    = indices[start_pos:]
                else:
                    idxs    = indices[start_pos:start_pos+self.total_batch_size]
                s, a    = train_state[idxs, :], train_action[idxs, :]
                label   = train_target[idxs, :]
                loss    = self.compute_loss(s, a, label, factored=False)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # check if model improves
            with torch.no_grad():
                holdout_loss    =   self.compute_loss(holdout_state, holdout_action, holdout_target, factored=True)
                sorted_loss_idx =   torch.argsort(holdout_loss)
                self.elite_model_indexes    =   sorted_loss_idx[:self.elite_size].tolist()
                break_train     =   self._save_best(step, holdout_loss)
                if break_train:
                    break

        return holdout_loss.cpu().numpy()

    def _save_best(self, train_step: int, holdout_losses: torch.tensor) -> bool:
        holdout_losses = holdout_losses.cpu().numpy()
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (train_step, current)
                updated = True

        if updated:
            self._steps_since_update = 0
        else:
            self._steps_since_update += 1

        if self._steps_since_update > self._max_train_step_since_update:
            return True
        else:
            return False

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