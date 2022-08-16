from typing import Dict, List, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from RL.PETS.dynamics import BatchGaussianEnsemble
from RL.PETS.common import Module
from RL.PETS.buffer import Buffer
from RL.PETS.cem import CEMOptimizer



class PETS(Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.s_dim = config['model_config']['s_dim']
        self.a_dim = config['model_config']['a_dim']
        self.a_bound = config['model_config']['a_bound']
        self.a_lb  = np.ones((self.a_dim)) * self.a_bound
        self.a_ub  = - np.ones((self.a_dim)) * self.a_bound

        self.horizon    = config['horizon']
        self.n_particel = config['n_particel']
        self.device     = config['device']

        self.current_s = None
        self.action_buf = np.array([]).reshape(0, self.a_dim)
        self.prev_sol = np.tile((self.a_ub + self.a_lb) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.a_ub - self.a_lb) / 16, [self.horizon])

        self.model = BatchGaussianEnsemble(
            self.s_dim,
            self.a_dim,
            self.device,
            config['model_config']['ensemble_size'],
            config['model_config']['model_trunk_hiddens'],
            config['model_config']['model_head_hiddens'],
            config['model_config']['model_inner_nonlinear'],
            config['model_config']['model_initializer'],
            config['model_config']['model_min_log_var'],
            config['model_config']['model_max_log_var'],
            config['log_var_bound_weight'],
            config['batch_size'],
            config['learning_rate'],
            optim.Adam
        )

        self.cem = CEMOptimizer(
            self.horizon * self.a_dim,
            config['cem_max_iter'],
            config['cem_pop_size'],
            config['cem_num_elites'],
            self._eval_action_seq,
            self.a_ub,
            self.a_lb,
        )

        self.to(self.device)

    def reset(self) -> None:
        self.prev_sol = np.tile((self.a_lb + self.a_ub) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.a_ub + self.a_lb) / 16, [self.horizon])

    def get_action(self, states: np.array) -> np.array:
        if self.action_buf.shape[0] > 0:
            action, self.action_buf = self.action_buf[0], self.action_buf[1:]
            action = torch.from_numpy(action).unsqueeze(0)
            return action

        self.cur_states = torch.from_numpy(states).to(self.device).float()
        solution = self.cem.obtain_solution(self.prev_sol, self.init_var)       # [n_horizon * a_dim]
        self.prev_sol = np.concatenate([np.copy(solution)[self.a_dim:], np.zeros(self.a_dim, dtype=np.float32)])
        self.action_buf = solution[: self.a_dim].reshape(-1, self.a_dim)
        return self.get_action(states)

    def update(self, buffer, num_step: int) -> List[float]:
        return self.model.train(buffer, num_step)

    @torch.no_grad()
    def _eval_action_seq(self, a_seq: np.array) -> np.array:
        n_opt = a_seq.shape[0]
        actions_sequences   = torch.from_numpy(a_seq).float().to(self.device)               # [pop_size, horizon, a_dim]
        actions_sequences   = actions_sequences.view(-1, self.horizon, self.a_dim)
        transposed          = actions_sequences.transpose(0, 1)
        expanded            = transposed[:, :, None]

        tiled               = expanded.expand(-1, -1, self.n_particel, -1)                  # [horizon, pop_size, n_particel, a_dim]  
        actions_sequences   = tiled.contiguous().view(self.horizon, -1, self.a_dim)         # [horizon, pop * part, a_dim]
        
        cur_states          = self.cur_states.expand(n_opt * self.n_particel, -1).to(self.device)    # [pop * part, s_dim]

        costs = torch.zeros(n_opt, self.n_particel, device=self.device)
        for t in range(self.horizon):
            cur_acs                 =   actions_sequences[t]
            next_states, rewards    =   self.model.sample(cur_states, cur_acs)              # [pop * part, s_dim], [pop * part, 1]
            step_rewards            =   rewards.reshape([-1, self.n_part])                  # [pop, part]

            costs += step_rewards
            cur_states = next_states

        return costs.mean(dim=1).detach().cpu().numpy()