from typing import Dict, Tuple, List, Union
import gym
import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from RL.MBPO.buffer import Buffer
from RL.MBPO.utils import soft_update, confirm_path_exist
from RL.MBPO.model.policy import SquashedGaussianPolicy
from RL.MBPO.model.value import QEnsemble
from RL.MBPO.model.dynamics import BatchGaussianEnsemble


class MBPO_Agent:
    def __init__(self, config: Dict) -> None:
        self.config             =       config
        self.model_config       =       config['model_config']
        self.s_dim              =       self.model_config['s_dim']
        self.a_dim              =       self.model_config['a_dim']
        self.device             =       config['device']
        self.exp_path           =       config['exp_path']

        self.lr                 =       config['lr']
        self.gamma              =       config['gamma']
        self.tau                =       config['tau']
        self.alpha              =       config['alpha']
        self.training_delay     =       config['training_delay']
        self.env_batch_ratio    =       config['env_batch_ratio']

        self.batch_size         =       config['batch_size']
        self.rollout_batch_size =       config['rollout_batch_size']

        self.ac_train_repeat    =       config['ac_train_repeat']
        self.model_train_freq   =       config['model_train_freq']

        self.roll_model_length  =       config['min_roll_length']

        self.min_roll_length    =       config['min_roll_length']
        self.max_roll_length    =       config['max_roll_length']
        self.min_step_for_length=       config['min_step_for_length']
        self.max_step_for_length=       config['max_step_for_length']

        self.dynamics_max_train_step_since_update   =   config['dynamics_max_train_step_since_update']
        self.dynamics_train_holdout_ratio           =   config['dynamics_train_holdout_ratio']

        self.training_count =       0
        self.loss_log       =       {}

        # adaptive alpha
        if self.alpha is None:
            self.train_alpha    =   True
            self.log_alpha      =   torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha          =   torch.exp(self.log_alpha)
            self.target_entropy =   - torch.tensor(config['model_config']['a_dim'], dtype=torch.float64)
            self.optimizer_alpha=   optim.Adam([self.log_alpha], lr= self.lr)
        else:
            self.train_alpha    =   False

        # policy
        self.policy         =       SquashedGaussianPolicy(
            s_dim           =   self.s_dim,
            a_dim           =   self.a_dim,
            hidden_layers   =   self.model_config['policy_hiddens'],
            inner_nonlinear =   self.model_config['policy_nonlinear'],
            log_std_min     =   self.model_config['policy_log_std_min'],
            log_std_max     =   self.model_config['policy_log_std_max'],
            initializer     =   self.model_config['policy_initializer']
        ).to(self.device)
        self.optimizer_policy   =   optim.Adam(self.policy.parameters(), self.lr)
        
        # value functions
        self.QFunction      =       QEnsemble(
            ensemble_size   =   2,
            s_dim           =   self.s_dim,
            a_dim           =   self.a_dim,
            hiddens         =   self.model_config['value_hiddens'],
            inner_nonlinear =   self.model_config['value_nonlinear'],
            initializer     =   self.model_config['value_initializer']
        ).to(self.device)
        self.QFunction_tar  =      QEnsemble(
            ensemble_size   =   2,
            s_dim           =   self.s_dim,
            a_dim           =   self.a_dim,
            hiddens         =   self.model_config['value_hiddens'],
            inner_nonlinear =   self.model_config['value_nonlinear'],
            initializer     =   self.model_config['value_initializer']
        ).to(self.device)
        self.QFunction_tar.load_state_dict(self.QFunction.state_dict())
        self.optimizer_value    =   optim.Adam(self.QFunction.parameters(), self.lr)

        # batch dynamics model
        self.dynamics   =   BatchGaussianEnsemble(
            self.s_dim,
            self.a_dim,
            self.model_config['dynamics_ensemble_size'],
            self.model_config['dynamics_trunk_hiddens'],
            self.model_config['dynamics_head_hiddens'],
            self.model_config['dynamics_inner_nonlinear'],
            self.model_config['dynamics_initializer'],
            self.model_config['dynamics_init_min_log_var'],
            self.model_config['dynamics_init_max_log_var'],
            self.model_config['dynamics_log_var_bound_weight'],
            self.batch_size,
            self.lr,
            self.model_config['dynamics_weight_decay_coeff'],
            optim.Adam,
            self.model_config['dynamics_elite_size'],
            self.device
        ).to(self.device)

        self.env_buffer         =   Buffer(config['env_buffer_size'])
        self.model_buffer       =   Buffer(int(self.roll_model_length * self.rollout_batch_size * (1000 / self.model_train_freq)))

    def sample_action(self, s: np.array, with_noise: bool) -> np.array:
        with torch.no_grad():
            s               =   torch.from_numpy(s).float().to(self.device)
            action          =   self.policy.sample_action(s, with_noise)
        return action.detach().cpu().numpy()

    def rollout_model(self, deterministic: bool, terminal_func: callable) -> None:
        state, action, reward, done, next_state = self.env_buffer.sample_duplicated_batch(self.rollout_batch_size)
        state   =   np.repeat(state[np.newaxis, :, :], self.dynamics.ensemble_size, 0)
        for i in range(self.roll_model_length):
            # arr2tensor
            state   = torch.from_numpy(state).float().to(self.device)
            # forward
            action  = self.policy.sample_action(state, with_noise=True)
            means, log_vars = self.dynamics.forward_all(state, action)
            stds            = torch.exp(log_vars).sqrt()
            # tensor2arr
            state   =   state.detach().cpu().numpy()
            action  =   action.detach().cpu().numpy()
            means   =   means.detach().cpu().numpy()
            stds    =   stds.detach().cpu().numpy()
            if deterministic:
                samples =   means
            else:
                samples =   means + np.random.randn(*stds.shape) * stds
            pred_state, reward      =   samples[:, :, :-1], samples[:, :, -1]
            # select the prediction of one elite model
            current_batch_size      =   state.shape[1]
            elite_idxs                      =   np.random.choice(self.dynamics.elite_model_indexes, size=current_batch_size)
            batch_idxs                      =   np.arange(0, current_batch_size)

            elite_state                     =   state[elite_idxs, batch_idxs]
            elite_action                    =   action[elite_idxs, batch_idxs]
            elite_pred_state                =   pred_state[elite_idxs, batch_idxs]
            elite_reward                    =   reward[elite_idxs, batch_idxs]
            elite_terminal                  =   terminal_func(elite_state, elite_action, elite_pred_state)
            # store the samples
            for s, a, r, d, s_ in zip(elite_state,elite_action,elite_reward,elite_terminal,elite_pred_state):
                self.model_buffer.store((s,a,r,d,s_))
            # check if all done
            not_terminal    =   ~elite_terminal
            if not_terminal.sum() == 0:
                break
            # change the state
            state   =   elite_pred_state[not_terminal, :]
            state   =   np.repeat(state[np.newaxis, :, :], self.dynamics.ensemble_size, 0)

    def reset_rollout_length_and_reallocate_model_buf(self, current_step: int) -> None:
        # reset the rollout length
        new_rollout_length  =   min(
            self.max_roll_length,
            max(
                self.min_roll_length,
                self.min_roll_length + (current_step - self.min_step_for_length) / (self.max_step_for_length - self.min_step_for_length) * (self.max_roll_length - self.min_roll_length)
            )
        )
        new_rollout_length = int(new_rollout_length)
        if new_rollout_length != self.roll_model_length:
            self.roll_model_length = new_rollout_length
            # reallocate the model buffer
            new_model_buffer_size  = int(self.roll_model_length * self.rollout_batch_size * (1000 / self.model_train_freq))
            new_model_buffer       = Buffer(new_model_buffer_size)
            old_trans              = self.model_buffer.data.copy()
            for trans in old_trans:
                new_model_buffer.store(trans)
            self.model_buffer = new_model_buffer
        # log
        self.loss_log['rollout_length'] = self.roll_model_length

    def _pack_batch(
        self, 
        s_batch: np.array, 
        a_batch: np.array, 
        r_batch: np.array, 
        done_batch: np.array, 
        next_s_batch: np.array,
    ) -> Tuple[torch.tensor]:
        s_batch         =       torch.from_numpy(s_batch).float().to(self.device)
        a_batch         =       torch.from_numpy(a_batch).float().to(self.device)
        r_batch         =       torch.from_numpy(r_batch).float().to(self.device).unsqueeze(-1)
        done_batch      =       torch.from_numpy(done_batch).float().to(self.device).unsqueeze(-1)
        next_s_batch    =       torch.from_numpy(next_s_batch).float().to(self.device)
        return s_batch, a_batch, r_batch, done_batch, next_s_batch

    def train_model(self,) -> None:
        holdout_loss = self.dynamics.train(
            self.env_buffer, 
            self.dynamics_max_train_step_since_update,
            self.dynamics_train_holdout_ratio
        )
        self.loss_log['loss_dynamics'] = np.mean(holdout_loss)

    def train_ac(self) -> None:
        for train_step in range(self.ac_train_repeat):
            env_batch_size      =       int(self.batch_size * self.env_batch_ratio)
            img_batch_size      =       self.batch_size - env_batch_size

            env_s, env_a, env_r, env_done, env_next_s       =   self.env_buffer.sample(env_batch_size)
            img_s, img_a, img_r, img_done, img_next_s       =   self.model_buffer.sample(img_batch_size)

            s_batch         =       np.concatenate([env_s, img_s], 0)
            a_batch         =       np.concatenate([env_a, img_a], 0)
            r_batch         =       np.concatenate([env_r, img_r], 0)
            done_batch      =       np.concatenate([env_done, img_done], 0)
            next_s_batch    =       np.concatenate([env_next_s, img_next_s], 0)

            s_batch, a_batch, r_batch, done_batch, next_s_batch = self._pack_batch(
                s_batch, a_batch, r_batch, done_batch, next_s_batch
            )

            # training value function
            loss_value      =       self._compute_value_loss(s_batch, a_batch, r_batch, done_batch, next_s_batch)
            self.optimizer_value.zero_grad()
            loss_value.backward()
            self.optimizer_value.step()
            self.loss_log['loss_value'] = loss_value.detach().cpu().item()

            if self.training_count % self.training_delay == 0:
                # train policy
                loss_policy, new_a_log_prob = self._compute_policy_loss(s_batch)
                self.optimizer_policy.zero_grad()
                loss_policy.backward()
                self.optimizer_policy.step()
                self.loss_log['loss_policy']    = loss_policy.cpu().item()

                if self.train_alpha:
                    loss_alpha  =  (- torch.exp(self.log_alpha) * (new_a_log_prob.detach() + self.target_entropy)).mean()
                    self.optimizer_alpha.zero_grad()
                    loss_alpha.backward()
                    self.optimizer_alpha.step()
                    self.alpha = torch.exp(self.log_alpha)
                    
                    self.loss_log['alpha'] = self.alpha.detach().cpu().item()
                    self.loss_log['loss_alpha'] = loss_alpha.detach().cpu().item()

                # soft update target networks
                soft_update(self.QFunction, self.QFunction_tar, self.tau)
            
            self.training_count += 1

    def _compute_value_loss(self, s: torch.tensor, a: torch.tensor, r: torch.tensor, done: torch.tensor, next_s: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            next_a, next_a_log_prob, _          = self.policy(next_s)
            next_sa_value_1, next_sa_value_2    = self.QFunction_tar(next_s, next_a)
            next_sa_value                       = torch.min(next_sa_value_1, next_sa_value_2) 
            target_value                        = r + (1 - done) * self.gamma * (next_sa_value - self.alpha * next_a_log_prob)
        sa_value_1, sa_value_2 = self.QFunction(s, a)
        return F.mse_loss(sa_value_1, target_value) + F.mse_loss(sa_value_2, target_value)

    def _compute_policy_loss(self, s: torch.tensor) -> torch.tensor:
        a, a_log_prob, _    =   self.policy(s)
        q1_value, q2_value  =   self.QFunction(s, a)
        q_value             =   torch.min(q1_value, q2_value)
        neg_entropy         =   a_log_prob
        return (- q_value + self.alpha * neg_entropy).mean(), a_log_prob

    def evaluate(self, env, num_episode: int) -> float:
        total_r = 0
        for _ in range(num_episode):
            s = env.reset()
            done = False
            while not done:
                a = self.sample_action(s, False)
                s_, r, done, _ = env.step(a)
                total_r += r
                s = s_
        return total_r / num_episode

    def save_all_module(self, remark: str) -> None:
        model_path = self.exp_path + 'model/'
        confirm_path_exist(model_path)
        model_path = model_path + f'{remark}'
        torch.save({
            'policy':           self.policy.state_dict(),
            'value':            self.QFunction.state_dict(),
            'dynamics':         self.dynamics.state_dict()
        }, model_path)
        print(f"------- All modules saved to {model_path} ----------")