from typing import Dict, List, Tuple, Type
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pipe

from env import Env_wrapper
from buffer import Buffer
from worker import Worker
from model import Policy, TwinQFuncion
from utils import MeanStdFilter, check_path, soft_update, hard_update


class Master(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        
        self.env_config = config['env_config']
        self.model_config = config['model_config']
        self.device = torch.device(config['device'])
        self.num_workers = config['num_workers']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.exp_path = config['exp_path']
        self.train_policy_delay = config['train_policy_delay']
        self.soft_update_interval = config['soft_update_interval']

        self.learning_rate = config['lr']
        self.gamma = config['gamma']
        self.rho = config['rho']
        self.target_entropy = - torch.tensor(self.model_config['a_dim'], device=self.device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha)

        self.episode_reward_max, self.episode_reward_min = 10, 0

        self.train_count = 0
        self.best_score = -1000
        
        self.init_buffer()
        self.init_value_functions()
        self.init_env()
        self.init_policy()
        self.init_workers()
        self.init_optimizer()
        self.init_obs_filter()
        self.init_logger_value()
        
    def init_buffer(self) -> None:
        self.buffer = Buffer(self.buffer_size)

    def init_env(self) -> None:
        self.env = Env_wrapper(self.env_config)

    def init_value_functions(self) -> None:
        self.Q = TwinQFuncion(self.model_config, self.device)
        self.Q_tar = TwinQFuncion(self.model_config, self.device)
        hard_update(source_model=self.Q, target_model=self.Q_tar)

    def init_policy(self) -> None:
        self.policy = Policy(self.model_config, self.device)

    def init_workers(self) -> None:
        self.parents_conn = []
        self.child_conn = []
        self.workers = []
        for i in range(self.num_workers):
            parent_conn, child_conn = Pipe()
            worker = Worker(i, Env_wrapper(self.env_config), child_conn)
            worker.start()
            self.parents_conn.append(parent_conn)
            self.child_conn.append(child_conn)
            self.workers.append(worker)

    def init_optimizer(self) -> None:
        self.optimizer_policy = optim.Adam(self.policy.parameters(), self.learning_rate)
        self.optimizer_q = optim.Adam(self.Q.parameters(), self.learning_rate)
        self.optimizer_alpha = optim.Adam([self.log_alpha], self.learning_rate)

    def init_obs_filter(self) -> None:
        self.obs_filter = MeanStdFilter(shape=self.model_config['o_dim'])

    def init_logger_value(self) -> None:
        self.log_q_value = 0
        self.log_q_loss = 0
        self.log_policy_loss = 0
        self.log_entropy_loss = 0
        self.log_alpha_value = 0

    def transfrom_trans_and_save(self, obs_seq: List[np.array], a_seq: List[np.array], episode_r: float, done_seq: List[bool], next_obs_seq: List[np.array]) -> None:
        assert len(obs_seq) == len(a_seq) == len(done_seq) == len(next_obs_seq)
        r_seq = [episode_r for _ in range(len(obs_seq))]
        self.buffer.push_batch(list(zip(obs_seq, a_seq, r_seq, done_seq, next_obs_seq)))
        self.episode_reward_max = max(episode_r, self.episode_reward_max)
        self.episode_reward_min = min(episode_r, self.episode_reward_min)

    def train(self) -> None:
        obs_batch, a_batch, r_batch, done_batch, next_obs_batch = self.buffer.sample(self.batch_size)

        self.obs_filter.push_batch(obs_batch)
        obs_batch, next_obs_batch = self.obs_filter.trans_batch(obs_batch), self.obs_filter.trans_batch(next_obs_batch)

        r_batch = (r_batch - self.episode_reward_min) / (self.episode_reward_max - self.episode_reward_min)

        obs_batch = torch.from_numpy(obs_batch).float().to(self.device)
        a_batch = torch.from_numpy(a_batch).float().to(self.device)
        r_batch = torch.from_numpy(r_batch).float().to(self.device).unsqueeze(dim=-1)
        done_batch = torch.from_numpy(done_batch).int().to(self.device).unsqueeze(dim=-1)
        next_obs_batch = torch.from_numpy(next_obs_batch).float().to(self.device)

        with torch.no_grad():
            next_action_batch, next_action_logprod = self.policy.forward_batch(obs_batch)
            q1_next_tar, q2_next_tar = self.Q_tar(next_obs_batch, next_action_batch)
            q_tar = torch.min(q1_next_tar, q2_next_tar)
            q_update_target = r_batch + (1 - done_batch) * self.gamma * (q_tar - self.alpha * next_action_logprod)
        q1_update_eval, q2_update_eval = self.Q(obs_batch, a_batch)
        loss_q = F.mse_loss(q1_update_eval, q_update_target) + F.mse_loss(q2_update_eval, q_update_target)
        self.optimizer_q.zero_grad()
        loss_q.backward(retain_graph=True)
        nn.utils.clip_grad.clip_grad_norm_(self.Q.parameters(), max_norm=1.)
        self.optimizer_q.step()
        
        if self.train_count % self.train_policy_delay == 0:
            a_new_batch, a_new_logprob = self.policy.forward_batch(obs_batch)
            q1_new_value, q2_new_value = self.Q(obs_batch, a_new_batch)
            q_new_value = torch.min(q1_new_value, q2_new_value)
            loss_policy = (self.alpha * a_new_logprob - q_new_value).mean()
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            nn.utils.clip_grad.clip_grad_norm_(self.policy.parameters(), max_norm=1.)
            self.optimizer_policy.step()

            a_new_logprob = torch.tensor(a_new_logprob.tolist(), requires_grad=False, device=self.device)
            loss_alpha = (- torch.exp(self.log_alpha) * (a_new_logprob + self.target_entropy)).mean()
            self.optimizer_alpha.zero_grad()
            loss_alpha.backward()
            self.optimizer_alpha.step()
            nn.utils.clip_grad.clip_grad_norm_([self.log_alpha], max_norm=1.)
            self.alpha = torch.exp(self.log_alpha)

            self.log_policy_loss = loss_policy.item()
            self.log_entropy_loss = loss_alpha.item()
            self.log_alpha_value = self.alpha.item()
    
        if self.train_count % self.soft_update_interval == 0:
            soft_update(self.Q, self.Q_tar, self.rho)

        self.train_count += 1
        self.log_q_value = q1_update_eval.mean().item() + q2_update_eval.mean().item()
        self.log_q_loss = loss_q.item()

        return self.log_q_value/2, self.log_q_loss/2, self.log_policy_loss, self.log_entropy_loss, self.log_alpha_value

    def evaluation(self, evaluation_rollouts: int) -> float:
        accumulate_r = 0
        for rollout in range(evaluation_rollouts):
            done = False
            obs = self.env.reset()
            while not done:
                a = self.policy(self.obs_filter(obs), evaluation=True)
                obs, r, done, _ = self.env.step(a)
                accumulate_r += r
        return accumulate_r / evaluation_rollouts

    def save_policy(self, remark: str) -> None:
        model_path = self.exp_path + 'models/'
        check_path(model_path)
        torch.save(self.policy.state_dict(), model_path+f'model_{remark}')
        print(f"-------Policy model saved to {model_path}-------")

    def save_filter(self, remark: str) -> None:
        filter_path = self.exp_path + 'filters/'
        check_path(filter_path)
        filter_params = np.array([self.obs_filter.mean, self.obs_filter.square_sum, self.obs_filter.count])
        np.save(filter_path + remark, filter_params)
        print(f"-------Filter params saved to {filter_path}-------")