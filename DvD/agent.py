from typing import Dict, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import Policy, Twin_Q
from utils import soft_update
from memory import Memory

class TD3_agent(object):
    def __init__(self, config: Dict) -> None:
        super(TD3_agent).__init__()

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.noise_std = config['noise_std']
        self.noise_clip = config['noise_clip']
        self.a_max = config['a_max']
        self.a_min = config['a_min']
        self.batch_size = config['batch_size']
        self.update_delay = config['update_delay']
        self.device = torch.device(config['device'])

        self.policy = Policy(config).to(self.device)
        self.policy_target = Policy(config).to(self.device)

        self.twin_q = Twin_Q(config).to(self.device)
        self.twin_q_target = Twin_Q(config).to(self.device)

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_q = optim.Adam(self.twin_q.parameters(), lr=self.lr)

        self.policy_target.load_state_dict(self.policy.state_dict())
        self.twin_q_target.load_state_dict(self.twin_q.state_dict())

        self.memory = Memory(config['memory_size'])

    def choose_action(self, obs: np.array, use_noise: bool = True) -> np.array:
        obs = torch.from_numpy(obs).to(self.device).float()
        with torch.no_grad():
            action = self.policy(obs)
            if use_noise:
                noise = torch.randn_like(action, dtype=torch.float, device=self.device)*self.noise_std
                action = action + noise
        action = np.clip(action.cpu().numpy(), self.a_min, self.a_max)
        return action
    
    def get_target_action(self, next_obs_batch: torch.tensor) -> torch.tensor:
        target_action = self.policy_target(next_obs_batch)
        noise = (torch.randn_like(target_action) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
        return target_action + noise
    
    def update_critic(self, obs_batch: torch.tensor, a_batch: torch.tensor, next_obs_batch: torch.tensor, r_batch: torch.tensor, done_batch: torch.tensor) -> float:
        next_action_target = self.get_target_action(next_obs_batch)
        Q1_next, Q2_next = self.twin_q_target(next_obs_batch, next_action_target)     
        Q_target = r_batch + (1 - done_batch) * self.gamma * torch.min(Q1_next, Q2_next)
        Q1_predict, Q2_predict = self.twin_q(obs_batch, a_batch)
        
        loss_critic = F.mse_loss(Q1_predict, Q_target) + F.mse_loss(Q2_predict, Q_target)
        self.optimizer_q.zero_grad()
        loss_critic.backward()
        self.optimizer_q.step()

        return loss_critic.item()

    def update_actor(self, obs_batch: torch.tensor) -> float:
        loss_actor = - self.twin_q.Q1_value(obs_batch, self.policy(obs_batch)).mean()
        self.optimizer_pi.zero_grad()
        loss_actor.backward()
        self.optimizer_pi.step()

        return loss_actor.item()

    def update_target(self) -> None:
        soft_update(self.policy, self.policy_target, self.tau)
        soft_update(self.twin_q, self.twin_q_target, self.tau)

    def update(self, step: int) -> None:
        batch = self.memory.sample(batch_size=self.batch_size)
        o, a, r, o_, done = batch
        o = torch.from_numpy(np.array(o)).to(self.device).float()
        a = torch.from_numpy(np.array(a)).to(self.device).float()
        r = torch.from_numpy(np.array(r)).to(self.device).float()
        o_ = torch.from_numpy(np.array(o_)).to(self.device).float()
        done = torch.from_numpy(np.array(done)).to(self.device).int()

        loss_critic = self.update_critic(o, a, r, o_, done)
        loss_actor = 0.

        if step % self.update_delay == 0:
            loss_actor = self.update_actor(o)
            self.update_target()
        
        return loss_actor, loss_critic

    def save_transition(self, transition: List) -> None:
        self.memory.save_trans(transition)