from io import SEEK_CUR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from model import TwinQFunction, GaussianPolicy
from memory import Memory


class SACAgent():
    def __init__(self, args, env_params):
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.action_scale = np.array(env_params['action_scale'], dtype=np.float32)
        self.action_bias = np.array(env_params['action_bias'], dtype=np.float32)
        self.action_boundary = env_params['action_boundary']
        self.device = torch.device(args.device)

        self.lr = args.lr
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size
        self.target_update_interval = args.target_update_interval

        self.memory = Memory(self.memory_size, self.o_dim, self.a_dim)

        self.policy = GaussianPolicy(self.o_dim, self.a_dim).to(self.device)
        self.critic = TwinQFunction(self.o_dim, self.a_dim).to(self.device)
        self.critic_target = TwinQFunction(self.o_dim, self.a_dim).to(self.device)
        
        self.target_entropy = - torch.prod(torch.Tensor(self.a_dim).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_q = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.optimizer_alpha = optim.Adam([self.log_alpha], lr=self.lr)

        self.hard_update_target()

    def hard_update_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())

    def soft_update_target(self):
        for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))

    def choose_action(self, obs, is_evluate=False):
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            action, log_prob, mean = self.policy(obs)
        if is_evluate:
            action = mean
        action = action.cpu().detach().numpy()
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_boundary[0], self.action_boundary[1])
        return action

    def update(self, update_count):
        obs, a, r, obs_, done = self.memory.sample_batch(self.batch_size)

        obs = torch.from_numpy(obs).float().to(self.device)
        a = torch.from_numpy(a).float().to(self.device)
        r = torch.from_numpy(r).float().to(self.device)
        obs_ = torch.from_numpy(obs_).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy(obs_)
            Q_next_1, Q_next_2 = self.critic_target(obs_, next_action)
            Q_next = torch.min(Q_next_1, Q_next_2) 
            critic_tar = r + (1 - done) * self.gamma * (Q_next - self.alpha * next_log_prob)
        critic_eval_1, critic_eval_2 = self.critic(obs, a)
        loss_critic = F.mse_loss(critic_eval_1, critic_tar) + F.mse_loss(critic_eval_2, critic_tar)
        self.optimizer_q.zero_grad()
        loss_critic.backward()
        self.optimizer_q.step()

        action, log_prob, _ = self.policy(obs)
        q_1, q_2 = self.critic(obs, action)
        q_value = torch.min(q_1, q_2)
        loss_pi = (self.alpha * log_prob - q_value).mean()
        self.optimizer_pi.zero_grad()
        loss_pi.backward(retain_graph = True)
        self.optimizer_pi.step()

        loss_alpha = - (self.log_alpha * (log_prob + self.target_entropy)).mean()
        self.optimizer_alpha.zero_grad()
        loss_alpha.backward()
        self.optimizer_alpha.step()
        self.alpha = torch.exp(self.log_alpha)

        if update_count % self.target_update_interval == 0 and update_count > 0:
            self.soft_update_target()

        return loss_critic.item(), loss_pi.item(), loss_alpha.item(), self.alpha.clone().item()

    def save_model(self, remarks):
        if not os.path.exists('pretrained_models/'):
            os.mkdir('pretrained_models/')
        path = 'pretrained_models/{}.pt'.format(remarks)
        print('Saving model to {}'.format(path))

        pretrained_model = dict(
            policy = self.policy.state_dict(),
            critic = self.critic.state_dict()
        )
        torch.save(pretrained_model, path)

    def load_model(self, remarks):
        path = 'pretrained_models/{}.pt'.format(remarks)
        print('Loading models from {}'.format(path))
        
        model = torch.load(path)
        self.policy.load_state_dict(model['policy'])
        self.critic.load_state_dict(model['critic'])