import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import Policy, QFunction
from utils import *


class DDPG_Her_Agent():
    def __init__(self, args, env_params):
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.g_dim = env_params['g_dim']
        self.action_bound = env_params['action_max']

        self.lr = args.lr
        self.l2_coefficient = args.l2_coefficient
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.device = torch.device(args.device)
        self.tau = args.tau
        self.noise_eps = args.noise_eps

        self.policy = Policy(o_dim=self.o_dim, a_dim=self.a_dim, g_dim=self.g_dim).to(self.device)
        self.policy_target = Policy(o_dim=self.o_dim, a_dim=self.a_dim, g_dim=self.g_dim).to(self.device)
        self.Q = QFunction(o_dim=self.o_dim, a_dim=self.a_dim, g_dim=self.g_dim).to(self.device)
        self.Q_target = QFunction(o_dim=self.o_dim, a_dim=self.a_dim, g_dim=self.g_dim).to(self.device)
        sync_networks(self.policy)
        sync_networks(self.Q)

        self.optimizer_p = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_q = optim.Adam(self.Q.parameters(), lr=self.lr)

        self.normalizer_o = Normalizer(size=self.o_dim, eps=1e-2, clip_range=1.)
        self.normalizer_g = Normalizer(size=self.g_dim, eps=1e-2, clip_range=1.)

        self.hard_update()

    def hard_update(self):
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.Q_target.load_state_dict(self.Q.state_dict())

    def soft_update(self):
        for param, param_target in zip(self.policy.parameters(), self.policy_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))
        for param, param_target in zip(self.Q.parameters(), self.Q_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))

    def normalize_input(self, o, g, o_=None):
        o = self.normalizer_o.normalize(o)
        g = self.normalizer_g.normalize(g)
        if o_ is not None:
            o_ = self.normalizer_o.normalize(o_)
            return o, g, o_
        else:
            return o, g

    def select_action(self, observation, goal, train_mode=True):
        observation, goal = self.normalize_input(observation, goal)
        observation, goal = torch.tensor(observation, dtype=torch.float32).to(self.device), torch.tensor(goal, dtype=torch.float32).to(self.device)
        o_g = torch.cat([observation, goal], dim=0)
        with torch.no_grad():
            action = self.policy(o_g).cpu().numpy()

        if train_mode:
            action += np.random.randn(self.a_dim) * self.noise_eps * self.action_bound #Gaussian Noise
        else:
            pass
        action = np.clip(action, a_min=-self.action_bound, a_max=self.action_bound)
        return action

    def learn(self, memory):
        o, a, r, o_, g = memory.sample_batch(batch_size=self.batch_size)
        o, g, o_ = self.normalize_input(o, g, o_)
        o = torch.from_numpy(o).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        r = torch.from_numpy(r).to(self.device).unsqueeze(dim=1)
        o_ = torch.from_numpy(o_).to(self.device)
        g = torch.from_numpy(g).to(self.device)

        #update Q
        a_next_target = self.policy_target(torch.cat([o_, g], dim=1))
        q_tar = r + self.gamma * self.Q_target(torch.cat([o_, a_next_target, g], dim=1))
        q_tar = torch.clamp(q_tar, -1/(1-self.gamma), 0)
        q_pred = self.Q(torch.cat([o, a, g], dim=1))
        loss_q = F.mse_loss(q_pred, q_tar.detach())
        self.optimizer_q.zero_grad()
        loss_q.backward()
        sync_grads(self.Q)
        self.optimizer_q.step()

        #update policy
        a_eval = self.policy(torch.cat([o, g], dim=1))
        loss_p = - self.Q(torch.cat([o, a_eval, g], dim=1)).mean() + self.l2_coefficient * (a_eval/self.action_bound).pow(2).mean() #actions
        self.optimizer_p.zero_grad()
        loss_p.backward()
        sync_grads(self.policy)
        self.optimizer_p.step()

        return loss_q.cpu().item(), loss_p.cpu().item(), q_pred.mean().cpu().item()

    def update_normalizer(self, observations, goals):
        observations, goals = np.array(observations, dtype=np.float32), np.array(goals, dtype=np.float32)
        self.normalizer_o.update(observations)
        self.normalizer_g.update(goals)

    def save_model(self, remarks):
        if not os.path.exists('pretrained_models_DDPG/'):
            os.mkdir('pretrained_models_DDPG/')
        path = 'pretrained_models_DDPG/{}.pt'.format(remarks)
        print('Saving model to {}'.format(path))
        torch.save([self.normalizer_o.mean, self.normalizer_o.std, self.normalizer_g.mean, self.normalizer_g.std, self.policy.state_dict()], path)

    def load_model(self, remarks):
        print('Loading models with remark {}'.format(remarks))
        self.normalizer_o.mean, self.normalizer_o.std, self.normalizer_g.mean, self.normalizer_g.std, policy_model = torch.load('pretrained_models_DDPG/{}.pt'.format(remarks), map_location=lambda storage, loc: storage)
        self.policy.load_state_dict(policy_model)