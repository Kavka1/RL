import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import os
from model import QFunction, VFunction, GaussianPolicy, DiscretPolicy

class SACAgent():
    def __init__(
            self,
            s_dim,
            a_dim,
            lr_pi,
            lr_q,
            lr_alpha,
            gamma,
            alpha_type,
            alpha,
            tau,
            batch_size,
            target_update_interval,
            action_space = None,
    ):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.action_space = action_space
        self.lr_pi = lr_pi
        self.lr_q = lr_q
        self.lr_alpha = lr_alpha
        self.alpha_type = alpha_type
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if alpha_type == 'learn':
            self.entropy_target = -torch.prod(torch.tensor(action_space.shape)).to(self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.optimizer_alpha = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = alpha
        else:
            self.alpha = alpha

        self.policy = GaussianPolicy(s_dim=s_dim, a_dim=a_dim, action_space=action_space, std=None, device=self.device).to(self.device)
        self.Q_1 = QFunction(s_dim, a_dim).to(self.device)
        self.Q_2 = QFunction(s_dim, a_dim).to(self.device)
        self.Q_1_target = QFunction(s_dim, a_dim).to(self.device)
        self.Q_2_target = QFunction(s_dim, a_dim).to(self.device)

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=lr_pi)
        self.optimizer_Q1 = optim.Adam(self.Q_1.parameters(), lr=lr_q)
        self.optimizer_Q2 = optim.Adam(self.Q_2.parameters(), lr=lr_q)

        self.hard_update_target()

    def hard_update_target(self):
        self.Q_1_target.load_state_dict(self.Q_1.state_dict())
        self.Q_2_target.load_state_dict(self.Q_2.state_dict())

    def soft_update_target(self):
        for param, param_target in zip(self.Q_1.parameters(), self.Q_1_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))
        for param, param_target in zip(self.Q_2.parameters(), self.Q_2_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))

    def choose_action(self, s):
        s = torch.from_numpy(s).to(self.device).float().unsqueeze(0)
        action, log_prob, mean = self.policy.sample(s)
        return action.cpu().numpy()[0]

    def learn(self, memory, update_num):
        s, a, r, s_next, done = memory.sample_batch(self.batch_size)
        s, a, r, s_next, done = torch.from_numpy(s).to(self.device), torch.from_numpy(a).to(self.device), torch.from_numpy(r).to(self.device).unsqueeze(dim=1), torch.from_numpy(s_next).to(self.device), torch.from_numpy(done).to(self.device).unsqueeze(dim=1)

        a_next, log_prob_a_next, _ = self.policy.sample(s_next)
        q_target_next = torch.min(self.Q_1_target(s_next, a_next), self.Q_2_target(s_next, a_next))
        #Q1 update
        q1_loss_target = r + (1 - done) * self.gamma * (q_target_next - self.alpha * log_prob_a_next)
        q1_loss_pred = self.Q_1(s, a)
        q1_loss = F.mse_loss(q1_loss_pred, q1_loss_target.detach())
        self.optimizer_Q1.zero_grad()
        q1_loss.backward()
        self.optimizer_Q1.step()

        #Q2 update
        q2_loss_target = r + (1 - done) * self.gamma * (q_target_next - self.alpha * log_prob_a_next)
        q2_loss_pred = self.Q_2(s, a)
        q2_loss = F.mse_loss(q2_loss_pred, q2_loss_target.detach())
        self.optimizer_Q2.zero_grad()
        q2_loss.backward()
        self.optimizer_Q2.step()

        #pi update
        action, log_prob_a, mean = self.policy.sample(s)
        min_q = torch.min(self.Q_1(s, action), self.Q_2(s, action))
        pi_loss = (self.alpha * log_prob_a - min_q).mean()
        self.optimizer_pi.zero_grad()
        pi_loss.backward()
        self.optimizer_pi.step()

        #update alpha
        if self.alpha_type == 'learn':
            loss_alpha = - (self.log_alpha * (log_prob_a + self.entropy_target).detach()).mean()
            self.optimizer_alpha.zero_grad()
            loss_alpha.backward()
            self.optimizer_alpha.step()
            self.alpha = self.log_alpha.exp()
            self.alpha = torch.max(self.alpha, torch.tensor(0.1).to(self.device))
            alpha_tlogs = self.alpha.clone()
        else:
            loss_alpha = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if update_num % self.target_update_interval == 0:
            self.soft_update_target()

        return q1_loss.cpu().item(), q2_loss.cpu().item(), pi_loss.cpu().item(), loss_alpha.cpu().item(), alpha_tlogs.cpu().item()

    def save_model(self, env_name, remarks='', pi_path=None, q1_path=None, q2_path=None):
        if not os.path.exists('pretrained_models/'):
            os.mkdir('pretrained_models/')

        if pi_path == None:
            pi_path = 'pretrained_models/policy_{}_{}'.format(env_name, remarks)
        if q1_path == None:
            q1_path = 'pretrained_models/q1_{}_{}'.format(env_name, remarks)
        if q2_path == None:
            q2_path = 'pretrained_models/q2_{}_{}'.format(env_name, remarks)
        print('Saving model to {} , {} and {}'.format(pi_path, q1_path, q2_path))
        torch.save(self.policy.state_dict(), pi_path)
        torch.save(self.Q_1.state_dict(), q1_path)
        torch.save(self.Q_2.state_dict(), q2_path)

    def load_model(self, pi_path, q1_path, q2_path):
        print('Loading models from {} , {} and {}'.format(pi_path, q1_path, q2_path))
        self.policy.load_state_dict(pi_path)
        self.Q_1.load_state_dict(torch.load(q1_path))
        self.Q_2.load_state_dict(torch.load(q2_path))