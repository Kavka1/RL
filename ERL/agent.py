import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import DeterministicPolicy, QFunction, GaussianPolicy, TwinQFunction


class DDPGAgent():
    def __init__(self, args, env_params):
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.action_boundary = env_params['action_boundary']

        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.noise_eps = args.noise_eps
        self.batch_size = args.batch_size

        self.device = torch.device(args.device)

        self.actor = DeterministicPolicy(self.o_dim, self.a_dim).to(self.device)
        self.actor_tar = DeterministicPolicy(self.o_dim, self.a_dim).to(self.device)
        self.critic = QFunction(self.o_dim, self.a_dim).to(self.device)
        self.critic_tar = QFunction(self.o_dim, self.a_dim).to(self.device)

        self.optimizer_a = optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.hard_update()

    def hard_update(self):
        self.actor_tar.load_state_dict(self.actor.state_dict())
        self.critic_tar.load_state_dict(self.critic.state_dict())

    def soft_update(self):
        for params, params_tar in zip(self.actor.parameters(), self.actor_tar.parameters()):
            params_tar.data.copy_(self.tau * params.data + (1 - self.tau) * params_tar.data)
        for params, params_tar in zip(self.critic.parameters(), self.critic_tar.parameters()):
            params_tar.data.copy_(self.tau * params.data + (1 - self.tau) * params_tar.data)

    def choose_action(self, obs, is_evaluete=False):
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            action = self.actor(obs)
        if not is_evaluete:
            action += torch.normal(torch.tensor(0.), torch.tensor(self.noise_eps))
        action = torch.clamp(action, -self.action_boundary, self.action_boundary).cpu().detach().numpy()
        return action

    def rollout(self, env, memory, is_evaluate=False):
        total_reward = 0.
        obs = env.reset()
        done = False
        while not done:
            a = self.choose_action(obs, is_evaluate)
            obs_, r, done, info = env.step(a)

            memory.store(obs, a, r, obs_, done)
            
            total_reward += r
            obs = obs_
        return total_reward

    def update(self, memory):
        obs, a, r, obs_, done = memory.sample_batch(self.batch_size)

        obs = torch.from_numpy(obs).float().to(self.device)
        a = torch.from_numpy(a).float().to(self.device)
        r = torch.from_numpy(r).float().to(self.device)
        obs_ = torch.from_numpy(obs_).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        with torch.no_grad():
            next_action_tar = self.actor_tar(obs_)
            next_q_tar = self.critic_tar(obs_, next_action_tar)
            critic_target = r + (1 - done) * self.gamma * next_q_tar
        critic_eval = self.critic(obs, a)
        loss_critic = F.mse_loss(critic_eval, critic_target.detach())
        self.optimizer_c.zero_grad()
        loss_critic.backward()
        self.optimizer_c.step()

        loss_actor = - self.critic(obs, self.actor(obs)).mean()
        self.optimizer_a.zero_grad()
        loss_actor.backward()
        self.optimizer_a.step()

        self.soft_update()

    def save_model(self, remark):
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')
        path = 'pretrained_model/{}.pt'.format(remark)
        print('Saving model to {}'.format(path))
        torch.save(self.actor.state_dict(), path)

    def load_model(self, remark):
        path = 'pretrained_model/{}.pt'.format(remark)
        print('Loading model from {}'.format(path))
        model = torch.load(path)
        self.actor.load_state_dict(model)


class SACAgent():
    def __init__(self, args, env_params):
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.action_boundary = env_params['action_boundary']
        self.device = torch.device(args.device)

        self.lr = args.lr
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.batch_size = args.batch_size

        self.target_entropy = - torch.prod(torch.Tensor(self.a_dim).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.policy = GaussianPolicy(self.o_dim, self.a_dim).to(self.device)
        self.critic = TwinQFunction(self.o_dim, self.a_dim).to(self.device)
        self.critic_target = TwinQFunction(self.o_dim, self.a_dim).to(self.device)

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_q = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.optimizer_alpha = optim.Adam([self.log_alpha], lr=self.lr)

        self.hard_update()

    def hard_update(self):
        self.critic_target.load_state_dict(self.critic.state_dict())

    def soft_update(self):
        for param, param_tar in zip(self.critic.parameters(), self.critic_target.parameters()):
            param_tar.data.copy_(self.tau * param.data + (1 - self.tau) * param_tar.data)
    
    def rollout(self, env, memory, is_evaluate=False):
        total_reward = 0.
        obs = env.reset()
        done = False
        while not done:
            a = self.choose_action(obs, is_evaluate)
            obs_, r, done, info = env.step(a)

            memory.store(obs, a, r, obs_, done)
            
            total_reward += r
            obs = obs_
        return total_reward

    def choose_action(self, obs, is_evaluate=False):
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            action, log_prob, mean = self.policy(obs)
        if is_evaluate:
            action = mean
        action = action.cpu().detach().numpy()
        action = np.clip(action, -self.action_boundary, self.action_boundary)
        return action

    def update(self, memory, update_count):
        obs, a, r, obs_, done = memory.sample_batch(self.batch_size)

        obs = torch.from_numpy(obs).float().to(self.device)
        a = torch.from_numpy(a).float().to(self.device)
        r = torch.from_numpy(r).float().to(self.device)
        obs_ = torch.from_numpy(obs_).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy(obs_)
            Q_next_1, Q_next_2 = self.critic_target(obs_, next_action)
            Q_next = torch.min(Q_next_1, Q_next_2)
            critic_tar = r + (1- done) * self.gamma * (Q_next - self.alpha * next_log_prob)
        critic_eval_1, crtic_eval_2 = self.critic(obs, a)
        loss_critic = F.mse_loss(critic_eval_1, critic_tar) + F.mse_loss(crtic_eval_2, critic_tar)
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

        self.soft_update()

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