import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import ForwardModel, StateEncoder, InverseModel, Policy, QFunction
from numpy.core.fromnumeric import size
from numpy.core.function_base import add_newdoc
from utils import Normalizer, get_state
from buffer import Memory


class DDPG_Agent():
    def __init__(self, args, env_params):
        self.s_dim = env_params['o_dim'] + env_params['g_dim']
        self.a_dim = env_params['a_dim']
        self.f_dim = args.f_dim
        self.action_bound = env_params['action_max']
        self.max_timestep = env_params['max_timestep']
        self.max_episode = args.max_episode
        self.evaluate_episode = args.evaluate_episode
        self.evaluate_interval = args.evaluate_interval
        self.log_interval = args.log_interval
        self.save_model_interval = args.save_model_interval
        self.save_model_start = args.save_model_start

        self.lr = args.lr
        self.lr_model = args.lr_model
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.eta = args.eta
        self.noise_eps = args.noise_eps
        self.device = torch.device(args.device)

        self.normalizer_s = Normalizer(size=self.s_dim, eps=1e-2, clip_range=1.)

        self.memory = Memory(size=args.memory_size, s_dim=self.s_dim, a_dim=self.a_dim)

        self.policy = Policy(s_dim=self.s_dim, a_dim=self.a_dim).to(self.device)
        self.policy_target = Policy(s_dim=self.s_dim, a_dim=self.a_dim).to(self.device)
        self.Q = QFunction(s_dim=self.s_dim, a_dim=self.a_dim).to(self.device)
        self.Q_target = QFunction(s_dim=self.s_dim, a_dim=self.a_dim).to(self.device)

        self.optimizer_p = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_q = optim.Adam(self.Q.parameters(), lr=self.lr)

        self.encoder = StateEncoder(s_dim=self.s_dim, f_dim=self.f_dim).to(self.device)
        self.EnvForward = ForwardModel(f_dim = self.f_dim, a_dim=self.a_dim).to(self.device)
        self.EnvInverse = InverseModel(f_dim = self.f_dim, a_dim=self.a_dim).to(self.device)
        
        self.optimizer_forward = optim.Adam([{'params': self.EnvForward.parameters()}, {'params': self.encoder.parameters()}], lr=self.lr_model)
        self.optimizer_inverse = optim.Adam([{'params': self.EnvInverse.parameters()}, {'params': self.encoder.parameters()}], lr=self.lr_model)

        self.hard_update()

        self.update_num = 0

    def select_action(self, state, train_mode=True):
        s = self.normalize_input(state)
        s = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.policy(s).cpu().numpy()

        if train_mode:
            action += np.random.randn(self.a_dim) * self.noise_eps * self.action_bound #Gaussian Noise
        else:
            pass

        action = np.clip(action, a_min=-self.action_bound, a_max=self.action_bound)
        return action
    
    def get_intrisic_reward(self, s, a, s_):
        s, a, s_ = torch.from_numpy(s).to(self.device).float(), torch.from_numpy(a).to(self.device).float(), torch.from_numpy(s_).to(self.device).float() 
        with torch.no_grad():
            feature = self.encoder(s)
            next_feature_pred = self.EnvForward(feature, a)
            next_feature = self.encoder(s_)
        r_i = self.eta * torch.norm(next_feature_pred - next_feature)
        r_i = torch.clamp(r_i, min=-0.1, max=0.1)
        return r_i.cpu().detach().numpy()

    def train(self, env, logger=None):
        total_step = 0
        loss_pi, loss_q, loss_forward, loss_inverse = 0., 0., 0., 0.
        for i_episode in range(self.max_episode):
            obs = env.reset()
            s = get_state(obs)

            cumulative_r = 0.
            for i_step in range(self.max_timestep):
                a = self.select_action(s)
                obs_, r_e, done, info = env.step(a)
                s_ = get_state(obs_)

                r_i = self.get_intrisic_reward(s, a, s_)
                r = r_e + r_i

                self.memory.store(s, a, r, s_)
                s = s_
            
                if len(self.memory) > self.batch_size:
                    loss_pi, loss_q, loss_forward, loss_inverse = self.learn()
                cumulative_r += r_e
                total_step += 1
                
            print('i_episode: {} total step: {} cumulative reward: {:.4f} is_success: {} '.format(i_episode, total_step, cumulative_r, info['is_success']))
            if logger is not None and i_episode % self.log_interval == 0:
                logger.add_scalar('Indicator/cumulative reward', cumulative_r, i_episode)
                logger.add_scalar('Loss/pi_loss', loss_pi, i_episode)
                logger.add_scalar('Loss/q_loss', loss_q, i_episode)
                logger.add_scalar('Loss/forward_loss', loss_forward, i_episode)
                logger.add_scalar('Loss/inverse_loss', loss_inverse, i_episode)
            if i_episode % self.evaluate_interval == 0:
                success_rate = self.evaluate(env)
                if logger is not None:
                    logger.add_scalar('Indicator/success rate', success_rate, i_episode)

            if i_episode > self.save_model_start and i_episode % self.save_model_interval == 0:
                self.save_model(remarks='{}_{}'.format(env.spec.id, i_episode))

    def evaluate(self, env, render=False):
        success_count = 0
        for i_episode in range(self.evaluate_episode):
            obs = env.reset()
            s = get_state(obs)
            for i_step in range(self.max_timestep):
                if render:
                    env.render()
                a = self.select_action(s, train_mode=False)
                obs_, r_e, done, info = env.step(a)
                s_ = get_state(obs_)
                s = s_
            success_count += info['is_success']

        return success_count / self.evaluate_episode

    def learn(self):
        s, a, r, s_ = self.memory.sample_batch(batch_size=self.batch_size)
        self.normalizer_s.update(s)

        s, s_ = self.normalize_input(s, s_)
        s = torch.from_numpy(s).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        r = torch.from_numpy(r).to(self.device).unsqueeze(dim=1)
        s_ = torch.from_numpy(s_).to(self.device)
        
        #update policy and Q
        with torch.no_grad():
            a_next_tar = self.policy_target(s_)
            Q_next_tar = self.Q_target(s_, a_next_tar)
            loss_q_tar = r + self.gamma * Q_next_tar
        loss_q_pred = self.Q(s, a)
        loss_q = F.mse_loss(loss_q_pred, loss_q_tar.detach())
        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.optimizer_q.step()

        loss_p = - self.Q(s, self.policy(s)).mean()
        self.optimizer_p.zero_grad()
        loss_p.backward()
        self.optimizer_p.step()

        self.soft_update()

        #update env model and encoder
        feature = self.encoder(s)
        next_feature = self.encoder(s_)
        a_pred = self.EnvInverse(feature, next_feature)
        loss_inverse = F.mse_loss(a_pred, a)

        next_feature_pred = self.EnvForward(feature, a)
        with torch.no_grad():
            next_feature_tar = self.encoder(s_)
        loss_forward = F.mse_loss(next_feature_pred, next_feature_tar.detach())
        
        self.optimizer_forward.zero_grad()
        self.optimizer_inverse.zero_grad()
        loss_forward.backward(retain_graph=True)
        loss_inverse.backward()
        self.optimizer_forward.step()
        self.optimizer_inverse.step()

        self.update_num += 1
        return loss_p.cpu().detach().numpy(), loss_q.cpu().detach().numpy(), loss_forward.cpu().detach().numpy(), loss_inverse.cpu().detach().numpy()


    def update_normalizer(self, states):
        states = np.array(states, dtype=np.float32)
        self.normalizer_s.update(states)

    def hard_update(self):
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.Q_target.load_state_dict(self.Q.state_dict())

    def soft_update(self):
        for param, param_target in zip(self.policy.parameters(), self.policy_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))
        for param, param_target in zip(self.Q.parameters(), self.Q_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))

    def normalize_input(self, s, s_=None):
        s = self.normalizer_s.normalize(s)
        if s_ is not None:
            s_ = self.normalizer_s.normalize(s_)
            return s, s_
        else:
            return s

    def save_model(self, remarks):
        if not os.path.exists('pretrained_models_DDPG/'):
            os.mkdir('pretrained_models_DDPG/')
        path = 'pretrained_models_DDPG/{}.pt'.format(remarks)
        print('Saving model to {}'.format(path))
        torch.save([self.normalizer_s.mean, self.normalizer_s.std, self.policy.state_dict()], path)

    def load_model(self, remark):
        print('Loading models with remark {}'.format(remark))
        self.normalizer_s.mean, self.normalizer_s.std, policy_model = torch.load('pretrained_models_DDPG/{}.pt'.format(remark), map_location=lambda storage, loc: storage)
        self.policy.load_state_dict(policy_model)