import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import GoalGenerator, GoalDiscriminator


class GoalGAN:
    def __init__(self, args, env_params):
        self.g_dim = env_params['g_dim']
        self.max_episode_steps = env_params['max_episode_steps']
        self.noise_dim = args.noise_dim
        self.goal_coverage_scale = args.goal_coverage_scale
        self.r_max = args.r_max
        self.r_min = args.r_min
        self.lr_G = args.lr_G
        self.lr_D = args.lr_D
        self.initialize_goal_num = args.initialize_goal_num
        self.update_iteration = args.GAN_update_iteration
        self.device = torch.device(args.device)

        self.Generator = GoalGenerator(noise_dim=self.noise_dim, g_dim=self.g_dim).to(self.device)
        self.Discriminator = GoalDiscriminator(g_dim=self.g_dim).to(self.device)

        self.optimizer_G = optim.Adam(self.Generator.parameters(), lr=self.lr_G)
        self.optimizer_D = optim.Adam(self.Discriminator.parameters(), lr=self.lr_D)

        self.total_update_num = 0


    def initialize_gan(self, agent, env):
        goals = np.zeros(shape=[self.initialize_goal_num, self.g_dim], dtype=np.float32)
        for i_goal in range(self.initialize_goal_num):
            obs = env.reset()
            for i_step in range(50):
                #a, log_prob = agent.select_action(obs['observation'], obs['desired_goal'])
                a = agent.select_action(obs['observation'], obs['desired_goal'])
                obs_, reward, done, info = env.step(a)
                obs = obs_
            goals[i_goal] = obs['achieved_goal'].copy()
        returns = agent.evaluate_goal(goals, env)
        labels = self.label_goals(returns)
        self.update(goals, labels)

    def sample_noise(self, size):
        noise = torch.randn(size, self.noise_dim, device=self.device)
        return noise

    def sample_goals(self, num):
        z = self.sample_noise(num)
        with torch.no_grad():
            goals = self.Generator(z) #+ torch.randn(size=self.g_dim, device=self.device)
            goals = torch.clamp(goals, -self.goal_coverage_scale, self.goal_coverage_scale).cpu().detach().numpy()
        return goals

    def label_goals(self, returns):
        labels = [(r > self.r_min)&(r < self.r_max) for r in returns]
        return np.array(labels, dtype=np.float32)

    def update(self, goals, labels, logger=None):
        goals = torch.from_numpy(goals).float().to(self.device)
        labels = torch.from_numpy(labels).float().to(self.device)
        for i in range(self.update_iteration):
            #train Generator
            z = self.sample_noise(len(goals))
            loss_G = torch.mean(self.Discriminator(self.Generator(z))**2)
            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

            #train Discriminator
            z = self.sample_noise(len(goals))
            dis_pred = self.Discriminator(goals)
            loss_D = (labels * (dis_pred - 1)**2 + (1 - labels) * (dis_pred + 1)**2).mean() + ((self.Discriminator(self.Generator(z)) + 1)**2).mean()
            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()

            if logger is not None:
                logger.add_scalar('loss/loss_G', loss_G.cpu().detach().numpy(), self.total_update_num)
                logger.add_scalar('loss/loss_D', loss_D.cpu().detach().numpy(), self.total_update_num)
                self.total_update_num += 1
            

    def save_model(self):
        pass

    def load_model(self):
        pass