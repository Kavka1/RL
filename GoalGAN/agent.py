import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import GaussianPolicy, VFunction, QFunction
from buffer import TrajectoryBuffer, MemoryBuffer


class PPOAgent():
    def __init__(self, args, env_params):
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.g_dim = env_params['g_dim']
        self.action_boundary = env_params['action_boundary']
        self.max_episode_steps = env_params['max_episode_steps']

        self.evaluate_episodes = args.evaluate_episodes
        self.lr_pi = args.lr_pi
        self.lr_v = args.lr_v
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.action_var = args.action_var
        self.clip_range = args.clip_range
        self.temperature_coef = args.temperature_coef
        self.K_updates = args.K_updates
        self.device = torch.device(args.device)
        
        self.load_model_remark = args.load_model_remark

        self.total_trained_goal_num = 0
        self.total_episode_num = 0
        self.total_update_num = 0

        self.buffer = TrajectoryBuffer(self.max_episode_steps, self.o_dim, self.g_dim, self.a_dim)

        self.policy = GaussianPolicy(self.o_dim, self.g_dim, self.a_dim, self.action_var, self.device).to(self.device)
        self.policy_old = GaussianPolicy(self.o_dim, self.g_dim, self.a_dim, self.action_var, self.device).to(self.device)
        self.V = VFunction(self.o_dim, self.g_dim).to(self.device)
        self.V_old = VFunction(self.o_dim, self.g_dim).to(self.device)

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.V.parameters(), lr=self.lr_v)

        self.hard_update()

    def select_action(self, observation, goal):
        observation = torch.from_numpy(observation).float().to(self.device)
        goal = torch.from_numpy(goal).float().to(self.device)
        input_tensor = torch.cat([observation, goal], dim=0)
        with torch.no_grad():
            dist = self.policy_old(input_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).cpu().detach().numpy()
            
            action = action.cpu().detach().numpy()
            action = np.clip(action, -self.action_boundary, self.action_boundary)

        return action, log_prob

    def train_and_evaluate(self, goals, env, logger = None):
        returns = np.zeros(shape=[len(goals)], dtype=np.float32)
        
        for i_goal, goal in enumerate(goals):
            success_count = 0
            cumulative_r = 0. #for log
            cumulative_loss_pi, cumulative_loss_v = 0., 0.
            print('--{} goal: ({:.4f}, {:.4f}):-----------------------'.format(self.total_trained_goal_num, goal[0], goal[1]))

            for i_episode in range(self.evaluate_episodes):
                success_flag = 0
                self.buffer.clear()
                
                _ = env.reset()
                obs = env.set_goal(goal)
                for i_step in range(self.max_episode_steps):
                    #if i_goal % 100 == 0 and i_episode == 0:
                    #    env.render()
                    a, log_prob = self.select_action(obs['observation'], obs['desired_goal'])
                    obs_, reward, done, info = env.step(a)
                    
                    cumulative_r += reward
                    self.buffer.store(obs['observation'], a, reward, obs_['observation'], obs['desired_goal'], log_prob)
                    
                    if info['is_success'] == 1:
                        success_flag = 1
                        break

                    obs = obs_
                
                loss_pi, loss_v = self.update()
                success_count += success_flag
                
                cumulative_loss_pi += loss_pi
                cumulative_loss_v += loss_v

            average_success = success_count / self.evaluate_episodes
            returns[i_goal] = average_success
            
            if logger is not None:
                logger.add_scalar('Indicator/episode_reward', cumulative_r/self.evaluate_episodes, self.total_episode_num)
                logger.add_scalar('Indicator/goal_success_rate', average_success, self.total_trained_goal_num)
                logger.add_scalar('loss/loss_pi', cumulative_loss_pi/self.evaluate_episodes, self.total_update_num)
                logger.add_scalar('loss/loss_v', cumulative_loss_v/self.evaluate_episodes, self.total_update_num)
            self.total_trained_goal_num += 1
            self.total_episode_num += self.evaluate_episodes
            self.total_update_num += self.K_updates * self.evaluate_episodes

            print('\t success_rate: {:.2f}'.format(average_success))
            print('\t average_episode_return: {:.4f}'.format(cumulative_r/self.evaluate_episodes))

        return returns

    def update(self):
        o, a, r, o_, g, log_prob = self.buffer.get_trajectory()
        o = torch.from_numpy(o).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        g = torch.from_numpy(g).to(self.device)
        o_ = torch.from_numpy(o_).to(self.device)
        log_prob = torch.from_numpy(log_prob).to(self.device)

        inputs_tensor = torch.cat([o, g], dim=1)
        values = self.V_old(inputs_tensor).squeeze(dim=1).detach().cpu().numpy()

        returns, deltas, advantages = self.gae_estimator(r, values)
        returns = torch.from_numpy(returns).to(self.device)
        returns = (returns - returns.mean())/(returns.std() + 1e-3)
        advantages = torch.from_numpy(advantages).to(self.device)

        for i in range(self.K_updates):
            dist = self.policy(inputs_tensor)
            new_log_prob = dist.log_prob(a)
            entropy = dist.entropy()
            state_values = self.V(inputs_tensor).squeeze(dim=1)

            ratio = torch.exp(new_log_prob - log_prob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            loss_pi = (- torch.min(surr1, surr2)).mean() - (self.temperature_coef * entropy).mean()
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()

            loss_v = F.mse_loss(state_values, returns)
            self.optimizer_v.zero_grad()
            loss_v.backward()
            self.optimizer_v.step()

        self.hard_update()

        return loss_pi.cpu().detach().item(), loss_v.cpu().detach().item()


    def evaluate_goal(self, goals, env):
        returns = np.zeros(shape=[len(goals)], dtype=np.float32)
        for i_goal, goal in enumerate(goals):
            success_count = 0
            for i_episode in range(self.evaluate_episodes):
                _ = env.reset()
                obs = env.set_goal(goal)
                for i_step in range(self.max_episode_steps):
                    #env.render()
                    a, log_prob = self.select_action(obs['observation'], obs['desired_goal'])
                    obs_, reward, done, info = env.step(a)
                    if info['is_success'] == 1:
                        success_count += 1
                        break
                    obs = obs_
            average_success = success_count / self.evaluate_episodes
            print('{} goal: {} {}   return: {}'.format(i_goal, goal[0], goal[1], average_success))
            returns[i_goal] = average_success
        return returns

    def gae_estimator(self, rewards, state_values):
        size = len(rewards)
        returns = np.zeros(shape=[size], dtype=np.float32)
        deltas = np.zeros(shape=[size], dtype=np.float32)
        advantages = np.zeros(shape=[size], dtype=np.float32)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(size)):
            returns[i] = rewards[i] + self.gamma * prev_return
            deltas[i] = rewards[i] + self.gamma * prev_value - state_values[i]
            advantages[i] = deltas[i] + self.gamma * self.lamda * prev_advantage

            prev_advantage = advantages[i]
            prev_value = state_values[i]
            prev_return = returns[i]

        return returns, deltas, advantages

    def hard_update(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.V_old.load_state_dict(self.V.state_dict())

    def save_model(self, remark):
        if not os.path.exists('GoalGAN/pretrained_models_PPO/'):
            os.mkdir('GoalGAN/pretrained_models_PPO/')
        path = 'GoalGAN/pretrained_models_PPO/{}.pt'.format(remark)
        print('Saving model to {}'.format(path))
        torch.save(self.policy.state_dict(), path)

    def load_model(self):
        print('Loading models with remark {}'.format(self.load_model_remark))
        policy_model = torch.load('GoalGAN/pretrained_models_PPO/{}.pt'.format(self.load_model_remark), map_location=lambda storage, loc: storage)
        self.policy.load_state_dict(policy_model)



class TD3Agent():
    def __init__(self, args, env_params):
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.g_dim = env_params['g_dim']
        self.action_boundary = env_params['action_boundary']
        self.max_episode_steps = env_params['max_episode_steps']

        self.evaluate_episodes = args.evaluate_episodes
        self.lr_pi = args.lr_pi_TD3
        self.lr_q = args.lr_q
        self.gamma = args.gamma
        self.tau = args.tau
        self.action_var = args.action_var
        self.noise_std = args.noise_std
        self.noise_clip = args.noise_clip
        self.K_updates = args.K_updates_TD3
        self.policy_update_interval = args.policy_update_interval
        self.batch_size = args.batch_size
        self.device = torch.device(args.device)
        self.load_model_remark = args.load_model_remark

        self.total_trained_goal_num = 0
        self.total_episode_num = 0
        self.total_update_num = 0
        self.policy_loss_log = 0.
        self.q1_loss_log = 0.
        self.q2_loss_log = 0.

        self.memory = MemoryBuffer(args.memory_capacity, self.o_dim, self.g_dim, self.a_dim)

        self.policy = GaussianPolicy(self.o_dim, self.g_dim, self.a_dim, self.action_var, self.device).to(self.device)
        self.policy_target = GaussianPolicy(self.o_dim, self.g_dim, self.a_dim, self.action_var, self.device).to(self.device)
        self.Q1 = QFunction(self.o_dim, self.g_dim, self.a_dim).to(self.device)
        self.Q1_target = QFunction(self.o_dim, self.g_dim, self.a_dim).to(self.device)
        self.Q2 = QFunction(self.o_dim, self.g_dim, self.a_dim).to(self.device)
        self.Q2_target = QFunction(self.o_dim, self.g_dim, self.a_dim).to(self.device)

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.lr_pi)
        self.optimizer_q1 = optim.Adam(self.Q1.parameters(), lr=self.lr_q)
        self.optimizer_q2 = optim.Adam(self.Q2.parameters(), lr=self.lr_q)

        self.hard_update()

    def hard_update(self):
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def soft_update(self):
        for param, param_target in zip(self.policy.parameters(), self.policy_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))
        for param, param_target in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))
        for param, param_target in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))

    def select_action(self, observation, goal):
        observation = torch.from_numpy(observation).float().to(self.device)
        goal = torch.from_numpy(goal).float().to(self.device)
        input_tensor = torch.cat([observation, goal], dim=0)
        with torch.no_grad():
            dist = self.policy(input_tensor)
            action = dist.sample()

            action = action.cpu().detach().numpy()
            action = np.clip(action, -self.action_boundary, self.action_boundary)

        return action

    def train_and_evaluate(self, goals, env, logger = None):
        returns = np.zeros(shape=[len(goals)], dtype=np.float32)
        
        for i_goal, goal in enumerate(goals):
            success_count = 0
            cumulative_r = 0. #for log
            used_steps = 0
            cumulative_loss_pi, cumulative_loss_q1, cumulative_loss_q2 = 0., 0., 0.
            print('--{} goal: ({:.4f}, {:.4f}):-----------------------'.format(self.total_trained_goal_num, goal[0], goal[1]))

            for i_episode in range(self.evaluate_episodes):
                success_flag = 0
                
                _ = env.reset()
                obs = env.set_goal(goal)
                for i_step in range(self.max_episode_steps):
                    a = self.select_action(obs['observation'], obs['desired_goal'])
                    obs_, reward, done, info = env.step(a)
                    
                    self.memory.store(obs['observation'], a, reward, obs_['observation'], obs['desired_goal'])
                    
                    cumulative_r += reward
                    used_steps += 1

                    if success_flag == 0 and info['is_success'] == 1:
                        success_flag = 1
                        break

                    obs = obs_

                if len(self.memory) > self.batch_size:
                    loss_q1, loss_q2, loss_pi = self.update() #need change
                    cumulative_loss_pi += loss_pi
                    cumulative_loss_q1 += loss_q1
                    cumulative_loss_q2 += loss_q2

                success_count += success_flag

            average_success = success_count / self.evaluate_episodes
            returns[i_goal] = average_success
            
            self.total_trained_goal_num += 1
            self.total_episode_num += self.evaluate_episodes
            if logger is not None:
                logger.add_scalar('Indicator/reward_per_step', cumulative_r/used_steps, self.total_trained_goal_num)
                logger.add_scalar('Indicator/goal_success_rate', average_success, self.total_trained_goal_num)
                logger.add_scalar('loss/loss_pi', cumulative_loss_pi/self.evaluate_episodes, self.total_update_num)
                logger.add_scalar('loss/loss_q1', cumulative_loss_q1/self.evaluate_episodes, self.total_update_num)
                logger.add_scalar('loss/loss_q2', cumulative_loss_q2/self.evaluate_episodes, self.total_update_num)
            
            print('\t success_rate: {:.2f}'.format(average_success))
            print('\t average_episode_return: {:.4f}'.format(cumulative_r/self.evaluate_episodes))

        return returns

    def evaluate_goal(self, goals, env):
        returns = np.zeros(shape=[len(goals)], dtype=np.float32)
        for i_goal, goal in enumerate(goals):
            success_count = 0
            for i_episode in range(self.evaluate_episodes):
                _ = env.reset()
                obs = env.set_goal(goal)
                for i_step in range(self.max_episode_steps):
                    a = self.select_action(obs['observation'], obs['desired_goal'])
                    obs_, reward, done, info = env.step(a)
                    if info['is_success'] == 1:
                        success_count += 1
                        break
                    obs = obs_
            average_success = success_count / self.evaluate_episodes
            print('{} goal: {} {}   return: {}'.format(i_goal, goal[0], goal[1], average_success))
            returns[i_goal] = average_success
        return returns

    def update(self):
        for i in range(self.K_updates):
            o, a, r, o_, g = self.memory.sample(self.batch_size)
            o = torch.from_numpy(o).to(self.device)
            a = torch.from_numpy(a).to(self.device)
            r = torch.from_numpy(r).to(self.device).unsqueeze(dim=1)
            o_ = torch.from_numpy(o_).to(self.device)
            g = torch.from_numpy(g).to(self.device)

            o_g_input = torch.cat([o, g], dim=1)
            next_o_g_input = torch.cat([o_, g], dim=1)
            o_g_a_input = torch.cat([o, g, a], dim=1)

            noise = (torch.randn_like(a) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
            a_target_next = self.policy_target(next_o_g_input).sample() + noise

            next_o_a_target_g_input = torch.cat([o_, g, a_target_next], dim=1)
            q1_next = self.Q1_target(next_o_a_target_g_input)
            q2_next = self.Q2_target(next_o_a_target_g_input)
            q_next_min = torch.min(q1_next, q2_next)
            q_loss_tar = r + self.gamma * q_next_min

            q1_loss_pred = self.Q1(o_g_a_input)
            q1_loss = F.mse_loss(q1_loss_pred, q_loss_tar.detach())
            self.optimizer_q1.zero_grad()
            q1_loss.backward()
            self.optimizer_q1.step()

            q2_loss_pred = self.Q2(o_g_a_input)
            q2_loss = F.mse_loss(q2_loss_pred, q_loss_tar.detach())
            self.optimizer_q2.zero_grad()
            q2_loss.backward()
            self.optimizer_q2.step()

            self.total_update_num += 1
            self.q1_loss_log = q1_loss.cpu().detach().numpy()
            self.q2_loss_log = q2_loss.cpu().detach().numpy()

            if self.total_update_num % self.policy_update_interval == 0:
                actions = self.policy(o_g_input).sample()
                policy_loss = - self.Q1(torch.cat([o_g_input, actions], dim=1)).mean()
                self.optimizer_pi.zero_grad()
                policy_loss.backward()
                self.optimizer_pi.step()

                self.policy_loss_log = policy_loss.cpu().detach().numpy()

                self.soft_update()

        return self.q1_loss_log, self.q2_loss_log, self.policy_loss_log

    def save_model(self, remark):
        if not os.path.exists('pretrained_models_TD3/'):
            os.mkdir('pretrained_models_TD3/')
        path = 'pretrained_models_TD3/{}.pt'.format(remark)
        print('Saving model to {}'.format(path))
        torch.save(self.policy.state_dict(), path)

    def load_model(self):
        print('Loading models with remark {}'.format(self.load_model_remark))
        policy_model = torch.load('pretrained_models_TD3/{}.pt'.format(self.load_model_remark), map_location=lambda storage, loc: storage)
        self.policy.load_state_dict(policy_model)