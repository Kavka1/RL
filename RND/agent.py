import torch
import torch.nn.functional as F
import os
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from model import CNNActorCritic, RNDNetwork
from buffer import Buffer
from utils import Normalizer, global_grad_norm_


class PPOAgent():
    def __init__(self, args, env_params):
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.r_dim = args.r_dim
        
        self.lr = args.lr
        self.gamma_e = args.gamma_e
        self.gamma_i = args.gamma_i
        self.lamda = args.lamda
        self.entropy_coef = args.entropy_coef
        self.ex_coef = args.ex_coef
        self.in_coef = args.in_coef
        self.clip_eps = args.clip_eps
        self.update_epoch = args.update_epoch
        self.batch_size = args.batch_size
        self.initialize_episode = args.initialize_episode
        self.update_proportion = args.update_proportion
        self.rollout_len = args.rollout_len
        self.obs_clip = args.obs_clip

        self.device = torch.device(args.device)

        self.actor_critic = CNNActorCritic(in_channel=self.o_dim[0], a_dim=self.a_dim).to(self.device)
        self.RND = RNDNetwork(in_channel=1).to(self.device)

        self.optimizer = optim.Adam(list(self.actor_critic.parameters()) + list(self.RND.predictor.parameters()), lr=self.lr)

        self.buffer = Buffer(capacity=self.rollout_len, o_dim=self.o_dim)

        self.normalizer_obs = Normalizer(shape=self.o_dim, clip=self.obs_clip)
        self.normalizer_ri = Normalizer(shape=1, clip=np.inf)

    def choose_action(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        with torch.no_grad():
            action_logits = self.actor_critic.act(obs)
        
        dist = Categorical(action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action, log_prob = action.cpu().detach().numpy(), log_prob.cpu().detach().numpy()
        return action, log_prob

    def compute_intrinsic_reward(self, obs_):
        obs_ = self.normalizer_obs.normalize(obs_)
        obs_ = torch.from_numpy(obs_[:, 3:, :, :]).float().to(self.device)
        with torch.no_grad():
            pred_feature, tar_feature = self.RND(obs_)
        reward_in = F.mse_loss(pred_feature, tar_feature, reduction='none').mean(dim=-1)
        reward_in = reward_in.cpu().detach().numpy()
        return reward_in

    def GAE_caculate(self, rewards, masks, values, gamma, lamda):
        returns = np.zeros(shape=len(rewards), dtype=np.float32)
        deltas = np.zeros(shape=len(rewards), dtype=np.float32)
        advantages = np.zeros(shape=len(rewards), dtype=np.float32)

        pre_return = 0.
        pre_advantage = 0.
        pre_value = 0.
        for i in reversed(range(len(rewards))):
            returns[i] = rewards[i] + masks[i] * gamma * pre_return
            deltas[i] = rewards[i] + masks[i] * gamma * pre_value - values[i]
            advantages[i] = deltas[i] + gamma * lamda * pre_advantage

            pre_return = returns[i]
            pre_value = values[i]
            pre_advantage = advantages[i]

        return returns, deltas, advantages

    def update(self, o, a, r_i, r_e, mask, o_, log_prob):
        self.normalizer_obs.update(o_.reshape(-1, 4, 84, 84).copy())
        self.normalizer_ri.update(r_i.reshape(-1).copy())

        r_i = self.normalizer_ri.normalize(r_i)
        o_ = self.normalizer_obs.normalize(o_)
        o = torch.from_numpy(o).to(self.device).float() / 255.
        
        returns_ex = np.zeros_like(r_e)
        returns_in = np.zeros_like(r_e)
        advantage_ex = np.zeros_like(r_e)
        advantage_in = np.zeros_like(r_e)
        for i in range(r_e.shape[0]):
            action_logits, value_ex, value_in = self.actor_critic(o[i])
            value_ex, value_in = value_ex.cpu().detach().numpy(), value_in.cpu().detach().numpy()
            returns_ex[i], _, advantage_ex[i] = self.GAE_caculate(r_e[i], mask[i], value_ex, self.gamma_e, self.lamda) #episodic
            returns_in[i], _, advantage_in[i] = self.GAE_caculate(r_i[i], np.ones_like(mask[i]), value_in, self.gamma_i, self.lamda) #non_episodic

        o = o.reshape((-1, 4, 84, 84))
        a = np.reshape(a, -1)
        o_ = np.reshape(o_[:, :, 3, :, :], (-1, 1, 84, 84))
        log_prob = np.reshape(log_prob, -1)
        returns_ex = np.reshape(returns_ex, -1)
        returns_in = np.reshape(returns_in, -1)
        advantage_ex = np.reshape(advantage_ex, -1)
        advantage_in = np.reshape(advantage_in, -1)        

        a = torch.from_numpy(a).float().to(self.device)
        o_ = torch.from_numpy(o_).float().to(self.device).float()
        log_prob = torch.from_numpy(log_prob).float().to(self.device)
        returns_ex = torch.from_numpy(returns_ex).float().to(self.device).unsqueeze(dim=1)
        returns_in = torch.from_numpy(returns_in).float().to(self.device).unsqueeze(dim=1)
        advantage_ex = torch.from_numpy(advantage_ex).float().to(self.device)
        advantage_in = torch.from_numpy(advantage_in).float().to(self.device)

        sample_range = list(range(len(o)))

        for i_update in range(self.update_epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(o) / self.batch_size)):
                idx = sample_range[self.batch_size*j : self.batch_size*(j+1)]
                #update RND
                pred_RND, tar_RND = self.RND(o_[idx])
                loss_RND = F.mse_loss(pred_RND, tar_RND.detach(), reduction='none').mean(-1)
                mask = torch.randn(len(loss_RND)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                loss_RND = (loss_RND * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))

                #update actor-critic
                action_logits, value_ex, value_in = self.actor_critic(o[idx])
                advantage = self.ex_coef * advantage_ex[idx] + self.in_coef * advantage_in[idx]
                dist = Categorical(action_logits)
                new_log_prob = dist.log_prob(a[idx])
                
                ratio = torch.exp(new_log_prob - log_prob[idx])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
                loss_actor = torch.min(surr1, surr2).mean() - self.entropy_coef * dist.entropy().mean()
                loss_critic = F.mse_loss(value_ex, returns_ex[idx]) + F.mse_loss(value_in, returns_in[idx])

                loss_ac = loss_actor + 0.5 * loss_critic
            
                loss = loss_RND + loss_ac
                self.optimizer.zero_grad()
                loss.backward()
                global_grad_norm_(list(self.actor_critic.parameters())+list(self.RND.predictor.parameters()))
                self.optimizer.step()
        
        return loss_RND.cpu().detach().numpy(), loss_actor.cpu().detach().numpy(), loss_critic.cpu().detach().numpy()

    def save_model(self, remark):
        if not os.path.exists('pretrained_models_PPO/'):
            os.mkdir('pretrained_models_PPO/')
        path = 'pretrained_models_PPO/{}.pt'.format(remark)
        print('Saving model to {}'.format(path))
        torch.save(self.actor_critic.state_dict(), path)

    def load_model(self, load_model_remark):
        print('Loading models with remark {}'.format(load_model_remark))
        model = torch.load('pretrained_models_PPO/{}.pt'.format(load_model_remark), map_location=lambda storage, loc: storage)
        self.actor_critic.load_state_dict(model)