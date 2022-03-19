from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from RL.CURL.actor import Actor
from RL.CURL.critic import Critic
from RL.CURL.encoder import PixelEncoder
from RL.CURL.contrastive import CURL_model
from RL.CURL.utils import center_crop_image, soft_update


class CURL_SACAgent(object):
    def __init__(self, config: Dict, env_params: Dict) -> None:
        self.obs_shape = env_params['obs_shape']
        self.a_dim = env_params['a_dim']

        self.gamma = config['gamma']
        self.lr = config['lr']
        self.encoder_tau = config['encoder_tau']
        self.critic_tau = config['critic_tau']
        self.actor_update_freq = config['actor_update_freq']
        self.cpc_update_freq = config['cpc_update_freq']
        self.target_update_freq = config['target_update_freq']
        self.logstd_min, self.logstd_max = config['logstd_min'], config['logstd_max']
        self.batch_size = config['batch_size']
        self.critic_update_detach_encoder = config['critic_update_detach_encoder']
        self.device = torch.device(config['device'])

        self.encoder = PixelEncoder(
            env_params['obs_shape'], 
            config['encoder_feature_dim'], 
            config['encoder_num_layers'],
            config['encoder_num_filters'],
            output_logits= True
        ).to(self.device)
        self.encoder_trg = PixelEncoder(
            env_params['obs_shape'],
            config['encoder_feature_dim'],
            config['encoder_num_layers'],
            config['encoder_num_filters'],
            output_logits= True
        ).to(self.device)
        self.encoder_trg.load_state_dict(self.encoder)

        self.critic = Critic(
            self.encoder,
            self.a_dim,
            config['critic_hidden_size']
        ).to(self.device)
        self.critic_trg = Critic(
            self.encoder_trg,
            self.a_dim,
            config['critic_hidden_size']
        ).to(self.device)
        self.critic_trg.load_state_dict(self.critic)

        self.actor = Actor(
            self.encoder, 
            self.a_dim, 
            config['actor_hidden_size'],
            self.logstd_max,
            self.logstd_min
        ).to(self.device)

        self.log_alpha = torch.tensor(np.log(0.01), requires_grad= True).to(self.device)
        self.target_entropy = - torch.tensor(self.a_dim).float()

        self.CURL = CURL_model(
            config['encoder_feature_dim'],
            self.encoder,
            self.encoder_trg
        ).to(self.device)

        # Optimizers Initialization
        self.optimizer_actor = optim.Adam(self.actor.parameters(), self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), self.lr)
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), self.lr)
        self.optimizer_cpc = optim.Adam(self.CURL.parameters(), self.lr)
        self.optimizer_logalpha = optim.Adam([self.log_alpha], self.lr)

        self.update_count = 0
        self.log_loss = {
            'loss_critic': 0,
            'loss_actor': 0,
            'loss_alpha': 0,
            'loss_cpc': 0
        }

    def selection_action(self, obs: np.array, output_mu: bool = False) -> np.array:
        if obs.shape[-1] != self.obs_shape[-1]:
            obs = center_crop_image(obs, self.obs_shape[-1])

        obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.unsqueeze(0)  # [1, c, h, w]
        with torch.no_grad():
            if output_mu:
                mu, pi, log_prob, log_std = self.actor(obs, 
                detach_encoder = False, compute_pi = False, compute_log_prob = False)
                return mu.detach().cpu().numpy().flatten()  # [1, |a|] -> [|a|]
            else:
                mu, pi, log_prob, log_std = self.actor(obs,
                detach_encoder = False, compute_pi = True, compute_log_prob = False)
                return pi.detach().cpu().numpy().flatten()  # [1, |a|] -> [|a|]

    def update_critic(self, obs: torch.tensor, a: torch.tensor, r: torch.tensor, next_obs: torch.tensor, notdone: torch.tensor) -> None:
        with torch.no_grad():
            _, next_pi_trg, next_pi_logprob, _ = self.actor(next_obs,
            detach_encoder = False, compute_pi = True, compute_log_prob = True)
            trg_q1, trg_q2 = self.critic_trg(next_obs, next_pi_trg, self.critic_update_detach_encoder)
            q_trg = torch.min(trg_q1, trg_q2) - self.log_alpha.exp().detach() * next_pi_logprob
            q_trg = r + notdone * self.gamma * q_trg
        q1_pred, q2_pred = self.critic(obs, a, self.critic_update_detach_encoder)
        loss_critic = F.mse_loss(q1_pred, q_trg) + F.mse_loss(q2_pred, q_trg)

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        self.log_loss['loss_citic'] = loss_critic.detach().cpu().item()

    def update_actor_alpha(self, obs: torch.tensor) -> None:
        mu, pi, log_prob, log_std = self.actor(obs,
        detach_encoder = True, compute_pi = True, compute_log_prob = True)

        q1_value, q2_value = self.critic(obs, pi, detach_encoder=True)
        q_value = torch.min(q1_value, q2_value)

        loss_actor = (self.log_alpha.exp().detach() * log_prob - q_value.mean()).mean()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        loss_alpha = self.log_alpha.exp() * (- log_prob.detach() - self.target_entropy).mean()
        self.optimizer_logalpha.zero_grad()
        loss_alpha.backward()
        self.optimizer_logalpha.step()

        self.log_loss['loss_actor'] = loss_actor.cpu().detach().item()
        self.log_loss['loss_alpha'] = loss_alpha.cpu().detach().item()

    def update_cpc(self, obs_anchor: torch.tensor, obs_pos: torch.tensor) -> float:
        z_a = self.encoder(obs_anchor)
        z_pos = self.encoder_trg(obs_pos)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = F.cross_entropy(logits, labels)

        self.optimizer_encoder.zero_grad()
        self.optimizer_cpc.zero_grad()
        loss.backward()
        self.optimizer_cpc.step()
        self.optimizer_encoder.step()

        self.log_loss['loss_cpc'] = loss.cpu().detach().item()

    def update(self, buffer) -> Dict:
        obs, action, reward, next_obs, not_done, cpc_kwargs = buffer.sample_cpc()

        self.update_critic(obs, action, reward, next_obs, not_done)

        if self.update_count % self.actor_update_freq == 0:
            self.update_actor_alpha(obs)
        if self.update_count % self.cpc_update_freq == 0:
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_cpc(obs_anchor, obs_pos)

        if self.update_count % self.target_update_freq == 0:
            soft_update(self.critic_trg.Q1, self.critic.Q1, self.critic_tau)
            soft_update(self.critic_trg.Q2, self.critic.Q2, self.critic_tau)
            soft_update(self.encoder_trg, self.encoder, self.encoder_tau)

        self.update_count += 1
        return self.log_loss