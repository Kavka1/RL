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


class CURL_SACAgent(object):
    def __init__(self, config: Dict, env_params: Dict) -> None:
        self.obs_shape = env_params['obs_shape']
        self.a_dim = env_params['a_dim']

        self.gamma = config['gamma']
        self.tau = config['tau']
        self.lr = config['lr']
        self.actor_train_delay = config['actor_train_delay']
        self.logstd_min, self.logstd_max = config['logstd_min'], config['logstd_max']
        self.batch_size = config['batch_size']
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