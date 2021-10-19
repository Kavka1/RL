import torch
import torch.nn as nn
import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNNActorCritic(nn.Module):
    def __init__(self, in_channel, a_dim):
        super(CNNActorCritic, self).__init__()
        self.in_channel = in_channel
        self.a_dim = a_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(7 * 7 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, a_dim),
            nn.Softmax(dim=-1)
        )

        self.extra_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.critic_ex = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_in = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        feature = self.encoder(obs)

        action_logits = self.actor(feature)

        pre_value = self.extra_layer(feature)
        value_ex = self.critic_ex(pre_value)
        value_in = self.critic_in(pre_value)

        return action_logits, value_ex, value_in

    def act(self, obs):
        feature = self.encoder(obs)
        action_logits = self.actor(feature)
        return action_logits


class RNDNetwork(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 128):
        super(RNDNetwork, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_channel)
        )

        self.target = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_channel)
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):
        pred_feature = self.predictor(obs)
        tar_feature = self.target(obs)
        return pred_feature, tar_feature