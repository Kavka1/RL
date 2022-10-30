import math
from typing import List, Dict, Tuple
from torch import Tensor

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, Bernoulli, TransformedDistribution, TanhTransform, Categorical



class ConvEncoder(nn.Module):
    def __init__(
        self,
        depth:      int         = 32,
        activation: nn.Module   = nn.ReLU()
    ) -> None:
        super().__init__()

        self.depth = depth
        self.conv1 = nn.Conv2d(3, 1 * depth, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(1 * depth, 2 * depth, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(2 * depth, 4 * depth, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(4 * depth, 8 * depth, kernel_size=4, stride=2)
        # 64 -> 31 -> 14 -> 6 -> 2
        self.act   = activation()

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = obs['image']

        T, B, *Other = x.size()
        x = x.view(T * B, *Other)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = x.flatten(start_dim=-3)

        TB, *Other = x.size()
        x = x.view(T, B, *Other)

        assert x.size(-1) == 32 * self.depth
        return x


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feature_dim:    int,
        depth:          int                     =   32,
        activation:     nn.Module               =   nn.ReLU,
        shape:          Tuple[int, int, int]    =   (3, 64, 64)
    ) -> None:
        super().__init__()
        self.depth = depth
        self.shape = shape

        self.fc    = nn.Linear(feature_dim, 32 * depth)
        self.conv1 = nn.ConvTranspose2d(32 * depth, 4 * depth, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(4 * depth, 2 * depth, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(2 * depth, 1 * depth, kernel_size=6, stride=2)
        self.conv4 = nn.ConvTranspose2d(1 * depth, 3, kernel_size=6, stride=2)
        self.act   = activation()

    def forward(self, feature: Tensor) -> Tensor:
        x = self.fc(feature)

        T, B, *Other = x.size()
        x = x.view(T * B, *Other)

        x = x[:, :, None, None]     #   [T*B, C, 1, 1]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        mean = self.conv4(x)
        assert mean.size()[-3:] == self.shape

        TB, *Other = mean.size()
        mean = mean.view(T, B, *Other)

        return Independent(Normal(mean, 1.0), reinterpreted_batch_ndims=len(self.shape))


class DenseDecoder(nn.Module):
    def __init__(
        self,
        input_dim:  int,
        shape:      Tuple[int, ...],
        layers:     int,
        units:      int,
        dist:       str         = 'normal',
        activation: nn.Module   = nn.ELU
    ) -> None:
        super().__init__()

        self.shape = shape
        self.layers= layers
        self.units = units
        self.dist  = dist

        self.act        = activation()
        self.fc_layers  = nn.ModuleList()
        for _ in range(layers):
            self.fc_layers.append(nn.Linear(input_dim, units))
            input_dim = units
        self.fc_output  = nn.Linear(input_dim, int(np.prod(self.shape)))

    def forward(self, features: Tensor) -> Tensor:
        x = features
        for layer in self.fc_layers:
            x = self.act(layer(x))
        x = self.fc_output(x)
        x = x.reshape(*x.size()[:-1], *self.shape)

        if self.dist == 'normal':
            return Independent(Normal(x, 1.0), reinterpreted_batch_ndims=len(self.shape))
        elif self.dist == 'binary':
            return Independent(Bernoulli(logits=x), reinterpreted_batch_ndims=len(self.shape))
        else:
            raise ValueError()


class ActionDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        a_dim:     int,
        layers:    int,
        units:     int,
        dist:      str = 'tanh_normal',
        activation:nn.Module = nn.ELU,
        min_std:   float = 1e-4,
        init_std:  float = 5.,
        mean_scale:float = 5.
    ) -> None:
        super().__init__()

        self.input_dim  = input_dim
        self.a_dim      = a_dim
        self.units      = units
        self.layers     = layers
        self.dist       = dist
        self.min_std    = min_std
        self.init_std   = init_std
        self.mean_scale = mean_scale

        self.act        = activation()
        self.fc_layers  = nn.ModuleList()
        for _ in range(layers):
            self.fc_layers.append(nn.Linear(input_dim, units))
            input_dim = units
        self.fc_output  = nn.Linear(
            units,
            a_dim * 2 if dist == 'tanh_normal' else a_dim
        )
        self.raw_init_std = math.log(math.exp(init_std - 1))

    def forward(self, features: Tensor) -> Tensor:
        x = features
        for layer in self.fc_layers:
            x = self.act(layer(x))
        x = self.fc_output(x)

        if self.dist == 'tanh_normal':
            mean, std = x.chunk(x, dim=-1)
            mean      = self.mean_scale * torch.tanh(mean / self.mean_scale)
            std       = nn.functional.softplus(std + self.raw_init_std) + self.min_std
            dist      = Normal(mean, std)
            dist      = TransformedDistribution(dist, TanhTransform())
            dist      = Independent(dist, 1)
        elif self.dist == 'onehot':
            dist      = Categorical(logits=x)
        else:
            raise ValueError()

        return dist