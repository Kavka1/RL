from copy import deepcopy
from typing import Dict, List, Tuple
from collections import deque
import numpy as np
import torch
import gym
import yaml
import random
import torch.nn as nn
import torch.nn.functional as F
from skimage.util.shape import view_as_windows
import os


def confirm_path(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_env_params(env: gym.Env) -> Dict:
    return {
        'preaug_obs_shape': list(env.observation_space.shape),
        'a_dim': env.action_space.shape[0],
        'action_bound': float(env.action_space.high[0])
    }


def make_exp_path(config: Dict) -> Dict:
    exp_path = config['result_path'] + f"{config['domain_name']}-{config['task_name']}_{config['seed']}"
    while os.path.exists(exp_path):
        exp_path = exp_path + '_*'
    exp_path += '/'
    os.makedirs(exp_path)
    config.update({
        'exp_path': exp_path
    })
    return config


def save_config_and_env_params(config: Dict, env_params: Dict) -> None:
    config.update({
        'env_params': env_params
    })
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, indent=2)


def tie_weights(trg: nn.Module, src: nn.Module) -> None:
    assert type(trg) == type(src)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def soft_update(trg_net: nn.Module, src_net: nn.Module, tau: float) -> None:
    for trg_param, src_param in zip(trg_net.parameters(), src_net.parameters()):
        trg_param.data.copy_(tau * src_param + (1 - tau) * trg_param)


def reparameterize(mu: torch.tensor, std: torch.tensor, return_noise: bool = True) -> Tuple[torch.tensor, torch.tensor]:
    noise = torch.rand_like(std)
    output = mu + std * noise
    if return_noise:
        return output, noise
    else:
        return output


def compute_gaussian_logprob(noise: torch.tensor, log_std: torch.tensor) -> torch.tensor:
    # noise denotes to the x = mu + "noise" * std
    return (- 0.5 * noise ** 2 - log_std).sum(-1, keepdim = True) - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu: torch.tensor, pi: torch.tensor, log_prob: torch.tensor) -> Tuple:
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_prob is not None:
        log_prob -= torch.log(1 - pi.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
    return mu, pi, log_prob


class FrameStack(gym.Wrapper):
    def __init__(self, env, k) -> None:
        gym.Wrapper.__init__(self, env)
        self._env = env
        self.k = k

        self._frames = deque([], self.k)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 1,
            shape = ((obs_shape[0] * self.k,) + obs_shape[1:]),
            dtype = env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def _get_obs(self) -> np.array:
        assert len(self._frames) == self.k
        return np.concatenate(list(self._frames), 0)

    def reset(self) -> np.array:
        obs = self._env.reset()
        for i in range(self.k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, a: np.array) -> Tuple:
        obs, r, done, info = self._env.step(a)
        self._frames.append(obs)
        return self._get_obs(), r, done, info


def center_crop_image(img: np.array, crop_size: int) -> np.array:
    h, w = img.shape[1:]
    new_h, new_w = crop_size, crop_size

    top_crop = (h - new_h) // 2
    left_crop = (w - new_w) // 2

    img = img[:, top_crop: top_crop + new_h, left_crop: left_crop + new_w]
    return img


def random_crop(imgs: np.array, output_size: int) -> np.array:
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size

    imgs = np.transpose(imgs, (0, 2, 3, 1))
    
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs