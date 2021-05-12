import gym
import numpy as np
import torch
from mpi4py import MPI


# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync
    """
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params')
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode='params')


def sync_grads(network):
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_params_or_grads(network, global_grads, mode='grads')


# get the flat grads or params
def _get_flat_params_or_grads(network, mode='params'):
    """
    include two kinds: grads and params
    """
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    """
    include two kinds: grads and params
    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()


def get_env_params(env):
    params = {
        'o_dim': env.observation_space['observation'].shape[0],
        'a_dim': env.action_space.shape[0],
        'g_dim': env.observation_space['desired_goal'].shape[0],
        'action_max': env.action_space.high[0],
        'max_episode_steps': env.spec.max_episode_steps
    }
    return params


class Normalizer:
    def __init__(self, size, eps=1e-2, clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range

        self.sum = np.zeros(size, np.float32)
        self.sumsq = np.zeros(size, np.float32)
        self.count = np.zeros(1, np.float32)

        self.mean = np.zeros(size, np.float32)
        self.std = np.ones(size, np.float32)

    def update(self, v):
        v = v.reshape(-1, self.size)
        self.sum += v.sum(axis=0)
        self.sumsq += np.square(v).sum(axis=0)
        self.count += v.shape[0]

        self.mean = self.sum / self.count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.sumsq/self.count)-np.square(self.sum/self.count)))

    def normalize(self, v):
        return np.clip((v - self.mean)/self.std, -self.clip_range, self.clip_range)

