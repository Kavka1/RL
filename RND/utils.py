import numpy as np
import torch
from torch._six import inf


def get_env_params(env, args):
    env_params = dict(
        o_dim = [4, 84, 84],
        a_dim = env.action_space.n
    )
    return env_params


class Normalizer():
    def __init__(self, shape, clip, epsilon=1e-4):
        self.mean = np.zeros(shape=shape, dtype=np.float32)
        self.var = np.ones(shape=shape, dtype=np.float32)
        self.count = 0
        self.clip = clip
        self.epsilon = epsilon

    def update(self, batch_data):
        batch_mean = batch_data.mean(axis=0)
        batch_var = batch_data.std(axis=0)
        batch_count = batch_data.shape[0]

        total_count = self.count + batch_count

        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (total_count + self.epsilon)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.count = total_count
        self.mean = new_mean
        self.var = new_var

    def normalize(self, data):
        data = (data - self.mean) / (np.sqrt(self.var) + self.epsilon)
        data = np.clip(data, -self.clip, self.clip)
        return data

def global_grad_norm_(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    
    return total_norm
