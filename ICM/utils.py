import numpy as np


def get_env_params(env):
    params = {
        'o_dim': env.observation_space['observation'].shape[0],
        'a_dim': env.action_space.shape[0],
        'g_dim': env.observation_space['desired_goal'].shape[0],
        'action_max': env.action_space.high[0],
        'max_timestep': env.spec.max_episode_steps
    }
    return params

def get_state(obs):
    state = np.concatenate([obs['observation'], obs['desired_goal']], axis=-1)
    return state

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
