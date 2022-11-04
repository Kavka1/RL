from typing import List, Optional
import random
import numpy as np
from gym import spaces


class ReplayBuffer:
    def __init__(self, action_space: spaces.Space, balance: bool = True):
        self.current_episode: Optional[list] = []
        self.action_space = action_space
        self.balance = balance
        self.episodes = []

    def start_episode(self, obs: dict):

        transition = obs.copy()
        transition['action'] = np.zeros(self.action_space.shape)
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        self.current_episode = [transition]

    def add(self, obs: dict, action: np.ndarray, reward: float, done: bool,
            info: dict):
        transition = obs.copy()
        transition['action'] = action
        transition['reward'] = reward
        transition['discount'] = info.get('discount',
                                          np.array(1 - float(done)))
        self.current_episode.append(transition)
        if done:
            episode = {
                k: [t[k] for t in self.current_episode]
                for k in self.current_episode[0]
            }
            episode = {k: self.convert(v) for k, v in episode.items()}
            self.episodes.append(episode)
            self.current_episode = []

    def sample_single_episode(self, length: int):
        episode = random.choice(self.episodes)
        total = len(next(iter(episode.values())))
        available = total - length
        while True:
            if available < 1:
                print(f'Skipped short episode of length {available}.')
            if self.balance:
                index = min(random.randint(0, total), available)
            else:
                index = int(random.randint(0, available))
            episode = {k: v[index:index + length] for k, v in episode.items()}
            return episode

    def sample(self, batch_size: int, length: int):
        """
        Args:
            length: number of observations, or transition + 1
        """
        episodes = [self.sample_single_episode(length) for _ in range(batch_size)]
        batch = {}
        for key in episodes[0]:
            batch[key] = np.array([ep[key] for ep in episodes])
        return batch

    def convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = np.float32
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = np.int32
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)