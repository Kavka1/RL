import numpy as np


class Memory:
    def __init__(self, size, s_dim, a_dim):
        self.size = size
        self.state = np.zeros(shape=[size, s_dim], dtype=np.float32)
        self.actions = np.zeros(shape=[size, a_dim], dtype=np.float32)
        self.rewards = np.zeros(shape=size, dtype=np.float32)
        self.next_state = np.zeros(shape=[size, s_dim], dtype=np.float32)

        self.write = 0
        self.num_samples = 0

    def __len__(self):
        return self.num_samples

    def clear(self):
        del self.state
        del self.actions
        del self.rewards
        del self.next_state
        self.num_samples = 0
        self.write = 0

    def store(self, s, a, r, s_):
        idx = self.write
        self.state[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_state[idx] = s_

        self.write = (self.write + 1) % self.size
        self.num_samples = min(self.num_samples + 1, self.size)

    def sample_batch(self, batch_size):
        if batch_size > self.num_samples:
            raise ValueError('No enough samples for one batch')
        idxs = np.random.randint(low=0, high=self.num_samples, size=batch_size)
        transitions = self.state[idxs], self.actions[idxs], self.rewards[idxs], self.next_state[idxs]
        return transitions