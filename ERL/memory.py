import torch
import numpy as np


class Memory():
    def __init__(self, memory_size, o_dim, a_dim):
        self.o = np.zeros([memory_size, o_dim], dtype=np.float32)
        self.a = np.zeros([memory_size, a_dim], dtype=np.float32)
        self.r = np.zeros([memory_size, 1], dtype=np.float32)
        self.o_ = np.zeros([memory_size, o_dim], dtype=np.float32)
        self.done = np.zeros([memory_size, 1], dtype=np.float32)

        self.memory_size = memory_size
        self.write = 0
        self.num_sample = 0

    def store(self, o, a, r, o_, done):
        idx = self.write
        self.o[idx] = o
        self.a[idx] = a
        self.r[idx] = r
        self.o_[idx] = o_
        self.done[idx] = done

        self.write = (self.write + 1) % self.memory_size
        self.num_sample = min(self.num_sample + 1, self.memory_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.num_sample, batch_size)
        transitions = self.o[idxs], self.a[idxs], self.r[idxs], self.o_[idxs], self.done[idxs]
        return transitions