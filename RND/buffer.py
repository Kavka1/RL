import numpy as np
import torch


class Buffer():
    def __init__(self, capacity, o_dim):
        self.obs = np.zeros(shape=[capacity, o_dim[0], o_dim[1], o_dim[2]], dtype=np.float32)
        self.a = np.zeros(shape=[capacity, 1], dtype=np.float32)
        self.r_i = np.zeros(shape=[capacity, 1], dtype=np.float32)
        self.r_e = np.zeros(shape=[capacity, 1], dtype=np.float32)
        self.mask = np.zeros(shape=[capacity, 1], dtype=np.float32)
        self.obs_ = np.zeros(shape=[capacity, o_dim[0], o_dim[1], o_dim[2]], dtype=np.float32)
        self.log_prob = np.zeros(shape=[capacity, 1], dtype=np.float32)

        self.write = 0
        self.sample_num = 0
        self.capacity = capacity

    def store(self, o, a, r_i, r_e, mask, o_, log_prob):
        idx = self.write
        self.obs[idx] = o
        self.a[idx] = a
        self.r_i[idx] = r_i
        self.r_e[idx] = r_e
        self.mask[idx] = mask
        self.obs_[idx] = o_
        self.log_prob[idx] = log_prob

        self.write = (self.write + 1) % self.capacity
        self.sample_num = min(self.sample_num + 1, self.capacity)

    def __len__(self):
        return self.sample_num

    def sample(self):
        obs = self.obs
        a = self.a
        r_i = self.r_i
        r_e = self.r_e
        mask = self.mask
        obs_ = self.obs_
        log_prob = self.log_prob
        
        return obs, a, r_i, r_e, mask, obs_, log_prob
