import numpy as np

class Memory(object):
    def __init__(self, memory_size, o_dim, a_dim):
        super(Memory, self).__init__()
        self.obs_buffer = np.zeros([memory_size, o_dim], dtype=np.float32)
        self.a_buffer = np.zeros([memory_size, a_dim], dtype=np.float32)
        self.r_buffer = np.zeros([memory_size, 1], dtype=np.float32)
        self.next_obs_buffer = np.zeros([memory_size, o_dim], dtype=np.float32)
        self.done_buffer = np.zeros([memory_size, 1], dtype=np.float32)

        self.memory_size = memory_size
        self.o_dim, self.a_dim = o_dim, a_dim
        self.write = 0
        self.num_sample = 0

    def __len__(self):
        return self.num_sample

    def store(self, o, a, r, o_, done):
        idx = self.write

        self.obs_buffer[idx] = o
        self.a_buffer[idx] = a
        self.r_buffer[idx] = r
        self.next_obs_buffer[idx] = o_
        self.done_buffer[idx] = done

        self.write = (self.write + 1) % self.memory_size
        self.num_sample = min(self.memory_size, self.num_sample+1)

    def sample_batch(self, batch_size):
        idx = np.random.randint(low=0, high=self.num_sample, size=batch_size)
        transitions = self.obs_buffer[idx], self.a_buffer[idx], self.r_buffer[idx], self.next_obs_buffer[idx], self.done_buffer[idx]
        return transitions