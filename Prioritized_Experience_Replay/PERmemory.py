import math
import numpy as np
import random

#a binary data structure where the parent's data is the sum of children's data
class SumTree(object):
    def __init__(self, size):
        self.size = size
        self.tree = np.zeros(2*self.size - 1)
        self.data = np.zeros(self.size, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    #find sample on leaf node
    def _retrive(self, idx, s):
        left = 2*idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrive(left, s)
        else:
            return self._retrive(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.size - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.size:
            self.write = 0

        if self.n_entries < self.size:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrive(0, s)
        dataIdx =  idx - self.size + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PERmemory(object):
    def __init__(self, memory_size, alpha, beta, beta_increment, basic_e=0.01):
        self.size = memory_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.basic_e = basic_e
        self.tree = SumTree(self.size)

    def _get_priority(self , tderror):
        return (np.abs(tderror) + self.basic_e)**self.alpha

    def add(self, sample, tderror):
        p = self._get_priority(tderror)
        self.tree.add(p, sample)

    def update(self, idx, tderror):
        p = self._get_priority(tderror)
        self.tree.update(idx, p)

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        self.beta = min(1., self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries*sampling_probilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight


if __name__ == '__main__':
    memory = PERmemory(1000, 0.5, 0.5, 0.001)