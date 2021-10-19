import torch
import numpy as np


class TrajectoryBuffer():
    def __init__(self, capacity, o_dim, g_dim, a_dim):
        self.o = np.zeros(shape=[capacity, o_dim], dtype=np.float32)
        self.a = np.zeros(shape=[capacity, a_dim], dtype=np.float32)
        self.r = np.zeros(shape=[capacity], dtype=np.float32)
        self.o_ = np.zeros(shape=[capacity, o_dim], dtype=np.float32)
        self.g = np.zeros(shape=[capacity, g_dim], dtype=np.float32)
        self.log_prob = np.zeros(shape=[capacity], dtype=np.float32)

        self.write = 0
        self.size = 0
        self.capacity = capacity

    def store(self, o, a, r, o_, g, lp):
        idx = self.write
        self.o[idx] = o
        self.a[idx] = a
        self.r[idx] = r
        self.o_[idx] = o_
        self.g[idx] = g
        self.log_prob[idx] = lp
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def get_trajectory(self):
        return self.o[:self.size], self.a[:self.size], self.r[:self.size], self.o_[:self.size], self.g[:self.size], self.log_prob[:self.size]

    def clear(self):
        self.size = 0
        self.write = 0
    

class MemoryBuffer():
    def __init__(self, capacity, o_dim, g_dim, a_dim):
        self.o_dim = o_dim
        self.g_dim = g_dim
        self.a_dim = a_dim
        self.o = np.zeros(shape=[capacity, o_dim], dtype=np.float32)
        self.a = np.zeros(shape=[capacity, a_dim], dtype=np.float32)
        self.r = np.zeros(shape=[capacity], dtype=np.float32)
        self.o_ = np.zeros(shape=[capacity, o_dim], dtype=np.float32)
        self.g = np.zeros(shape=[capacity, g_dim], dtype=np.float32)

        self.capacity = capacity
        self.write = 0
        self.num = 0

    def store(self, o, a, r, o_, g):
        idx = self.write
        self.o[idx] = o
        self.a[idx] = a
        self.r[idx] = r
        self.o_[idx] = o_
        self.g[idx] = g

        self.write = (self.write + 1) % self.capacity
        self.num = min(self.num + 1, self.capacity)

    def sample(self, batch_size):
        assert batch_size <= self.num
        idxs = np.random.randint(low = 0, high=self.num, size = batch_size)
        return self.o[idxs], self.a[idxs], self.r[idxs], self.o_[idxs], self.g[idxs]

    def __len__(self):
        return self.num


class GoalBuffer():
    def __init__(self, g_dim, dis_threshold=0.2):
        self.old_goal_list = []
        self.g_dim = g_dim

        self.size = 0
        self.mean_goal = np.zeros(shape=[g_dim], dtype=np.float32)
        self.goal_sum = np.zeros(shape=[g_dim], dtype=np.float32)

        self.dis_threshold = dis_threshold

    def update(self, new_goals):
        for goal in new_goals:
            dis = np.linalg.norm(goal - self.mean_goal, axis=-1)
            
            if dis <= self.dis_threshold:
                continue
            else:
                self.old_goal_list.append(goal)
                self.size += 1
                self.goal_sum += goal
                self.mean_goal = self.goal_sum / (self.size + 1e-6)

    def sample_goals(self, num):
        if self.size < num:
            return np.random.randn(num, self.g_dim)
        idxs = np.random.randint(low = 0, high = self.size, size = num)
        goals = [self.old_goal_list[i] for i in idxs]
        return np.array(goals, dtype = np.float32)

    def __len__(self):
        return self.size