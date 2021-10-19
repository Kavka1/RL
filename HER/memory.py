import numpy as np
import torch
import threading

class Trajectory(object):
    def __init__(self):
        self.trajectory = {
            'observation': [],
            'action': [],
            'reward': [],
            'next_observation': [],
            'goal': [],
            'achieved_goal': [],
            'achieved_goal_next': []
        }

    def __getitem__(self, item):
        return self.trajectory[item]

    def push(self, o, a, r, o_, g, ag, ag_):
        self.trajectory['observation'].append(o)
        self.trajectory['action'].append(a)
        self.trajectory['reward'].append(r)
        self.trajectory['next_observation'].append(o_)
        self.trajectory['goal'].append(g)
        self.trajectory['achieved_goal'].append(ag)
        self.trajectory['achieved_goal_next'].append(ag_)

    def clear(self):
        for key in self.trajectory.keys():
            del self.trajectory[key][:]

    def __len__(self):
        return len(self.trajectory['reward'])


class Memory:
    def __init__(self, size, o_dim, a_dim, g_dim):
        self.size = size
        self.observations = np.zeros(shape=[size, o_dim], dtype=np.float32)
        self.actions = np.zeros(shape=[size, a_dim], dtype=np.float32)
        self.rewards = np.zeros(shape=size, dtype=np.float32)
        self.next_observations = np.zeros(shape=[size, o_dim], dtype=np.float32)
        self.goals = np.zeros(shape=[size, g_dim], dtype=np.float32)

        self.write = 0
        self.num_samples = 0

        self.lock = threading.Lock()

    def __len__(self):
        return self.num_samples

    def clear(self):
        del self.observations
        del self.actions
        del self.rewards
        del self.next_observations
        del self.goals
        self.num_samples = 0
        self.write = 0

    def store(self, o, a, r, o_, g):
        with self.lock:
            idx = self.write
            self.observations[idx] = o
            self.actions[idx] = a
            self.rewards[idx] = r
            self.next_observations[idx] = o_
            self.goals[idx] = g

            self.write = (self.write + 1) % self.size
            self.num_samples = min(self.num_samples + 1, self.size)

    def sample_batch(self, batch_size):
        if batch_size > self.num_samples:
            raise ValueError('No enough samples for one batch')
        idxs = np.random.randint(low=0, high=self.num_samples, size=batch_size)
        with self.lock:
            transitions = self.observations[idxs], self.actions[idxs], self.rewards[idxs], self.next_observations[idxs], self.goals[idxs]
        return transitions

class HerSample:
    def __init__(self, method, k):
        self.method = method
        self.k = k

    def generate_new_goals(self, i, achieved_goals):
        length = len(achieved_goals)
        if self.method == 'future':
            new_goals_idx = np.random.randint(i, length, self.k)
            new_goals = [achieved_goals[idx] for idx in new_goals_idx]
        else:
            raise NotImplementedError('No other method but future')
        return new_goals

    def sample_new_goals(self, trajectory, memory):
        for i in range(len(trajectory)):
            new_goals = self.generate_new_goals(i, trajectory['achieved_goal'])
            for new_goal in new_goals:
                reward = 0 if np.array_equal(trajectory['achieved_goal_next'][i], new_goal) else -1
                memory.store(trajectory['observation'][i], trajectory['action'][i], reward, trajectory['next_observation'][i], new_goal)
