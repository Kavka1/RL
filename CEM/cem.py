from os.path import join as joindir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import datetime
import math
import gym
from pathos.multiprocessing import ProcessingPool as pool

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_episode', type=int, default=25)
    parser.add_argument('--max_steps_per_round', type=int, default=200)
    parser.add_argument('--log_num_episode', type=int, default=1)
    parser.add_argument('--num_parallel_run', type=int, default=5)
    parser.add_argument('--num_cores', type=int, default=5)
    parser.add_argument('--init_sig', type=float, default=10)
    parser.add_argument('--const_noise_sig', type=float, default=4.0)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--num_round', type=int, default=30)

    args = parser.parse_args()
    return args

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldmean = self._M.__copy__()
            self._M[...] = oldmean + (x - oldmean)*self._n
            self._S[...] = self._S + (x - oldmean)*(x-self._M)

    def n(self):
        return self._n

    def mean(self):
        return self._M

    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    def std(self):
        return np.sqrt(self.var())

    def shape(self):
        return self._M.shape
class ZFilter:
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean()
        if self.destd:
            x = x / (self.rs.std() + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

class Agent(object):
    def __init__(self, W):
        self._W = W

    def act(self, state):
        return np.matmul(state, self._W)

    def run(self, num_round, max_steps_per_round, env_name, env_seed):
        env = gym.make(env_name)
        env.seed(env_seed)

        total_steps = 0
        reward_sum_record = []
        for i_run in range(num_round):
            reward_sum = 0
            num_steps = 0
            done = False
            state = env.reset()
            while (not done) and  (num_steps < max_steps_per_round):
                action = self.act(state)
                state_, reward, done, _ = env.step(action)
                reward_sum += reward
                state = state_
                num_steps += 1
            total_steps += num_steps
            reward_sum_record.append(reward_sum)
        return total_steps, np.mean(reward_sum_record)

class Policy(object):
    def __init__(self, dim_states, dim_actions, init_sig):
        self.shape = (dim_states, dim_actions)
        self.mu = np.zeros(self.shape)
        self.sig = np.ones(np.prod(self.shape)) * init_sig

    def sample(self):
        W = np.random.normal(self.mu.reshape(-1), self.sig).reshape(self.shape)
        return W

    def update(self, weights, constant_noise_sig):
        '''
        given the selected good samples of weights, update accroding to CEM formulation
        weights: first I selected weights, each is the same size as shape
        '''
        self.mu = np.mean(weights, axis=0)
        self.sig = np.sqrt(np.array([np.square(w - self.mu).reshape(-1) for w in weights]).mean(axis=0) + constant_noise_sig)

def get_score_of_weight(W):
    agent = Agent(W)
    return agent.run(args.num_round, args.max_steps_per_round, args.env, args.seed)

def CEM():
    env = gym.make(args.env)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]
    del env

    p = pool(args.num_cores)
    policy = Policy(dim_state, dim_action, args.init_sig)

    reward_record = []
    global_steps = 0
    num_top_samples = int(max(1, np.floor(args.ratio*args.num_samples)))
    for  i_episode in range(args.num_episode):
        weights = []
        scores = []
        for i_sample in range(args.num_samples):
            weights.append(policy.sample())

        res = p.map(get_score_of_weight, weights)
        scores = [score for _, score in res]
        steps = [step for step, _ in res]

        global_steps += np.sum(steps)
        reward_record.append({'steps': global_steps, 'reward': np.mean(scores)})

        selected_samples = [x for _, x in sorted(zip(scores, weights), reverse=True)][:num_top_samples]
        policy.update(selected_samples, args.const_noise_sig)

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} steps: {} AvgReward: {:.4f}'.format(i_episode, reward_record[-1]['steps'], reward_record[-1]['reward']))
            print('------------------------------------------------')

    return reward_record


EPS = 1e-10
RESULT_DIR = '..'
if __name__ == '__main__':
    datestr = datetime.datetime.now().strftime('%Y-%m-%d')
    args = add_arguments()

    record_dfs = pd.DataFrame(columns=['steps', 'reward'])
    reward_cols = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_record = pd.DataFrame(CEM())
        record_dfs = record_dfs.merge(reward_record, how='outer', on='steps', suffixes=('','_{}'.format(i)))
        reward_cols.append('reward_{}'.format(i))

    record_dfs = record_dfs.drop(labels='reward').sort_values(by='steps', ascending=True).ffill().bfill()
    record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
    record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
    record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(span=1000).mean()
    record_dfs['reward_smooth_std'] = record_dfs['reward_std'].ewm(span=1000).mean()
    record_dfs.to_csv(joindir(RESULT_DIR, 'cem-record-{}-{}.csv'.format(args.env_name, datestr)))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(record_dfs['steps'], record_dfs['reward_smooth'], label='reward')
    plt.fill_between(record_dfs['steps'], record_dfs['reward_smooth'] - record_dfs['reward_smooth_std'],
                     record_dfs['reward_smooth'] + record_dfs['reward_smooth_std'], color='b', alpha=0.2)
    plt.legend()
    plt.xlabel('steps of env interaction (sample complexity)')
    plt.ylabel('average reward')
    plt.title('CEM on {}'.format(args.env_name))
    plt.savefig(joindir(RESULT_DIR, 'cem-plot-{}-{}.pdf'.format(args.env_name, datestr)))
