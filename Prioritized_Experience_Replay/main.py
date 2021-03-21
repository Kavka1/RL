import numpy as np
import torch
import gym
import argparse
from visdom import Visdom
from DQN import Agent

def Introduce_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCar-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.25)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--epsilon_decay_factor', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_period', type=int, default=100000)
    parser.add_argument('--memory_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta_increment', type=float, default=0.001)
    parser.add_argument('--train_start_step', type=int, default=500)
    parser.add_argument('--target_update_frequency', type=int, default=1000)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--log_frequency', type=int, default=20)

    args = parser.parse_args()
    return args

def train():
    args = Introduce_Arguments()

    env = gym.make(args.env)
    agent = Agent(
        s_dim = env.observation_space.shape[0],
        a_dim = env.action_space.n,
        lr = args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay_factor,
        epsilon_min=args.epsilon_min,
        epsilon_decay_period=args.epsilon_decay_period,
        targte_update_period=args.target_update_frequency,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta,
        beta_increment=args.beta_increment
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    viz = Visdom()
    viz.line([0.], [0.], win='avg_reward', opts=dict(title='Average reward'))

    rewards = 0
    avg_reward = []
    log_steps = []
    steps = 0
    for i_episode in range(args.episodes):
        s = env.reset()
        done = False
        while not done:
            if args.render:
                env.render()
            a = agent.act(s)
            s_, r, done, _ = env.step(a)
            #r = r if not done else -10
            agent.append_sample(s, a, r, s_, done)
            s = s_
            if steps > args.train_start_step:
                agent.train(steps)

            rewards += r
            steps += 1

        if i_episode % args.log_frequency == 0:
            average_r = rewards/args.log_frequency
            avg_reward.append(average_r)
            log_steps.append(steps)
            print('Epidoe: {} | Step: {} | Avg_Reward: {:.3f} | Epsilon: {:.2f}'.format(i_episode, steps, average_r, agent.epsilon))
            viz.line(avg_reward, log_steps, win='avg_reward', update=None, opts=dict(title='Average reward'))
            rewards = 0


if __name__ == '__main__':
    train()