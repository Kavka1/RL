import torch
import numpy as np
import os
import argparse
import gym
from memory import Memory
from agent import TD3Agent
from torch.utils.tensorboard import SummaryWriter
import datetime
import time

def add_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--load_model_remark', type=str, default='best')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--env', type=str, default='Reacher-v2')
    parser.add_argument('--lr_pi', type=float, default=3e-4)
    parser.add_argument('--lr_q', type=float, default=3e-4)
    parser.add_argument('--lr_alpha', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--noise_std', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--explore_noise_std', type=float, default=0.1)
    parser.add_argument('--memory_size', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_episode', type=int, default=3000)
    parser.add_argument('--training_start', type=int, default=10000)
    parser.add_argument('--policy_update_interval', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=10)

    args = parser.parse_args()
    return args

def train(args):
    env = gym.make(args.env)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = TD3Agent(s_dim=env.observation_space.shape[0], a_dim=env.action_space.shape[0], action_space=env.action_space, args= args)
    memory = Memory(memory_size=args.memory_size, s_dim=env.observation_space.shape[0], a_dim=env.action_space.shape[0])

    writer = SummaryWriter('results/{}_TD3_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env))
    start_time = time.time()
    accumulation_r = 0
    avg_r = 0
    average_rewards = []
    total_step = 0
    for i_episode in range(args.max_episode):
        s = env.reset()
        done = False
        while not done:
            if args.render:
                env.render()
            a = (agent.choose_action(s) + np.random.normal(0., env.action_space.high * args.explore_noise_std)).clip(env.action_space.low, env.action_space.high)
            s_, r, done, _ = env.step(a)
            memory.store(s, a, r, s_, done)

            if total_step >= args.training_start:
                q1_loss, q2_loss, pi_loss = agent.learn(memory, total_step)

                writer.add_scalar('Loss/q1_loss', q1_loss, total_step)
                writer.add_scalar('Loss/q2_loss', q2_loss, total_step)
                writer.add_scalar('Loss/pi_loss', pi_loss, total_step)

            accumulation_r += r
            total_step += 1
            s = s_

        if i_episode % args.log_interval == 0:
            avg_r_new = accumulation_r / args.log_interval
            average_rewards.append(avg_r_new)
            print('training time: {:.2f} | episode: {} | total steps: {} | average reward: {}'.format((time.time() - start_time)/60., i_episode, total_step, avg_r_new))
            writer.add_scalar('average reward per {} episode'.format(args.log_interval), avg_r_new, total_step)
            if avg_r_new > avg_r:
                agent.save_model(env_name=args.env, remarks='best')
                avg_r = avg_r_new
            accumulation_r = 0

def evaluate(args):
    env = gym.make(args.env)
    agent = TD3Agent(
        s_dim=env.observation_space.shape[0],
        a_dim=env.action_space.shape[0],
        action_space=env.action_space,
        args=args
    )
    agent.load_model(
        pi_path='pretrained_models/policy_{}_{}'.format(args.env, args.load_model_remark),
        q1_path='pretrained_models/q1_{}_{}'.format(args.env, args.load_model_remark),
        q2_path='pretrained_models/q2_{}_{}'.format(args.env, args.load_model_remark)
    )

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    accumulation_r = 0
    total_step = 0
    for i_episode in range(args.max_episode):
        s = env.reset()
        done = False
        while not done:
            env.render()
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)

            accumulation_r += r
            total_step += 1
            s = s_

        if i_episode % args.log_interval == 0:
            avg_r = accumulation_r / args.log_interval
            print('episode: {} | total steps: {} | average reward: {}'.format(i_episode, total_step, avg_r))
            accumulation_r = 0


if __name__ == '__main__':
    #env_seq = ['Reacher-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2']
    args = add_argument()
    if args.is_training:
        train(args)
    else:
        evaluate(args)