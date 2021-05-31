import torch
import numpy as np
import os
import argparse
import gym
from agent import SACAgent
from utils import get_env_params
from parser import add_arguments
from torch.utils.tensorboard import SummaryWriter
import datetime
import time


def train(args):
    env = gym.make(args.env)
    env_params = get_env_params(env)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = SACAgent(args, env_params)

    writer = SummaryWriter('results/{}_{}_{}'.format(args.env, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    total_step = 0
    update_count = 0
    for i_episode in range(args.max_episode):
        episode_reward = 0
        done = False

        obs = env.reset()        
        while not done:
            if args.render:
                env.render()
            a = agent.choose_action(obs)
            obs_, r, done, _ = env.step(a)
            agent.memory.store(obs, a, r, obs_, done)

            if total_step >= args.train_start_step:
                loss_q, loss_pi, loss_alpha, alpha = agent.update(update_count)

                writer.add_scalar('Loss/q_loss', loss_q, total_step)
                writer.add_scalar('Loss/pi_loss', loss_pi, total_step)
                writer.add_scalar('Loss/alpha_loss', loss_alpha, total_step)
                writer.add_scalar('Indicator/alpha', alpha, total_step)
                update_count += 1

            obs = obs_
            total_step += 1
            episode_reward += r
        
        print('episode: {} | total steps: {} | episode reward: {:.4f}'.format(i_episode, total_step, episode_reward))
        writer.add_scalar('Indicator/episodic reward', episode_reward, i_episode)

        if i_episode > 0 and i_episode % args.save_model_interval == 0:
            agent.save_model(remarks='{}_{}_{}'.format(args.env, args.seed, i_episode))


if __name__ == '__main__':
    args = add_arguments()
    train(args)