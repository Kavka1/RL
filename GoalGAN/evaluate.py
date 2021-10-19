import gym
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from parser import add_arguments
from utils import get_env_params, plot_goals_scatter
from agent import PPOAgent, TD3Agent
from GAN import GoalGAN
from buffer import GoalBuffer


def evaluate(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make(args.env)
    env.seed(args.seed)
    env_params = get_env_params(env)
    
    agent = TD3Agent(args, env_params)
    agent.load_model()

    for i_iter in range(args.N_iter):
        goal = np.random.randn(2)*3
        print('{} goal: {} {}'.format(i_iter, goal[0], goal[1]))

        _ = env.reset()
        obs = env.set_goal(goal)
        for i_step in range(env_params['max_episode_steps']):
            env.render()
            a = agent.select_action(obs['observation'], obs['desired_goal'])
            obs_, reward, done, info = env.step(a)
            obs = obs_


if __name__ == '__main__':
    args = add_arguments()
    evaluate(args)