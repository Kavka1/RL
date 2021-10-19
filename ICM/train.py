from gym import logger
import numpy as np
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
import gym
from agent import DDPG_Agent
from utils import get_env_params
from parser import add_arguments

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env)
    env.seed(args.seed)
    env_params = get_env_params(env)

    agent = DDPG_Agent(args, env_params)
    
    logger = SummaryWriter(log_dir='results/DDPG_{}_{}_{}'.format(args.env, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    
    agent.train(env, logger)


if __name__ == '__main__':
    args = add_arguments()
    train(args)
