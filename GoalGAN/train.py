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


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make(args.env)
    env.seed(args.seed)
    env_params = get_env_params(env)
    
    agent = TD3Agent(args, env_params)
    goal_GAN = GoalGAN(args, env_params)
    goal_buffer = GoalBuffer(env_params['g_dim'], dis_threshold = 0.5)

    logger = SummaryWriter(log_dir='/home/xukang/RL/GoalGAN/results/{}_{}_{}'.format(args.env, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    goal_GAN.initialize_gan(agent, env)
    
    for i_iter in range(args.N_iter):
        goals_from_gan = goal_GAN.sample_goals(num=int(args.goal_num_per_iter * 2/3))
        goals_from_buffer = goal_buffer.sample_goals(num=int(args.goal_num_per_iter * 1/3))
        goals = np.concatenate([goals_from_gan, goals_from_buffer], axis=0)

        plot_goals_scatter(goals, logger, i_iter)

        returns = agent.train_and_evaluate(goals, env, logger)
        
        labels = goal_GAN.label_goals(returns)
        goal_GAN.update(goals, labels, logger)

        goal_buffer.update(goals)

        agent.save_model(remark = '{}_{}_iter{}'.format(args.env, args.seed, i_iter))
    


if __name__ == '__main__':
    args = add_arguments()
    train(args)