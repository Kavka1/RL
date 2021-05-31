from agent import SACAgent
from utils import get_env_params
from parser import *
import gym
import torch
import numpy as np


def evaluate(args):
    env = gym.make(args.env)
    env_params = get_env_params(env)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = SACAgent(args, env_params)
    agent.load_model(remarks = args.load_model_remark)

    total_step = 0
    for i_episode in range(100):
        episode_reward = 0
        done = False

        obs = env.reset()        
        while not done:
            env.render()
            a = agent.choose_action(obs, is_evluate=True)
            obs_, r, done, _ = env.step(a)
            obs = obs_
            
            episode_reward += r
            total_step += 1
        
        print('episode: {} | total steps: {} | episode reward: {:.4f}'.format(i_episode, total_step, episode_reward))


if __name__ == '__main__':
    args = add_arguments()
    evaluate(args)