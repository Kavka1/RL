import torch
import numpy as np
import gym
import datetime
from torch.utils.tensorboard import SummaryWriter
from arguments import add_arguments
from agent import DDPG_Her_Agent
from memory import Memory, Trajectory
from utils import get_env_params
from mpi4py import MPI


def evaluate(args, agent, env, env_params):
    success_count = 0
    evaluate_episode = args.evaluate_episode
    for i_episode in range(evaluate_episode):
        obs = env.reset()
        cumulative_return = 0
        for step in range(env_params['max_episode_steps']):
            action = agent.select_action(obs['observation'], obs['desired_goal'], train_mode=False)
            obs_, reward, done, info = env.step(action)

            obs = obs_
            cumulative_return += reward
            if done:
                break
        success_count += info['is_success']

    success_rate = round(success_count/evaluate_episode, 4)
    global_success_rate = MPI.COMM_WORLD.allreduce(success_rate, op=MPI.SUM)

    return global_success_rate / MPI.COMM_WORLD.Get_size()


def test(args):
    seed = args.seed
    env = gym.make(args.env)
    env_params = get_env_params(env)

    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = DDPG_Her_Agent(args, env_params)
    agent.load_model(remarks=args.load_model_remark)

    total_steps = 0
    success_count = 0
    for i_episode in range(args.evaluate_episode):
        obs = env.reset()
        cumulative_return = 0
        round_steps = 0
        for step in range(env_params['max_episode_steps']):
            env.render()
            action = agent.select_action(obs['observation'], obs['desired_goal'], train_mode=False)
            obs_, reward, done, info = env.step(action)

            obs = obs_
            total_steps += 1
            round_steps += 1
            cumulative_return += reward
            if done:
                break

        success_count += info['is_success']
        print("Train: Episode: {}   Round_steps: {}   Total_steps: {}   Cumulative_return: {}  is_success: {}".format(i_episode, round_steps, total_steps, cumulative_return, info['is_success']))

    print('success_rate: {:.4f}'.format(success_count/args.evaluate_episode))


if __name__ == '__main__':
    args = add_arguments()
    test(args)
