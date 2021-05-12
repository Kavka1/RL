import torch
import numpy as np
import gym
import os
from mpi4py import MPI
import datetime
from torch.utils.tensorboard import SummaryWriter
from arguments import add_arguments
from agent import DDPG_Her_Agent
from memory import Memory, Trajectory, HerSample
from utils import get_env_params
from evaluate import evaluate


def train(args):
    seed = args.seed + MPI.COMM_WORLD.Get_rank()
    env = gym.make(args.env)
    env_params = get_env_params(env)

    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = DDPG_Her_Agent(args, env_params)
    memory = Memory(args.memory_size, env_params['o_dim'], env_params['a_dim'], env_params['g_dim'])
    trajectory = Trajectory()
    her_sample = HerSample('future', args.k)
    if MPI.COMM_WORLD.Get_rank() == 0:
        write = SummaryWriter(log_dir='results/DDPG_Her_{}_{}_{}'.format(args.env, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    total_steps = 0
    update_counts = 0
    for i_episode in range(args.max_episode):
        obs = env.reset()
        round_steps = 0
        cumulative_return = 0
        for step in range(env_params['max_episode_steps']):
            action = agent.select_action(obs['observation'], obs['desired_goal'])
            obs_, reward, done, info = env.step(action)

            trajectory.push(obs['observation'], action, reward, obs_['observation'], obs['desired_goal'], obs['achieved_goal'], obs_['achieved_goal'])
            memory.store(obs['observation'], action, reward, obs_['observation'], obs['desired_goal'])

            obs = obs_
            total_steps += 1
            round_steps += 1
            cumulative_return += reward
            if done:
                break

        agent.update_normalizer(trajectory['observation'], trajectory['achieved_goal'])
        her_sample.sample_new_goals(trajectory, memory)
        for i in range(args.update_N):
            loss_q, loss_p, q_value = agent.learn(memory)
            if MPI.COMM_WORLD.Get_rank() == 0:
                write.add_scalar('train_process/q_loss', loss_q, update_counts)
                write.add_scalar('train_process/p_loss', loss_p, update_counts)
                write.add_scalar('train_process/q_value', q_value, update_counts)
            update_counts += 1
        agent.soft_update()

        trajectory.clear()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Train: Episode: {}   Round_steps: {}   Total_steps: {}   Cumulative_return: {}  is_success: {}".format(i_episode, round_steps, total_steps, cumulative_return, info['is_success']))
            if MPI.COMM_WORLD.Get_rank() == 0:
                write.add_scalar('Indicators/episode_reward', cumulative_return, i_episode)

        if i_episode % args.evaluate_interval == 0:
            success_rate = evaluate(args, agent, env, env_params)
            if MPI.COMM_WORLD.Get_rank() == 0:
                write.add_scalar('Indicators/success_rate', success_rate, i_episode)

        if MPI.COMM_WORLD.Get_rank() == 0 and i_episode % args.save_model_interval == 0 and i_episode > args.save_model_start:
            agent.save_model(remarks='{}_{}_{}'.format(args.env, args.seed, i_episode))


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    args = add_arguments()
    train(args)
