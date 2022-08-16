from typing import Dict
import numpy as np
import torch
import os
import datetime
import gym
from torch.utils.tensorboard import SummaryWriter
from RL.PETS.buffer import Buffer
from RL.PETS.pets import PETS
from RL.PETS.utils import confirm_path


def train(config: Dict) -> None:
    config.update({'exp_path': config['result_path'] + f"{datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}/"})

    env = gym.make(config['env'])

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    env.seed(config['seed'])

    config['model_config'].update({
        's_dim':   env.observation_space.shape[0],
        'a_dim':   env.observation_space.shape[0],
        'a_bound': env.action_space.high[0],
    })
    # logger
    tb = SummaryWriter(config['exp_path'])
    # agent
    agent   = PETS(config)
    buffer  = Buffer(config['buffer_size'])

    total_step = 0
    total_episode = 0
    episode_r = 0
    best_episode_r_so_far = -100
    loss_log = 0
    
    s = env.reset()
    while total_step <= config['max_timestep']:
        if total_step < config['warm_up_step']:
            a = env.action_space.sample()
        else:
            a = agent.get_action(s)
        s_, r, done, info = env.step(a)
        buffer.store((s, s_, r, s_ - s))        

        if done:
            s = env.reset()
            total_episode += 1
            # update best episode r
            if episode_r > best_episode_r_so_far:
                best_episode_r_so_far = episode_r
            # logger
            print(f"Step: {total_step} Episode: {total_episode} Episodic Rewards: {episode_r} Best Return so far: {best_episode_r_so_far}")
            tb.add_scalar('episode_r_for_step', episode_r, total_step)
            tb.add_scalar('episode_r_for_epi', episode_r, total_episode)
            tb.add_scalar('best_r_so_far_for_step', best_episode_r_so_far, total_step)
            tb.add_scalar('best_r_so_far_for_epi', best_episode_r_so_far, total_episode)
            episode_r = 0
        else:
            s = s_

        if total_step % config['train_freq'] == 0 and total_step > config['train_start_step']:
            all_losses = agent.update(buffer, config['train_iter'])
            loss_log   = np.mean(all_losses)

        if total_step % config['log_freq'] == 0:
            print(f"Step: {total_step} Episode: {total_episode} Best Return so far: {best_episode_r_so_far} Loss: {loss_log}")
            tb.add_scalar('loss', loss_log, total_step)

        if total_step % config['save_freq'] == 0:
            confirm_path(config['exp_path'] + 'model/')
            agent.save(config['exp_path'] + f'model/{total_step}.pt')

        episode_r += r
        total_step += 1




if __name__ == '__main__':
    config = {
        'model_config': {
            's_dim': None,
            'a_dim': None,
            'a_bound': None,

            'ensemble_size': 5,
            'model_trunk_hiddens': [250, 250],
            'model_head_hiddens': [250],
            'model_inner_nonlinear': 'ReLU',
            'model_initializer': 'he normal',
            'model_min_log_var': -10,
            'model_max_log_var': 2,
        },

        'env': 'HalfCheetah-v2',

        'seed': 10,
        'device': 'cuda',

        'horizon': 30,
        'n_particel': 20,
        'log_var_bound_weight': 0.01,
        'batch_size': 256,
        'learning_rate': 0.0001,
        'train_iter': 300,

        'cem_max_iter': 5,
        'cem_pop_size': 500,
        'cem_num_elites': 50,

        'max_timestep': 300000,
        'warm_up_step': 1000,

        'train_freq': 25,
        'train_start_step': 500,

        'log_freq': 200,
        'save_freq': 20000,

        'results_path': '/home/xukang/GitRepo/RL/PETS/results/'
    }

    train(config)