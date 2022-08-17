from typing import Dict
import numpy as np
import torch
import yaml
from copy import copy
import datetime
import gym
from torch.utils.tensorboard import SummaryWriter
from RL.PETS.buffer import Buffer
from RL.PETS.pets import PETS
from RL.PETS.utils import confirm_path


def task_reward_for_inverted_pendulum(s: np.array, a: np.array = None) -> np.array:
    assert len(s.shape) == 2
    return np.array((np.abs(s[:, 1]) <= 0.2), dtype=np.float32)


def task_reward_for_cartpole(s: np.array, a: np.array = None) -> np.array:
    assert len(s.shape) == 2
    car_pos, pole_theta = s[:, :1], s[:, 2:3]
    ee_pos = np.concatenate([car_pos - 0.6 * np.sin(pole_theta), - 0.6 * np.cos(pole_theta)], axis=1)
    return np.exp( - np.sum(np.square(ee_pos - np.array([0.0, 0.6])), axis=1) / (0.6 ** 2))


def task_reward_for_reacher(s: np.array, a: np.array) -> np.array:
    x_diff_to_goal = s[:, 8:9]
    y_diff_to_goal = s[:, 9:10]
    dist_reward = - (x_diff_to_goal ** 2 + y_diff_to_goal ** 2)
    ctrl_reward = - np.sum(np.square(a), -1, keepdims=True)
    return dist_reward + ctrl_reward


def train(config: Dict) -> None:
    config.update({'exp_path': config['result_path'] + f"{config['env']}_{datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}/"})

    env = gym.make(config['env'])

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    env.seed(config['seed'])

    config['model_config'].update({
        's_dim':   env.observation_space.shape[0],
        'a_dim':   env.action_space.shape[0],
        'a_bound': copy(env.action_space.high[0]),
    })

    confirm_path(config['exp_path'])
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, indent=2)

    # logger
    tb = SummaryWriter(config['exp_path'])
    # agent
    if config['use_gt_reward_func']:
        if config['env'] == 'InvertedPendulum-v4':
            agent = PETS(config, task_reward_for_inverted_pendulum)
        elif config['env'] == 'CartPole-v1':
            agent = PETS(config, task_reward_for_cartpole)
        elif config['env'] == 'Reacher-v4':
            agent = PETS(config, task_reward_for_reacher)
        else:
            raise NotImplementedError(f"Invalid env name {config['env']}")
    else:
        agent   = PETS(config)
    
    buffer  = Buffer(config['buffer_size'])

    total_step = 0
    total_episode = 0
    episode_r = 0
    best_episode_r_so_far = -1000
    loss_log = 0
    
    s = env.reset()
    while total_step <= config['max_timestep']:
        if config['render']:
            env.render()

        if total_step < config['warm_up_step']:
            a = env.action_space.sample()
        else:
            a = agent.get_action(s)
        s_, r, done, info = env.step(a)
        buffer.store((s, a, r, s_ - s))        

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



def demo(exp_path: str, model_mark: str) -> None:
    with open(exp_path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    env = gym.make(config['env'])

    if config['use_gt_reward_func']:
        if config['env'] == 'InvertedPendulum-v4':
            agent = PETS(config, task_reward_for_inverted_pendulum)
        elif config['env'] == 'CartPole-v1':
            agent = PETS(config, task_reward_for_cartpole)
        elif config['env'] == 'Reacher-v4':
            agent = PETS(config, task_reward_for_reacher)
        else:
            raise NotImplementedError(f"Invalid env name {config['env']}")
    else:
        agent   = PETS(config)
    
    agent.load(exp_path + f'model/{model_mark}.pt', map_location='cpu')

    for _ in range(100):
        episode_r = 0
        done = False
        s = env.reset()
        while not done:
            env.render()
            a = agent.get_action(s)
            s, r, done, info = env.step(a.detach().cpu().numpy())
            episode_r += r
        print(f"Episode {_} Return: {episode_r}")




if __name__ == '__main__':
    config = {
        'model_config': {
            's_dim': None,
            'a_dim': None,
            'a_bound': None,

            'ensemble_size': 5,
            'model_trunk_hiddens': [200, 200, 200],
            'model_head_hiddens': [200],
            'model_inner_nonlinear': 'ReLU',
            'model_initializer': 'he normal',
            'model_min_log_var': -10,
            'model_max_log_var': 1,
        },

        'env': 'Reacher-v4',
        'use_gt_reward_func': True,
        'render': False,

        'seed': 10,
        'device': 'cuda',

        'horizon': 25,
        'n_particel': 20,
        'batch_size': 256,
        'learning_rate': 0.0001,
        'log_var_bound_weight': 0.001,
        'param_reg_weight': 0.00005,
        'train_iter': 100,

        'cem_max_iter': 25,
        'cem_pop_size': 400,
        'cem_num_elites': 40,

        'max_timestep': 100000,
        'buffer_size': 100000,
        'warm_up_step': 300,

        'train_freq': 5,
        'train_start_step': 300,

        'log_freq': 100,
        'save_freq': 5000,

        'result_path': '/home/xukang/GitRepo/RL/PETS/results/'
    }

    #train(config)
    #demo('/home/xukang/GitRepo/RL/PETS/results/Reacher-v4_08-17-15-32-37/', 5000)