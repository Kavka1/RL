from math import remainder
from typing import Dict, List, Tuple, Type
import numpy as np
import yaml
import pandas as pd
from master import Master
from utils import refine_config, seed_torch_np
from torch.utils.tensorboard import SummaryWriter


def load_config():
    config_path = '/home/xukang/GitRepo/RL/IRCR/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
    return config


def train():
    config = load_config()
    config = refine_config(config)
    seed_torch_np(config['seed'])

    master = Master(config)
    logger = SummaryWriter(config['exp_path'])

    log_q_value, log_q_loss, log_policy_loss, log_entropy_loss, log_alpha = 0, 0, 0, 0, 0

    obs_seq = [[] for _ in master.workers]
    a_seq = [[] for _ in master.workers]
    r_seq = [[] for _ in master.workers]
    done_seq = [[] for _ in master.workers]
    next_obs_seq = [[] for _ in master.workers]
    total_episode_count = 0

    obs_temp = [worker.initial_obs for worker in master.workers]
    for step in range(config['max_timestep']):
        for i in range(config['num_workers']):
            obs = obs_temp[i]
            action = master.policy(master.obs_filter(obs))
            master.parents_conn[i].send(action)
            obs_seq[i].append(obs)
            a_seq[i].append(action)

        for i in range(config['num_workers']):
            next_obs, r, done, info = master.parents_conn[i].recv()
            r_seq[i].append(r)
            done_seq[i].append(done)
            next_obs_seq[i].append(next_obs)

        obs_temp = [next_obs[-1] for next_obs in next_obs_seq]

        for i in range(config['num_workers']):
            if done_seq[i][-1] == True:
                episode_reward = r_seq[i][-1]
                master.transfrom_trans_and_save(obs_seq[i], a_seq[i], episode_reward, done_seq[i], next_obs_seq[i])
                obs_seq[i], a_seq[i], r_seq[i], done_seq[i], next_obs_seq[i] = [], [], [], [], []
                total_episode_count += 1
        
        total_step = step * config['num_workers']

        if total_step > config['start_training_step'] and total_episode_count > 1:
            log_q_value, log_q_loss, log_policy_loss, log_entropy_loss, log_alpha = master.train()
        
        if total_step % config['evaluation_interval'] == 0:
            score = master.evaluation(config['evaluation_rollouts'])
            if score > master.best_score:
                master.save_policy(remark='best')
                master.save_filter(remark='best')
                master.best_score = score
            logger.add_scalar('Indicator/Score', score, total_step)

        logger.add_scalar('Train/q_value', log_q_value, total_step)
        logger.add_scalar('Train/q_loss', log_q_loss, total_step)
        logger.add_scalar('Train/policy_loss', log_policy_loss, total_step)
        logger.add_scalar('Train/entropy_loss', log_entropy_loss, total_step)
        logger.add_scalar('Train/alpha', log_alpha, total_step)

    for worker in master.workers:
        worker.terminate()


if __name__ == '__main__':
    train()