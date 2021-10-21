from pickle import DICT
from typing import Dict, Tuple, List, Union, Type
from gym.core import Env
import numpy as np
import json
import torch
import gym
import datetime
from torch.utils.tensorboard import SummaryWriter

from population import Agent_Population
from utils import soft_update, evaluate_populations
from bandit import BernoulliBandit

def train(config: Dict):
    logger = SummaryWriter('results/{}_{}_{}'.format(config['env'], config['seed'], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    print("loading hyparameters:\n",config)
    
    envs = [gym.make(config['env']) for _ in range(config['population_size'])]
    for i, env in enumerate(envs):
        env.seed(config['seed'] + i)
    config.update(
        {
            'o_dim': envs[0].observation_space.shape[0], 
            'a_dim': envs[0].action_space.shape[0],
            'a_max': envs[0].action_space.high[0],
            'a_min': - envs[0].action_space.high[0],
            'env_episode_len': 100
        }
    )

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    bandit = BernoulliBandit(arms=config['bandit_arms'], random_choice_bound=config['bandit_random_bound'])
    swarm = Agent_Population(config)
    swarm.Div_weight = bandit.sample()

    total_step = 0
    all_obs = [env.reset() for env in envs]
    for i_round in range(config['total_round']):
        for i, agent in enumerate(swarm.population):
            env, obs = envs[i], all_obs[i]
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.save_transition([obs, action, reward, next_obs, done])

            if done:
                all_obs[i] = env.reset()
            else:
                all_obs[i] = next_obs    
        
        total_step += len(envs)
        if total_step > config['train_start_step']:
            swarm.update(total_step)

        if total_step % config['evaluate_step_interval'] == 0:
            mean_score, max_score = evaluate_populations(swarm, envs, config['evaluate_episodes'])

            bandit.update_bandit(mean_score)
            swarm.Div_weight = bandit.sample()

            print("Step: {} Div_weight:{} Mean score: {:.5f}  Max score: {:.5f} ".format(total_step, swarm.Div_weight, mean_score, max_score))
            logger.add_scalar("Indicator/mean_score", mean_score, total_step)
            logger.add_scalar("Indicator/max_score", mean_score, total_step)
            logger.add_scalar("Adapted_Parameter/div_weight", swarm.Div_weight, total_step)
            

if __name__ == '__main__':
    with open('/home/xukang/GitRepo/RL_repo/DvD/config.json', 'r') as source:
        config = json.load(source)

    train(config)