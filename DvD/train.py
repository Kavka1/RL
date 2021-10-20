from pickle import DICT
from typing import Dict, Tuple, List, Union, Type
from gym.core import Env
import numpy as np
import json
import torch
import gym

from population import Agent_Population
from utils import soft_update, evaluate_populations

def train(config: Dict):
    print("loading hyparameters:\n",config)
    
    envs = [gym.make(config['env']) for _ in range(config['population_size'])]
    for env in envs:
        env.seed(config['seed'])
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

    swarm = Agent_Population(config)
    
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
            print("Step: {}  Mean score: {:.5f}  Max score: {:.5f}".format(total_step, mean_score, max_score))


if __name__ == '__main__':
    with open('/home/xukang/GitRepo/RL_repo/DvD/config.json', 'r') as source:
        config = json.load(source)

    train(config)