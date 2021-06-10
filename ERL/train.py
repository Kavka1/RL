import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import datetime
from parser import add_arguments
from agent import DDPGAgent, SACAgent
from population import Population
from memory import Memory
from utils import get_env_params


def train(args):
    logger = SummaryWriter('results/{}_{}_{}'.format(args.env, args.seed, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env)
    env_params = get_env_params(env)
    env.seed(args.seed)

    #agent = DDPGAgent(args, env_params)
    agent = SACAgent(args, env_params)
    pop = Population(args, env_params)
    memory = Memory(args.memory_size, env_params['o_dim'], env_params['a_dim'])

    for i_generation in range(args.generation_episode):
        pop_average_fitness, champion_score = pop.population_iteration(env, memory, i_generation)

        episode_reward = agent.rollout(env, memory)
        
        if memory.num_sample > agent.batch_size:
            agent.update(memory, i_generation)
        
        if i_generation % args.sync_interval == 0:
            pop.synchronization(agent.policy)

        print('--------- Generation_iter: {}  RL episode reward: {}--------'.format(i_generation, episode_reward))
        logger.add_scalar('Indicator/RL_agent episodic rward', episode_reward, i_generation)
        logger.add_scalar('Indicator/Champion individual fitness', champion_score, i_generation)
        logger.add_scalar('Indicator/Population average fitness', pop_average_fitness, i_generation)


if __name__ == '__main__':
    args = add_arguments()
    train(args)