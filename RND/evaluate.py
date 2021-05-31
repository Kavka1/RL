from agent import PPOAgent
from env import AtariEnvironment
from parser import add_arguments
from utils import get_env_params
import gym
import numpy as np
from torch.multiprocessing import Pipe

def evaluate(args):
    env = gym.make(args.env)
    env_params = get_env_params(env, args)
    env.close()
    
    agent = PPOAgent(args, env_params)
    agent.load_model(load_model_remark=args.load_model_remark)

    parent_conn, child_conn = Pipe()
    worker = AtariEnvironment(args.env, 1, child_conn, is_render=True, max_episode_step=args.max_episode_step)
    worker.start()

    for i_episode in range(100):
        obs = worker.reset()
        while True: 
            obs = np.expand_dims(obs, axis=0)
            action = agent.choose_action(obs/255)

            parent_conn.send(action[0])
            obs_, r, done, info = parent_conn.recv()

            obs = obs_

            if done:
                break


if __name__ == '__main__':
    args = add_arguments()
    evaluate(args)