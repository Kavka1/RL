from typing import List, Dict, Tuple
import torch
import yaml
import argparse
from env import Env_wrapper
from model import Policy
from utils import MeanStdFilter


def load_exp():
    result_path = '/home/xukang/GitRepo/RL/IRCR/results/'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, default='results/HalfCheetah-v2_delay-True_12-20_00-46')
    parser.add_argument('--model', type=str, default='best')
    args = parser.parse_args()
    args = vars(args)

    exp_path = result_path + args['exp_path'].split('/')[-1] + '/'
    model_remark = args['model']

    with open(exp_path + 'config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
    return config, model_remark
    

def demo():
    config, model_remark = load_exp()
    policy = Policy(config['model_config'], device=torch.device('cpu'))
    
    obs_filter = MeanStdFilter(shape=config['model_config']['o_dim'])
    obs_filter.load_params(path = config['exp_path'] + f'filters/{model_remark}.npy')
    
    env = Env_wrapper(config['env_config'])

    for i_episode in range(50):
        policy.load_model(config['exp_path'] + f"models/model_" + model_remark)
        reward, step = 0, 0
        done = False
        obs = env.reset()
        while not done:
            env.render()
            obs = obs_filter(obs)
            a = policy(obs, evaluation=True)
            obs, r, done, _ = env.step(a)
            reward += r
            step += 1
        print(f"Episode: {i_episode}  total step: {step} episode reward: {reward}")


if __name__ == '__main__':
    demo()