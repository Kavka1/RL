from typing import Dict, List, Tuple
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import yaml
import dmc2gym

from RL.CURL.agent import CURL_SACAgent
from RL.CURL.buffer import ReplayBuffer
from RL.CURL.utils import FrameStack, save_config_and_env_params, seed_all, get_env_params, make_exp_path


config_path = '/home/xukang/GitRepo/RL/CURL/config.yaml'


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train(config: Dict) -> None:
    make_exp_path(config)

    env = dmc2gym.make(
        domain_name= config['domain_name'],
        task_name= config['task_name'],
        seed= config['seed'],
        visualize_reward= False,
        from_pixels= True,
        height= config['pre_transform_image_size'],
        width= config['pre_transform_image_size']
    )
    env = FrameStack(env, config['frame_stack'])
    
    env.seed(config['seed'])
    seed_all(config['seed'])

    env_params = get_env_params(env)
    env_params.update({
        'obs_shape': (config['frame_stack']*3, config['img_size'], config['img_size'])
    })

    save_config_and_env_params(config, env_params)

    agent = CURL_SACAgent(config, env_params)
    buffer = ReplayBuffer(
        env_params['preaug_obs_shape'], 
        env_params['a_dim'], 
        config['buffer_size'],
        config['batch_size'],
        config['img_size']
    )

    done = True
    episode_num = 0
    episode_step = 0
    episode_reward = 0
    log_loss = {
        'loss_critic': 0,
        'loss_actor': 0,
        'loss_alpha': 0,
        'loss_cpc': 0
    }

    logger = SummaryWriter(config['exp_path'])

    for step in range(config['max_timestep']):
        if done:
            obs = env.reset()

            print(f"Episode: {episode_num} Total Steps: {step} Episode Steps: {episode_step} Episode Reward: {episode_reward}")
            logger.add_scalar('Indicator/Episode Reward', episode_reward, step)
            for loss_key in list(log_loss.keys()):
                logger.add_scalar(f'Loss/{loss_key}', log_loss[loss_key], step)

            episode_num += 1
            episode_step = 0
            episode_reward = 0
        
        action = agent.selection_action(obs, output_mu=False)

        if step > config['train_start_step']:
            log_loss = agent.update(buffer)

        obs_, r, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )

        buffer.add(obs, action, r, done_bool, obs_)

        obs = obs_
        episode_step += 1
        episode_reward += r

