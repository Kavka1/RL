from gym.core import Env
import numpy as np
import torch
from torch.multiprocessing import Pipe
import gym
import datetime
from torch.utils.tensorboard import SummaryWriter
from parser import add_arguments
from utils import get_env_params
from agent import PPOAgent
from env import AtariEnvironment


def workers_initialize(args):
    workers = []
    parent_conns = []
    childen_conns = []
    for idx in range(args.num_worker):
        parent_conn, child_conn = Pipe()
        worker = AtariEnvironment(args.env, idx, child_conn, is_render=False, max_episode_step=args.max_episode_step)

        worker.start()

        workers.append(worker)
        parent_conns.append(parent_conn)
        childen_conns.append(child_conn)
    return workers, parent_conns, childen_conns


def paralle_train(args):
    logger = SummaryWriter(log_dir='results/{}_{}_{}'.format(args.env, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env)
    env_params = get_env_params(env, args)
    env.close()

    agent = PPOAgent(args, env_params)
    workers, parent_conns, children_conns = workers_initialize(args)
    
    obs = np.zeros(shape=[args.num_worker, 4, 84, 84], dtype=np.float32)

    #initialize obs_normalizer
    print('Start initialize obs normalizer....')
    next_obs_batch = []
    for step in range(args.initialize_episode * args.max_episode_step):
        actions = np.random.randint(0, env_params['a_dim'], size = (args.num_worker))
        
        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)
        for parent_conn in parent_conns:
            obs_, r, done, info = parent_conn.recv() 
            next_obs_batch.append(obs_)
        
        if len(next_obs_batch) % (10 * args.num_worker) == 0:
            next_obs_batch = np.stack(next_obs_batch)
            agent.normalizer_obs.update(next_obs_batch)
            next_obs_batch = []
    print('End initialize obs normalizer....')

    log_reward_ex = 0
    log_reward_in = 0
    log_step = 0
    log_episode = 0
    for i_epoch in range(args.max_epoch):
        epoch_obs, epoch_action, epoch_ri, epoch_re, epoch_mask, epoch_next_obs, epoch_logprob = [], [], [], [], [], [], []
        for i_step in range(args.rollout_len):
            actions, log_probs = agent.choose_action(obs)

            for action, parent_conn in zip(actions, parent_conns):
                parent_conn.send(action)

            batch_re, batch_mask, batch_next_obs = [], [], []
            for parent_conn in parent_conns:
                obs_, r_e, done, info = parent_conn.recv()

                batch_next_obs.append(obs_)
                batch_re.append(r_e)
                batch_mask.append(0 if done else 1)
            
            batch_next_obs = np.stack(batch_next_obs)
            batch_re = np.stack(batch_re)
            batch_mask = np.stack(batch_mask)
            batch_ri = agent.compute_intrinsic_reward(batch_next_obs.copy())

            #for log
            log_reward_ex += batch_re[args.log_env_idx]
            log_reward_in += batch_ri[args.log_env_idx]
            log_step += 1
            if batch_mask[args.log_env_idx] == 0:
                log_episode += 1
                logger.add_scalar('Indicator/Reward_ex', log_reward_ex, log_episode)
                logger.add_scalar('Indicator/Reward_in', log_reward_in, log_episode)
                log_reward_ex = 0
                log_reward_in = 0

            epoch_obs.append(obs)
            epoch_action.append(actions)
            epoch_next_obs.append(batch_next_obs)
            epoch_ri.append(batch_ri) 
            epoch_re.append(batch_re)
            epoch_mask.append(batch_mask)
            epoch_logprob.append(log_probs)
            
            obs = batch_next_obs[:, :, :, :]

        epoch_obs = np.stack(epoch_obs)
        epoch_action = np.stack(epoch_action)
        epoch_ri = np.stack(epoch_ri)
        epoch_re = np.stack(epoch_re)
        epoch_mask = np.stack(epoch_mask)
        epoch_next_obs = np.stack(epoch_next_obs)
        epoch_logprob = np.stack(epoch_logprob)

        epoch_obs = np.transpose(epoch_obs, axes=[1, 0, 2, 3, 4])
        epoch_action = np.transpose(epoch_action, axes=[1, 0])
        epoch_ri = np.transpose(epoch_ri, axes=[1, 0])
        epoch_re = np.transpose(epoch_re, axes=[1, 0])
        epoch_mask = np.transpose(epoch_mask, axes=[1, 0])
        epoch_next_obs = np.transpose(epoch_next_obs, axes=[1, 0, 2, 3, 4])
        epoch_logprob = np.transpose(epoch_logprob, axes=[1, 0])

        loss_rnd, loss_a, loss_c = agent.update(epoch_obs, epoch_action, epoch_ri, epoch_re, epoch_mask, epoch_next_obs, epoch_logprob)
            
        used_sample_num = args.rollout_len * args.num_worker * i_epoch
        logger.add_scalar('Loss/loss_RND', loss_rnd, used_sample_num)
        logger.add_scalar('Loss/loss_a', loss_a, used_sample_num)
        logger.add_scalar('Loss/loss_c', loss_c, used_sample_num)

        if i_epoch % args.save_model_interval == 0:
            agent.save_model(remark='{}'.format(i_epoch))

'''
def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env)
    env.seed(args.seed)
    env_params = get_env_params(env)
    agent = PPOAgent(args, env_params)

    logger = SummaryWriter(log_dir='results/{}_{}_{}'.format(args.env, args.seed, datetime.datetime.now().strftime("%Y-%M-%D-%H-%M-%S")))

    total_step = 0
    
    agent.initialize(env)
    for i_episode in range(args.max_episode):
        episode_reward = 0
        obs = env.reset()
        for i_step in range(args.max_episode_step):
            env.render()
            a, log_prob = agent.choose_action(obs)
            obs_, r_e, done, _ = env.step(a)
            mask = 0 if done else 1
            r_i = agent.compute_intrinsic_reward(obs_)
            agent.buffer.store(obs, a, r_i, r_e, mask, obs_, log_prob)

            if total_step % args.rollout_len == 0 and total_step > 0:
                loss_RND, loss_actor, loss_critic = agent.update()
                logger.add_scalar('loss/loss_RND', loss_RND, total_step)
                logger.add_scalar('loss/loss_actor', loss_actor, total_step)
                logger.add_scalar('loss/loss_critic', loss_critic, total_step)

            obs = obs_
            total_step += 1
            episode_reward += r_e
            if done:
                break
            
        print('episode: {}  total step: {}  episode reward: {}'.format(i_episode, total_step, episode_reward))
        logger.add_scalar('Indicator/episode_reward', episode_reward, i_episode)

        if i_episode > args.max_episode * 2/3 and i_episode % args.save_model_interval == 0:
            agent.save_model(remark='episode_{}'.format(i_episode))
'''

if __name__ == '__main__':
    args = add_arguments()
    paralle_train(args)
    