from typing import Dict, List
from torch.utils.tensorboard import SummaryWriter
import gym

from RL.MBPO.agent import MBPO_Agent
from RL.MBPO.env_utils import call_terminal_func
from RL.MBPO.utils import seed_all, make_exp_path


def train(config: Dict, exp_name: str = None) -> False:
    env = gym.make(f"{config['env']}")
    config['model_config'].update({
        's_dim':    env.observation_space.shape[0],
        'a_dim':    env.action_space.shape[0],
    })
    seed_all(config['seed'], env)
    make_exp_path(config, exp_name)
    # logger
    tb = SummaryWriter(config['exp_path'])
    # agent
    agent = MBPO_Agent(config)
    # start rollout
    total_step, total_episode = 0, 0
    # exploration stage
    s = env.reset()
    while total_step < config['init_exploration_step']:
        a = agent.sample_action(s, True)
        s_, r, done, info = env.step(a)
        agent.env_buffer.store((s,a,r,done,s_))
        if done:
            s = env.reset()
            total_episode += 1
        else:
            s = s_
        total_step += 1
    print('exploration stage finished')
    # formal train stage
    s = env.reset()
    while total_step < config['max_step']:
        # evaluate
        if total_step % config['eval_freq'] == 0:
            eval_score = agent.evaluate(env, config['eval_episode'])
            # log
            print(f"step {total_step} episode: {total_episode} eval_score: {eval_score} loss: {agent.loss_log}")
            # tb
            tb.add_scalar(f'train/eval score', eval_score, total_step)
            for key, val in list(agent.loss_log.items()):
                tb.add_scalar(f'train/{key}', val, total_step)
            # reset the env
            s = env.reset()
        if total_step % config['save_freq'] == 0:
            agent.save_all_module(f"{total_step}")
        # interaction
        a = agent.sample_action(s, True)
        s_, r, done, info = env.step(a)
        agent.env_buffer.store((s,a,r,done,s_))
        if done:
            s = env.reset()
            total_episode += 1
        else:
            s = s_
        # rollout and train model
        if total_step % config['model_train_freq'] == 0:
            agent.train_model()
            agent.reset_rollout_length_and_reallocate_model_buf(total_step)
            agent.rollout_model(deterministic=False, terminal_func=call_terminal_func(config['env']))
        # train ac
        agent.train_ac()
        total_step += 1


if __name__ == '__main__':
    config = {
        'seed': 123456,
        'env':  'Hopper-v2',

        'model_config': {
            's_dim': None,
            'a_dim': None,

            'policy_hiddens': [128, 128],
            'policy_nonlinear': 'ReLU',
            'policy_log_std_min': -10.,
            'policy_log_std_max': 2.,
            'policy_initializer': 'xavier uniform',

            'value_hiddens': [128, 128],
            'value_nonlinear': 'ReLU',
            'value_initializer': 'xavier uniform',

            'dynamics_ensemble_size': 7,
            'dynamics_elite_size':    5,
            'dynamics_trunk_hiddens': [200, 200, 200],
            'dynamics_head_hiddens': [200],
            'dynamics_inner_nonlinear': 'Swish',
            'dynamics_initializer': 'truncated normal',
            'dynamics_init_min_log_var': -10,
            'dynamics_init_max_log_var': 0.5,
            'dynamics_log_var_bound_weight': 0.01,
            'dynamics_weight_decay_coeff': 0.00005,
        },

        'device': 'cuda',

        'lr': 0.0003,
        'gamma':            0.99,
        'tau':              0.005,
        'batch_size':       256,
        'training_delay':   1,
        'alpha':            0.2,

        'min_roll_length':      1,
        'max_roll_length':      15,
        'min_step_for_length':  20000,
        'max_step_for_length':  150000,

        'dynamics_max_train_step_since_update': 5,
        'dynamics_train_holdout_ratio':         0.2,    

        'env_batch_ratio':      0.05,
        'rollout_batch_size':   50000,
        'ac_train_repeat':      20,
        'model_train_freq':     250,
        'env_buffer_size':      1000000,

        'init_exploration_step':    5000,
        'max_step':                 150000,
        'eval_freq':                1000,
        'eval_episode':             5,
        'save_freq':                5000,

        'result_path':              '/home/xukang/GitRepo/RL/MBPO/results/'
    }

    train(config, 'test')