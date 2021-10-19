import gym
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def get_env_params(env):
    return dict(
        o_dim = env.observation_space.spaces['observation'].shape[0],
        a_dim = env.action_space.shape[0],
        g_dim = env.observation_space.spaces['desired_goal'].shape[0],
        action_boundary = env.action_space.high,
        max_episode_steps = env._max_episode_steps
    )


def plot_goals_scatter(goals, logger, remark):
    xticks = range(-5, 6)
    yticks = range(-5, 6)

    fig = plt.figure(1, figsize=(6, 6))
    plt.scatter(goals[:, 0], goals[:, 1], color='r', s=5)
    plt.xticks(xticks)
    plt.yticks(yticks)

    logger.add_figure('Goal Dist/Iter_{}'.format(remark), figure=fig)



if __name__ ==  '__main__':
    logger = SummaryWriter(log_dir='/home/xukang/RL/GoalGAN/results/demo')
    for i in range(5):
        goals = np.random.randn(300, 2)
        plot_goals_scatter(goals, logger, i)
