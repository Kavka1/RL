import numpy as np


def get_env_params(env):
    env_prams = dict(
        o_dim = env.observation_space.shape[0],
        a_dim = env.action_space.shape[0],
        action_boundary = env.action_space.high[0]
    )
    return env_prams