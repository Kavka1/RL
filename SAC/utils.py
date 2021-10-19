import numpy as np


def get_env_params(env):
    obs = env.reset()
    env_params = dict(
        o_dim = env.observation_space.shape[0],
        a_dim = env.action_space.shape[0],
        action_boundary = [env.action_space.low[0], env.action_space.high[0]],
        action_scale = env.action_space.high[0],
        action_bias = np.array(0, dtype=np.float32)
    )

    return env_params