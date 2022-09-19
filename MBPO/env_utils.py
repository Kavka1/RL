from typing import Callable, Dict, List
import numpy as np
import gym


def call_terminal_func(env_name: str) -> Callable:
    if env_name == "Hopper-v2":
        def is_terminal_for_hp(state: np.array, action: np.array, next_state: np.array) -> np.array:
            assert len(state.shape) == len(next_state.shape) == len(action.shape) == 2

            height = next_state[:, 0]
            angle = next_state[:, 1]
            not_done = np.isfinite(next_state).all(axis=-1) \
                        * np.abs(next_state[:, 1:] < 100).all(axis=-1) \
                        * (height > .7) \
                        * (np.abs(angle) < .2)

            done = ~not_done
            return done
        return is_terminal_for_hp
    else:
        raise NotImplementedError(f'no terminal func for env {env_name}')