from typing import List, Dict, Tuple, Type
import numpy as np
import gym


class Env_wrapper(object):
    def __init__(self, env_config: Dict) -> None:
        super().__init__()
        
        self.env_name = env_config['env_name']
        self._max_episode_step = env_config['max_episode_step']
        self.seed = env_config['env_seed']
        self.delaye_reward = env_config['delay_reward']

        self._elapsed_step = 0
        self.episode_reward = 0

        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict]:
        obs, r, done, info = self.env.step(action)
        
        self.episode_reward += r
        self._elapsed_step += 1
        
        if self.delaye_reward:
            r = 0

        if self._elapsed_step >= self._max_episode_step:
            done = True
            if self.episode_reward:
                r = self.episode_reward

        return obs, r, done, info

    def reset(self) -> np.array:
        self._elapsed_step = 0
        self.episode_reward = 0
        return self.env.reset()

    def render(self) -> None:
        return self.env.render()
    
    @property
    def observation_space(self) -> gym.spaces.Box:
        return self.env.observation_space
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return self.env.action_space

