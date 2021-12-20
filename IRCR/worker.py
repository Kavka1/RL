from typing import Dict, List, Tuple, Type
import numpy as np
from multiprocessing.connection import Connection
from torch.multiprocessing import Pipe
from torch.multiprocessing import Process

from env import Env_wrapper


class Worker(Process):
    def __init__(self, index: int, env: Env_wrapper, child_conn: Connection) -> None:
        super().__init__()
        self.id = index
        self.env = env
        self.child_conn = child_conn

        self.episode_reward = 0
        self.episode_length = 0
        self.episode_count = 0
        self.total_steps = 0
        self.initial_obs = self.env.reset()

    def run(self) -> None:
        super(Worker, self).run()
        while True:
            action = self.child_conn.recv()
            obs, r, done, info = self.env.step(action)
            self.episode_reward += r
            self.episode_length += 1
            self.total_steps += 1

            if done:
                info.update({
                    'episode_step': self.episode_length, 
                    'episode_reward': self.episode_reward, 
                    'episode_count': self.episode_count, 
                    'total_steps': self.total_steps
                })
                print(f"Worker {self.id} complete episode {self.episode_count} rewards: {self.episode_reward} length: {self.episode_length}")
                self.episode_length = 0
                self.episode_reward = 0
                self.episode_count += 1
                obs = self.env.reset()

            self.child_conn.send([obs, r, done, info])
