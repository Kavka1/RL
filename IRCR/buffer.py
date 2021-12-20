from typing import Dict, List, Tuple
import collections
import numpy as np
import random


class Buffer(object):
    def __init__(self, max_length: int) -> None:
        super().__init__()
        self.max_len = max_length
        self.buffer = collections.deque(maxlen=max_length)
    
    def push(self, transition: Tuple) -> None:
        self.buffer.append(transition)

    def push_batch(self, trans_batch: List[Tuple]) -> None:
        for item in trans_batch:
            self.push(item)

    def sample(self, batch_size: int) -> List[Tuple]:
        data = random.sample(self.buffer, batch_size)
        obs, a, r, done, obs_ = zip(*data)
        obs, a, r, done, obs_ = np.stack(obs, axis=0), np.stack(a, axis=0), np.array(r), np.array(done), np.stack(obs_, axis=0)
        return obs, a, r, done, obs_