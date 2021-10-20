from typing import List, Dict
import torch
from collections import deque
import numpy as np
import random

class Memory:
    def __init__(self, memory_size) -> None:
        self.size = memory_size
        self.data = deque(maxlen=self.size)
    
    def save_trans(self, transition: List) -> None:
        self.data.append(transition)
    
    def sample(self, batch_size: int) -> List:
        batch = random.sample(self.data, batch_size)
        return list(zip(*batch))