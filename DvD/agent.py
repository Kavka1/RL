from typing import Dict, List, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TD3(object):
    def __init__(self, o_dim: Union[int, np.int32], a_dim: Union[int, np.int32], config: Dict) -> None:
        super(TD3).__init__()
        # Todo: complement