from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


class TrajectoryModel(nn.Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        max_len: int
    ) -> None:
        super().__init__()

        self.s_dim      =   s_dim
        self.a_dim      =   a_dim
        self.max_len    =   max_len

    def forward(
        self,
        states:  torch.tensor,
        actions: torch.tensor,
        rewards: torch.tensor,
        masks:   torch.tensor,
        attention_masks: torch.tensor
    ) -> Tuple:
        return None, None, None

    def get_action(
        self,
        states:     torch.tensor,
        actions:    torch.tensor,
        rewards:    torch.tensor,
        **kwargs
    )   -> torch.Tensor:
        return torch.zeros_like(actions[-1])