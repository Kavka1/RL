import torch
import numpy as np

def soft_update(source_net: torch.nn.Module, target_net: torch.nn.Module, tau: float) -> None:
    for param, param_tar in zip(source_net.parameters(), target_net.parameters()):
        param_tar.data.copy_(tau * param.data + (1 - tau) * param_tar.data)