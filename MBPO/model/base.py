from typing import Callable, List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weight(layer, initializer="he normal"):
    if isinstance(layer, nn.Module):
        if initializer == "xavier uniform":
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif initializer == 'xavier normal':
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif initializer == "he normal":
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif initializer == 'orthogonal':
            nn.init.orthogonal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif initializer == 'truncated normal':
            nn.init.trunc_normal_(layer.weight, std=1/(2*np.sqrt(layer.weight.shape[1])))
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Parameter):
        if initializer == "xavier uniform":
            nn.init.xavier_uniform_(layer)
        elif initializer == 'xavier normal':
            nn.init.xavier_normal_(layer)
        elif initializer == "he normal":
            nn.init.kaiming_normal_(layer)
        elif initializer == 'orthogonal':
            nn.init.orthogonal_(layer)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def call_activation(name: str) -> nn.Module:
    if name == 'Identity':
        return nn.Identity
    elif name == 'ReLU':
        return nn.ReLU
    elif name == 'Tanh':
        return nn.Tanh
    elif name == 'Sigmoid':
        return nn.Sigmoid
    elif name == 'SoftMax':
        return nn.Softmax
    elif name == 'ELU':
        return nn.ELU
    elif name == 'LeakyReLU':
        return nn.LeakyReLU
    elif name == 'Swish':
        return Swish
    else:
        raise NotImplementedError(f"Invalid activation name: {name}")


def call_mlp(
    in_dim: int, 
    out_dim: int, 
    hidden_layers: List[int],
    inner_activation: str = 'ReLU',
    output_activation: str = 'Tanh',
    initializer: str = 'he normal',
    layer_factory: Callable = None,
) -> nn.Module:
    module_seq = []
    InterActivation = call_activation(inner_activation)
    OutActivation = call_activation(output_activation)
    
    if not layer_factory:
        factory = nn.Linear
    else:
        factory = layer_factory

    last_dim = in_dim
    for hidden in hidden_layers:
        linear = factory(last_dim, hidden)
        init_weight(linear, initializer)

        module_seq += [linear, InterActivation()]
        last_dim = hidden

    linear = factory(last_dim, out_dim)
    init_weight(linear)
    module_seq += [linear, OutActivation()]

    return nn.Sequential(*module_seq)


class Module(nn.Module):
    def __call__(self, *args, **kwargs) -> None:
        return super().__call__(*args, **kwargs)

    def save(self, f: str, prefix: str = '', keep_vars: bool = False) -> None:
        state_dict = self.state_dict(prefix= prefix, keep_vars=keep_vars)
        torch.save(state_dict, f)

    def load(self, f: str, map_location, strict: bool = True) -> None:
        state_dict = torch.load(f, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)


class Normalizer(Module):
    def __init__(self, dim: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.dim        = dim
        self.epsilon    = epsilon
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.zeros(dim))

    def fit(self, X: torch.tensor) -> None:
        assert torch.is_tensor(X)
        assert X.dim() == 2
        assert X.shape[1] == self.dim
        self.mean.data.copy_(X.mean(dim=0))
        self.std.data.copy_(X.std(dim=0))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return (x - self.mean) / (self.std + self.epsilon)

    def unnormalize(self, normal_x: torch.tensor) -> torch.tensor:
        return self.mean + (self.std * normal_x)



class BatchedLinear(nn.Module):
    def __init__(self, ensemble_size: int, in_dim: int, out_dim: int, initializer: str) -> None:
        super().__init__()
        self.ensemble_size      =       ensemble_size
        self.in_dim             =       in_dim
        self.out_dim            =       out_dim
        self.weight             =       nn.Parameter(torch.empty(ensemble_size, out_dim, in_dim))
        self.bias               =       nn.Parameter(torch.empty(ensemble_size, out_dim))
        # weight initialization
        init_weight(self.weight, initializer)
        init_weight(self.bias, initializer)

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert len(x.shape) == 3
        assert x.shape[0] == self.ensemble_size
        return torch.bmm(x, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)


def unbatch_forward(batch_sequential: nn.ModuleList, input: torch.tensor, index: int) -> torch.tensor:
    for layer in batch_sequential:
        if isinstance(layer, BatchedLinear):
            input = F.linear(input, layer.weight[index], layer.bias[index])
        else:
            input = layer(input)
    return input


