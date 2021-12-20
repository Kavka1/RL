from typing import Dict, List, Tuple, Union
import numpy as np
import yaml
import os
import datetime
import torch
from env import Env_wrapper


def hard_update(source_model: torch.nn.Module, target_model: torch.nn.Module) -> None:
    target_model.load_state_dict(source_model.state_dict())


def soft_update(source_model: torch.nn.Module, target_model: torch.nn.Module, rho: float) -> None:
    for param_s, param_t in zip(source_model.parameters(), target_model.parameters()):
        param_t.data.copy_(rho * param_s.data + (1 - rho) * param_t.data)


def seed_torch_np(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def check_path(path: str) -> None:
    if os.path.exists(path) is False:
        os.makedirs(path)


def refine_model_config(config: Dict) -> Dict:
    env = Env_wrapper(config['env_config'])
    config['model_config'].update({
        'o_dim': env.observation_space.shape[0],
        'a_dim': env.action_space.shape[0],
        'action_low': float(env.action_space.low[0]),
        'action_high': float(env.action_space.high[0])
    })
    return config


def create_exp_path(config: Dict) -> Dict:
    exp_name = f"{config['env_config']['env_name']}_delay-{config['env_config']['delay_reward']}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}"
    exp_path = config['results_path'] + exp_name + '/'
    check_path(exp_path)
    config.update({'exp_path': exp_path})
    return config


def refine_config(config: Dict) -> Dict:
    config = refine_model_config(config)
    config = create_exp_path(config)
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, indent=2)
    print(f"Experiment config.yaml saved to {config['exp_path']}")
    return config


class RunningStats(object):
    def __init__(self, shape: Union[int, Tuple]) -> None:
        super().__init__()

        self.shape = shape
        self._mean = np.zeros(shape = shape, dtype = np.float64)
        self._square_sum = np.zeros(shape=shape, dtype = np.float64)
        self._count = 0

    def push(self, x):
        n = self._count
        self._count += 1
        if self._count == 1:
            self._mean[...] = x
        else:
            delta = x - self._mean
            self._mean[...] += delta / self._count
            self._square_sum[...] += delta**2 * n / self._count

    @property
    def var(self) -> np.array:
        return self._square_sum / (self._count - 1) if self._count > 1 else np.square(self._mean)

    @property
    def std(self) -> np.array:
        return np.sqrt(self.var)


class MeanStdFilter(object):
    def __init__(self, shape: Union[int, Tuple]) -> None:
        super().__init__()

        self.shape = shape
        self.rs = RunningStats(shape)

    def __call__(self, x: np.array) -> np.array:
        assert x.shape[0] == self.shape, (f"Filter.__call__: x.shape-{x.shape} != filter.shape-{self.shape}")
        return (x - self.rs._mean) / (self.rs.std + 1e-6)

    def trans_batch(self, x_batch: np.array) -> np.array:
        for i in range(len(x_batch)):
            x_batch[i] = (x_batch[i] - self.rs._mean) / (self.rs.std + 1e-6)
        return x_batch

    def push_batch(self, x_batch: np.array) -> None:
        assert len(x_batch.shape) == 2 and x_batch[0].shape[0] == self.shape
        for x in x_batch:
            self.rs.push(x)

    def update(self, mean: np.array, square_sum: np.array, count: int) -> None:
        assert mean.shape[0] == square_sum.shape[0] == self.shape, (f"Filter.update: mean_shape {mean.shape} std_shape: {square_sum.shape} filter_shape: {self.shape}")
        self.rs._mean[...] = mean
        self.rs._square_sum[...] = square_sum
        self.rs._count = count

    def load_params(self, path: str) -> None:
        assert os.path.exists(path)
        mix_params = np.load(path, allow_pickle=True)
        self.update(mix_params[0], mix_params[1], mix_params[2])
        print(f"------Loaded obs filter params from {path}------")

    @property
    def mean(self) -> np.array:
        return self.rs._mean

    @property
    def square_sum(self) -> np.array:
        return self.rs._square_sum

    @property
    def count(self) -> int:
        return self.rs._count
