from typing import Dict, List, Tuple
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from RL.CURL.utils import random_crop


class ReplayBuffer(Dataset):
    def __init__(
        self,
        preaug_obs_shape: Tuple,
        a_dim: int,
        capacity: int,
        batch_size: int,
        img_size: int,
    ) -> None:
        super().__init__()
        self.obs_shape = preaug_obs_shape
        self.a_dim = a_dim
        self.capacity = capacity
        self.batch_size = batch_size
        self.img_size = img_size

        obs_dtype = np.float32 if len(preaug_obs_shape) == 1 else np.int8

        self.obs_buf = np.empty((capacity, *preaug_obs_shape), dtype=obs_dtype)
        self.a_buf = np.empty((capacity, a_dim), dtype=np.float32)
        self.next_obs_buf = np.empty((capacity, *preaug_obs_shape), dtype=obs_dtype)
        self.r_buf = np.empty((capacity, 1), dtype=np.float32)
        self.notdone_buf = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.is_full = False

    def add(self, obs: np.array, a: np.array, r: float, done: bool, obs_: np.array) -> None:
        np.copyto(self.obs_buf[self.idx], obs)
        np.copyto(self.a_buf[self.idx], a)
        np.copyto(self.r_buf[self.idx], r)
        np.copyto(self.notdone_buf[self.idx], not done)
        np.copyto(self.next_obs_buf[self.idx], obs_)

        self.idx = (self.idx + 1) % self.capacity
        self.is_full = self.is_full or (self.idx == 0)

    def sample_cpc(self, device: torch.device) -> Tuple:
        idxs = np.random.randint(0, self.capacity if self.is_full else self.idx, self.batch_size)

        obs_batch = self.obs_buf[idxs]
        pos = obs_batch.copy()
        next_obs_batch = self.next_obs_buf[idxs]
        
        obs_batch = random_crop(obs_batch, self.img_size)
        pos = random_crop(pos, self.img_size)
        next_obs_batch = random_crop(next_obs_batch, self.img_size)
        
        obs_batch = torch.from_numpy(obs_batch).float().to(device)
        next_obs_batch = torch.from_numpy(next_obs_batch).float().to(device)
        a_batch = torch.from_numpy(self.a_buf[idxs]).float().to(device)
        r_batch = torch.from_numpy(self.r_buf[idxs]).float().to(device)
        notdone_batch = torch.from_numpy(self.notdone_buf[idxs]).float().to(device)
        
        pos_batch = torch.from_numpy(pos).float().to(device)
        cpc_kwargs = dict(obs_anchor = obs_batch, obs_pos = pos_batch, time_anchor = None, time_pos = None)
        
        return obs_batch, a_batch, r_batch, next_obs_batch, notdone_batch, cpc_kwargs