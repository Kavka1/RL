from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Independent, Normal


class RSSM(nn.Module):
    def __init__(
        self,
        a_dim:      int,
        stoch:      int = 30,
        deter:      int = 200,
        hidden:     int = 200,
        embed:      int = 1024,
        activation: nn.Module = nn.ELU
    ) -> None:
        super().__init__()

        self.stoch_size     = stoch
        self.deter_size     = deter
        self.hidden_size    = hidden

        self.act            = activation()

        self.cell   =   nn.GRUCell(
            input_size  =   self.hidden_size,
            hidden_size =   self.deter_size
        )

        # s + a -> GRU input
        self.fc_input = nn.Sequential(
            nn.Linear(stoch + a_dim, hidden), 
            self.act
        )
        # deter state -> next s prior
        self.fc_prior = nn.Sequential(
            nn.Linear(deter, hidden),
            self.act,
            nn.Linear(hidden, 2 * stoch)
        )
        # deter state + image -> next posterior
        self.fc_post  = nn.Sequential(
            nn.Linear(deter + embed, hidden),
            self.act,
            nn.Linear(hidden, 2 * stoch)
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def initial(self, batch_size: int) -> Dict:
        return dict(
            mean = torch.zeros(batch_size, self.stoch_size, device=self.device),
            std  = torch.zeros(batch_size, self.stoch_size, device=self.device),
            stoch= torch.zeros(batch_size, self.stoch_size, device=self.device),
            deter= torch.zeros(batch_size, self.deter_size, device=self.device)
        )

    def get_feat(self, state: Dict) -> Tensor:
        return torch.cat([state['stoch'], state['deter']], -1)

    def get_dist(self, state: Dict) -> Tensor:
        return Independent(Normal(state['mean'], state['std']), reinterpreted_batch_ndims=1)

    def img_step(self, prev_state: Tensor, prev_action: Tensor) -> Dict:
        """
        Compute next prior given previous prior and action
        Args:
            embed: (B,  D) embeded observations
            prev_action: (B,  D) actions. 
            prev_state: (B, D) or None, initial state
        Returns:
            post: dict, same key as initial(), each (B, D)
            prior: dict, same key as initial(), each (B, D)
        """
        x = torch.cat([prev_state['stoch'], prev_action], dim=-1)
        x = self.fc_input(x)
        x = deter = self.cell(x, prev_state['deter'])
        x = self.fc_prior(x)
        mean, std   = x.chunk(2, dim=-1)
        std         = F.softplus(std) + 0.1
        stoch       = self.get_dist(dict(mean=mean, std=std)).rsample()
        prior       = dict(mean=mean, std=std, stoch=stoch, deter=deter)
        return prior

    def obs_step(self, prev_state: Tensor, prev_action: Tensor, embed: Tensor) -> Tuple[Tensor]:
        """
        Compute next prior and posterior given previous prior and action
        Args:
            embed: (B,  D) embeded observations
            prev_action: (B,  D) actions. 
            prev_state: (B, D) or None, initial state
        Returns:
            post: dict, same key as initial(), each (B, D)
            prior: dict, same key as initial(), each (B, D)
        """
        prior = self.img_step(prev_state, prev_action)
        # compute posterior distribution
        x = torch.cat([prior['deter'], embed], dim=-1)
        x = self.fc_post(x)
        mean, std = x.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        stoch = self.get_dist(dict(mean=mean, std=std)).rsample()
        post  = dict(mean=mean, std=std, stoch=stoch, deter=prior['deter'])
        return post, prior       

    def observe(self, embed: Tensor, action: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor]:
        """
        Compute prior and posterior given initial prior, actions and observations.

        Args:
            embed: (B, T, D) embeded observations
            action: (B, T, D) actions. Note action[t] leads to embed[t]
            state: (B, D) or None, initial state
        Returns:
            post: dict, same key as initial(), each (B, T, D)
            prior: dict, same key as initial(), each (B, T, D)
        """
        B, T, D = action.size()
        if state is None:
            state = self.initial(B)
        prior_list, post_list = [], []
        for t in range(T):
            post_state, prior_state = self.obs_step(state, action[:, t], embed[:, t])
            prior_list.append(prior_state)
            post_list.append(post_state)
            state = post_state
        prior = {k: torch.stack([state[k] for state in prior_list], dim=1) for k in prior_list[0]}      # [B, T, ...]
        post  = {k: torch.stack([state[k] for state in post_list], dim=1) for k in post_list[0]}        # [B, T, ...]
        return post, prior

    def imagine(self, action: Tensor, state: Optional[Tensor] = None) -> Dict:
        """
        Compute priors given initial prior and actions.

        Almost the same as observe so nothing special here
        Args:
            action: (B, T, D) actions. Note action[t] leads to embed[t]
            state: (B, D) or None, initial state
        Returns:
            prior: dict, same key as initial(), each (B, T, D)
        """
        B, T, D = action.size()
        if state is None:
            state = self.initial(B)
        assert isinstance(state, dict)
        prior_list = []
        for t in range(T):
            state = self.img_step(state, action[:, t])
            prior_list.append(state)
        prior = {k: torch.stack([state[k] for state in prior_list], dim=1) for k in prior_list[0]}
        return prior