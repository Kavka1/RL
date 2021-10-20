from os import posix_fadvise
from typing import Dict, List, Tuple, Union, Type
from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import Policy, Twin_Q
from agent import TD3_agent
from memory import Memory
from utils import soft_update

class Agent_Population(object):
    def __init__(self, config: Dict) -> None:
        super(Agent_Population, self).__init__()

        self.population_size = config['population_size']
        self.embedding_dim = config['embedding_dim']
        self.Div_weight = config['Div_weight']
        self.lr = config['lr']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.update_delay = config['update_delay']
        self.l = config['l']
        self.device = torch.device(config['device'])

        self.population = []

        self.critic = Twin_Q(config).to(self.device)
        self.critic_target = Twin_Q(config).to(self.device)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), self.lr)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.memory = Memory(config['memory_size'])

        for _ in range(self.population_size):
            agent = TD3_agent(config)
            agent.twin_q = self.critic
            agent.twin_q_target = self.critic_target
            agent.optimizer_q = None
            agent.memory = self.memory
            self.population.append(agent)
        
        self.optimizer_pop = optim.Adam([{'params': agent.policy.parameters()} for agent in self.population], lr=self.lr)

    def update_critic(self, total_o: List[torch.tensor], total_a: List[torch.tensor], total_r: List[torch.tensor], total_next_o: List[torch.tensor], total_done: List[torch.tensor]) -> float:
        obs_batch = torch.cat(total_o)
        action_batch = torch.cat(total_a)
        r_batch = torch.cat(total_r)
        next_obs_batch = torch.cat(total_next_o)
        done_batch = torch.cat(total_done)

        # compute update label
        with torch.no_grad():
            next_a_batch = [agent.get_target_action(next_obs) for agent, next_obs in zip(self.population, total_next_o)]
            next_a_batch = torch.cat(next_a_batch)
            Q1_target, Q2_target = self.critic_target(next_obs_batch, next_a_batch)
        update_target = r_batch + (1 - done_batch) * torch.min(Q1_target, Q2_target)

        # compute update predict
        Q1_predict = self.critic.Q1_value(obs_batch, action_batch)
        Q2_predict = self.critic.Q2_value(obs_batch, action_batch)

        loss_critic = F.mse_loss(Q1_predict, update_target) + F.mse_loss(Q2_predict, update_target)

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        return loss_critic.item()

    def update_population(self, total_o: List[torch.tensor]) -> Tuple[float, float]:
        performance_loss = self.calculate_performance_loss(total_o)
        divergence_loss = self.calculate_divergence_loss()
        
        loss_population = performance_loss + self.Div_weight * divergence_loss
        self.optimizer_pop.zero_grad()
        loss_population.backward()
        self.optimizer_pop.step()

        return performance_loss.item(), divergence_loss.item()

    def calculate_performance_loss(self, total_o: List[torch.tensor]) -> torch.tensor:
        agent_performance_loss = 0
        for agent, obs in zip(self.population, total_o):
            action = agent.policy(obs)
            agent_performance_loss += - agent.twin_q.Q1_value(obs, action).mean()
        return agent_performance_loss

    def calculate_divergence_loss(self) -> torch.tensor:
        # sample states for each agent
        obs_sample_for_each_agent = [torch.from_numpy(np.array(agent.memory.sample(self.embedding_dim)[0])).to(self.device).float() for agent in self.population]
        # get the behavior embedding
        behavior_embedding = [agent.policy(obs).flatten() for agent, obs in zip(self.population, obs_sample_for_each_agent)]
        behavior_embedding = torch.stack(behavior_embedding)  # |a_dim| * embedding_dim
        # calculate the Div loss via squared exponential kernel
        def SEK(a: torch.tensor, b: torch.tensor) -> torch.tensor:
            return torch.exp(- torch.square(a - b).sum(dim=-1) / (2 * self.l))
        matrix_left = behavior_embedding.unsqueeze(dim=0).expand(behavior_embedding.size(0), -1, -1) # [k, k, |a_dim|*embed_dim]
        matrix_right = behavior_embedding.unsqueeze(dim=1).expand(-1, behavior_embedding.size(0), -1)
        divergence_loss = - torch.logdet(SEK(matrix_left, matrix_right))
        return divergence_loss

    def update(self, step: int) -> Tuple[float, float, float]:
        total_o, total_a, total_r, total_next_o, total_done = [], [], [], [], []

        for i in range(len(self.population)):
            o, a, r, o_, done = self.population[i].memory.sample(self.batch_size)
            total_o.append(torch.from_numpy(np.array(o)).to(self.device).float())
            total_a.append(torch.from_numpy(np.array(a)).to(self.device).float())
            total_r.append(torch.from_numpy(np.array(r)).to(self.device).float().unsqueeze(dim=-1))
            total_next_o.append(torch.from_numpy(np.array(o_)).to(self.device).float())
            total_done.append(torch.from_numpy(np.array(done)).to(self.device).int().unsqueeze(dim=-1))

        loss_critic = self.update_critic(total_o, total_a, total_r, total_next_o, total_done)
        loss_performace, loss_divergence = 0, 0

        if step % self.update_delay:
            loss_performace, loss_divergence = self.update_population(total_o)
            
            for agent in self.population:
                agent.update_target()
        
        return loss_critic, loss_performace, loss_divergence

