import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import random
from ActorCritic_Net import Actor, Critic
from replay_buffer import ReplayBuffer

class DDPG(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate_a = 1e-3,
                 learning_rate_c = 1e-3,
                 gamma = 0.99,
                 update_tau = 1e-3,
                 batch_size = 100,
                 buffer_size = 10000,
                 training_start = 1000,
                 ):
        super(DDPG, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_a = learning_rate_a
        self.lr_c = learning_rate_c
        self.gamma = gamma
        self.update_tau = update_tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.training_start = training_start
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.actor = Actor(input_dim=self.s_dim, output_dim=self.a_dim, update_tau=self.update_tau).to(self.device)
        self.critic = Critic(state_dim=self.s_dim, action_dim=self.a_dim, update_tau=self.update_tau).to(self.device)
        self.buffer = ReplayBuffer(buffer_size=self.buffer_size)

        self.loss_actor = 0
        self.loss_critic = 0
        self.optimizer_a = optim.Adam(self.actor.eval_net.parameters(), lr=self.lr_a)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, s):
        s = torch.Tensor(s).to(self.device)
        return self.actor.get_eval(s).to(torch.device('cpu')).detach().numpy().tolist()

    def percive(self, state, action, reward, state_, done):
        self.buffer.add(state, action, reward, state_, done)
        if self.training_start < self.buffer.count():
            self.Train()

    def get_critic_loss(self, reward, state_next, state, action, done):
        action_next = self.actor.get_target(state_next)
        q_next_tar = self.critic.get_target(s = state_next, a = action_next)
        Q_target = reward + self.gamma * q_next_tar * (1 - done)
        Q_eval = self.critic.get_eval(s=state, a=action)
        return F.mse_loss(Q_target, Q_eval)

    def Train(self):
        minibatch = self.buffer.get_batch(batch_size=self.batch_size)
        state_batch = torch.Tensor([data[0] for data in minibatch]).to(self.device)
        action_batch = torch.Tensor([data[1] for data in minibatch]).to(self.device)
        reward_batch = torch.Tensor([data[2] for data in minibatch]).to(self.device)
        state_next_batch = torch.Tensor([data[3] for data in minibatch]).to(self.device)
        done_batch = torch.Tensor([data[4] for data in minibatch]).to(self.device)

        #train critic
        self.loss_critic = self.get_critic_loss(reward_batch, state_next_batch, state_batch, action_batch, done_batch)
        self.optimizer_c.zero_grad()
        self.loss_critic.backward()
        self.optimizer_c.step()

        #train actor
        self.loss_actor = -self.critic.get_eval(state_batch, action_batch).mean()
        self.optimizer_a.zero_grad()
        self.loss_actor.backward()
        self.optimizer_a.step()

        #update the target net
        self.actor.soft_update()
        self.critic.soft_update()


if __name__ == '__main__':
    ddpg = DDPG(state_dim=16, action_dim=2)
    for _ in range(5000):
        state = [1.2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        action = ddpg.choose_action(state)
        state_ = [1.2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        reward = 1
        done = 0

        ddpg.percive(state, action, reward, state_, done)
        print('epoch: {}  actor loss:{:.4f}  critic loss:{:.4f}'.format(_, ddpg.loss_actor, ddpg.loss_critic))