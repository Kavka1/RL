import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from PERmemory import PERmemory

class QValueFunction(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_units=[24, 24]):
        super(QValueFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], a_dim)
        )

    def forward(self, s):
        return self.model(s)

class Agent():
    def __init__(self, s_dim, a_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min, epsilon_decay_period, targte_update_period, memory_size, batch_size, alpha, beta, beta_increment):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_decay_period = epsilon_decay_period
        self.target_update_period = targte_update_period
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.memory = PERmemory(memory_size, alpha=alpha, beta=beta, beta_increment=beta_increment)
        self.Q = QValueFunction(s_dim, a_dim, hidden_units=[24, 24]).to(self.device)
        self.Qtar = QValueFunction(s_dim, a_dim, hidden_units=[24, 24]).to(self.device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

        self.Qtar.load_state_dict(self.Q.state_dict()) #hard update

    def act(self, s):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.a_dim))
        else:
            s = torch.Tensor(s).to(self.device)
            action_prob = self.Q(s)
            action = torch.argmax(action_prob, dim=-1)
            return action.to(torch.device('cpu')).item()

    def append_sample(self, s, a, r, s_, done):
        s, a, r, s_, done = torch.from_numpy(s).to(self.device), a, torch.tensor(r).to(self.device), torch.from_numpy(s_).to(self.device), torch.tensor(int(done)).to(self.device)
        Q_target = (1 - done) * self.gamma * torch.max(self.Qtar(s_.float())) + r
        Q_eval = self.Q(s.float())[a]
        td_error = torch.abs(Q_target - Q_eval)

        self.memory.add(np.array([s.cpu().tolist(), a, r.cpu().item(), s_.cpu().tolist(), done.cpu().item()]), td_error.cpu().detach().numpy())

    def train(self, steps):
        sample_batch, idx_batch, weight_batch = self.memory.sample(self.batch_size)

        s_batch = []
        a_batch = []
        r_batch = []
        sn_batch = []
        done_batch = []
        for item in sample_batch:
            s_batch.append(item[0])
            a_batch.append(item[1])
            r_batch.append(item[2])
            sn_batch.append(item[3])
            done_batch.append(item[4])

        s = torch.Tensor(s_batch).to(self.device)
        a = torch.LongTensor(a_batch).to(self.device)
        r = torch.Tensor(r_batch).to(self.device)
        s_ = torch.FloatTensor(sn_batch).to(self.device)
        done = torch.Tensor(done_batch).to(self.device)

        Q_target = r + (1 - done) * self.gamma * torch.max(self.Qtar(s_))
        Q_eval = self.Q(s).gather(dim=1, index=a.unsqueeze(dim=1)).squeeze(dim=1)
        #Q_eval = torch.FloatTensor([q_eval[index_a] for q_eval, index_a in zip(Q_eval, a)]).to(self.device)
        td_error = abs(Q_target - Q_eval).to(torch.device('cpu')).detach().numpy()
        for i in range(len(idx_batch)):
            self.memory.update(idx_batch[i], td_error[i])

        weight_batch = torch.FloatTensor(weight_batch).to(self.device)
        Loss = (weight_batch * F.mse_loss(Q_eval, Q_target)).mean()
        self.optimizer.zero_grad()
        Loss.backward()
        self.optimizer.step()

        if steps % self.epsilon_decay_period == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        if steps % self.target_update_period == 0:
            self.Qtar.load_state_dict(self.Q.state_dict())