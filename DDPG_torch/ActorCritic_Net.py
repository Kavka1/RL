import torch
from torch import nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 update_tau = 1e-3
                 ):
        super(Actor, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.tau = update_tau

        #init eval net
        self.eval_net = nn.Sequential(
            nn.Linear(self.input_dim, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, self.output_dim),
            nn.Tanh()
        )
        #init target net
        self.target_net = nn.Sequential(
            nn.Linear(self.input_dim, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, self.output_dim),
            nn.Tanh()
        )
        for eval_param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(eval_param.data)

    def get_eval(self, x):
        return self.eval_net(x)

    def get_target(self, x):
        return self.target_net(x)

    def soft_update(self):
        for eval_param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(eval_param.data*self.tau + target_param.data*(1-self.tau))


class Critic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 update_tau = 1e-3):
        super(Critic, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.tau = update_tau

        #init eval net
        self.l1 = nn.Linear(self.s_dim, 400)
        self.l2 = nn.Linear(400 + self.a_dim, 300)
        self.l3 = nn.Linear(300, 1)
        self.eval_layers = nn.Sequential(self.l1, self.l2, self.l3)
        #init target net
        self.l1_t = nn.Linear(self.s_dim, 400)
        self.l2_t = nn.Linear(400 + self.a_dim, 300)
        self.l3_t = nn.Linear(300, 1)
        self.target_layers = nn.Sequential(self.l1_t, self.l2_t, self.l3_t)
        #update the params
        for eval_param, target_param in zip(self.eval_layers.parameters(), self.target_layers.parameters()):
            target_param.data.copy_(eval_param.data)

    def get_eval(self, s, a):
        l1 = F.relu(self.l1(s))
        l2 = F.relu(self.l2(torch.cat((l1, a), dim=1)))
        return self.l3(l2)

    def get_target(self, s, a):
        l1_t = F.relu(self.l1_t(s))
        l2_t = F.relu(self.l2(torch.cat((l1_t, a), dim=1)))
        return self.l3(l2_t)

    def soft_update(self):
        self.eval_layers = nn.Sequential(self.l1, self.l2, self.l3)
        self.target_layers = nn.Sequential(self.l1_t, self.l2_t, self.l3_t)
        for eval_param, target_param in zip(self.eval_layers.parameters(), self.target_layers.parameters()):
            target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * eval_param.data)


if __name__ == '__main__':
    actor = Actor(input_dim=36, output_dim=2)
    critic = Critic(state_dim=36, action_dim=2)
    print(actor.eval_net, actor.target_net,'\n')

    x = torch.rand(1, 36)
    a_e = actor.get_eval(x)
    a_t = actor.get_target(x)

    c_e = critic.get_eval(x, a_e)
    c_t = critic.get_target(x, a_t)

    print(a_e, a_t, c_e, c_t)

    actor.soft_update()
    critic.soft_update()