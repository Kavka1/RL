import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import gym
import numpy as np

class Memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, device):
        super(ActorCritic, self).__init__()
        self.device = device
        #Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        #Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, epislon_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.epsilon_clip = epislon_clip
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.policy = ActorCritic(state_dim, action_dim, action_std, self.device).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        #Monte-Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        #Normalize the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        #Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs).to(self.device), 1).detach()

        #optimize policy for K epochs
        for _ in range(self.K_epochs):
            #Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            #Finding the ratio(pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            #Finding the Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            #take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        #Copy the weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

def main():
    env_name = 'Halfcheetah-v2'
    render = False
    solved_reward = 2000
    log_interval = 20 #print avg reward in the interval
    max_episodes = 10000
    max_timesteps = 1500

    update_timesteps = 4000
    action_std = 0.3
    K_epochs = 80
    epsilon_clip = 0.2
    gamma = 0.99

    lr = 0.0003
    betas = (0.9, 0.999)

    random_seed = 123

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, epsilon_clip)
    print(lr, betas)

    #logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    #training loop
    for i_episode in range(max_episodes):
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1

            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if time_step % update_timesteps == 0:
                ppo.update(memory)
                memory.clear()
                time_step = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        if running_reward > solved_reward*log_interval:
            print("###### Solved! ######")
            torch.save(ppo.policy.state_dict(), './ppo_{}'.format(env_name))
            break

        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = running_reward/log_interval

            print('Episode {} \t Avg Length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()
