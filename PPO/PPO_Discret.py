import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import gym

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        super(ActorCritic, self).__init__()
        self.device = device
        #actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        #critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_prob = self.actor(state)
        dist = Categorical(action_prob)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, betas, gamma, epsilon_clip, K_epochs):
        self.lr = lr
        self.gamma = gamma
        self.betas = betas
        self.epsilon_clip = epsilon_clip
        self.K_epochs = K_epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.net =ActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, device=self.device).to(self.device)
        self.net_old = ActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, device=self.device).to(self.device)
        self.net_old.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        #Monte-Carlo estimate of state reward
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        #Normalize reward
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        #convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        #optimize policy for K epochs
        for _ in range(self.K_epochs):
            #Evaluate old actions and values
            logprobs, state_value, dist_entropy = self.net.evaluate(old_states, old_actions)

            #Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            #Finding Surrogate Loss
            advantages = rewards - state_value.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip)*advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_value, rewards) - 0.01*dist_entropy

            #gradient ascent
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        #hard update
        self.net_old.load_state_dict(self.net.state_dict())

def main():
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = True
    solved_reward = 230 #stop trainning if avg_reward > solved_reward
    log_interval = 20 #print avg reward in the interval
    max_episodes = 50000
    max_timesteps = 300 #max timesteps in each episode
    hidden_size = 64
    update_timesteps = 2000 #update net every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4 #update policy for K epochs
    epsilon_clip = 0.2
    random_seed = 1

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, hidden_size, lr, betas, gamma, epsilon_clip, K_epochs)
    print(lr, betas)

    #logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    #training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            #running policy_old
            action = ppo.net_old.act(state, memory)
            state, reward, done, _ = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            #update id its time
            if timestep % update_timesteps == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        #stopping training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########### Solved! ############")
            torch.save(ppo.net.state_dict(), './PPO_{}'.format(env_name))
            break

        #logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward /= log_interval

            print('Episode {} \t avg_length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()