import gym
from DQN_model import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n, n_features=env.observation_space.shape[0], learning_rate=0.01, e_greedy=0.9, replace_target_iter=100, memory_size=300, e_greedy_increment=0.001,)

total_steps = 0

for i_episode in range(2000):
    observation = env.reset()
    ep_r = 0
    while True:
        env.render()
        #选择动作
        action = RL.choose_action(observation)
        #新状态
        observation_, reward, done, info = env.step(action)

        #定义奖励函数(越靠近中心，theta越小，奖励值越大)
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = 0.2*r1 + 0.8*r2

        RL.store_transition(observation, action, reward, observation)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()
        
        if done:
            print('episode:', i_episode, 'ep_r:', round(ep_r, 2), 'epsilon:', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
    