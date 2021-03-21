import numpy
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from AC_model import Actor, Critic
import gym

RENDER = False
DISPLAY_REWARD_THRESHOLD = 200

env = gym.make("CartPole-v0")
env.seed(1)
env = env.unwrapped

n_features = env.observation_space.shape[0]
n_actions = env.action_space.n

sess = tf.Session()
actor = Actor(n_actions=n_actions, n_features=n_features, sess= sess, learning_rate=0.001)
critic = Critic(sess=sess, n_features=n_features, learning_rate=0.01)

sess.run(tf.global_variables_initializer())

for i_episode in range(1000):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)
        s_, r, done, info = env.step(a)
        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        s = s_
        t += 1

        if done:
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward *0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD : RENDER = True
            print("i_episode:", i_episode ,"  reward:", int(running_reward))
            break