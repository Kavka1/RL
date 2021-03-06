#coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#随机种子设置
np.random.seed(1) 
tf.random.set_seed(1)

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False
        ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        #total learning steps
        self.learn_step_counter = 0
        #initialize zero memory[s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        #consist of [target_et, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())

        self.cost_his = []
        
        self.saver = tf.train.Saver()
    
    def _build_net(self):
        #------------all inputs---------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s') #input state
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') #input next state
        self.r = tf.placeholder(tf.float32, [None, ], name='r') #input reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a') #input action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.random_normal_initializer(0.1)

        #--------------------build evaluate_net---------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 64, tf.nn.relu, kernel_initializer = w_initializer, bias_initializer = b_initializer, name = 'e1')
            e2 = tf.layers.dense(e1, 128, tf.nn.relu, kernel_initializer = w_initializer, bias_initializer = b_initializer, name = 'e2')
            e3 = tf.layers.dense(e2, 256, tf.nn.relu, kernel_initializer = w_initializer, bias_initializer = b_initializer, name = 'e3')
            self.q_eval = tf.layers.dense(e3, self.n_actions,  kernel_initializer = w_initializer, bias_initializer  = b_initializer, name = 'q')
        
        #-----------------build target net----------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 64, tf.nn.relu, kernel_initializer = w_initializer, bias_initializer = b_initializer, name = 't1')
            t2 = tf.layers.dense(t1, 128, tf.nn.relu, kernel_initializer = w_initializer, bias_initializer = b_initializer, name = 't2')
            t3 = tf.layers.dense(t2, 256, tf.nn.relu, kernel_initializer = w_initializer, bias_initializer = b_initializer, name = 't3')
            self.q_next = tf.layers.dense(t3, self.n_actions, kernel_initializer = w_initializer, bias_initializer = b_initializer, name = 't')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma*tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices = a_indices)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name = 'TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict = {self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')
        
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size = self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict = {
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            }
        )

        self.cost_his.append(cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
    
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
    
    def save_net(self, path):
        save_path = self.saver.save(self.sess, path)
        print("model saved in %s" % save_path)

if __name__ == '__main__':
    DQN = DeepQNetwork(3, 4, output_graph=True)