import gym
from PIL import Image
import numpy as np
import cv2
from torch.multiprocessing import Process
from copy import copy


def unwrap(env):
    if hasattr(env, 'unwrapped'):
        return env.unwrapped
    elif hasattr(env, 'env'):
        return unwrap(env.env)
    elif hasattr(env, 'leg_env'):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkip(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        gym.Wrapper.__init__(self, env)
        self.obs_buffer = np.zeros(shape=(2,)+env.observation_space.shape, dtype=np.uint8)
        self.skip = skip
        self.is_render = is_render

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self.skip):
            if self.is_render:
                self.env.render()

            obs, reward, done, info = self.env.step(action)

            if i == self.skip - 2:
                self.obs_buffer[0] = obs
            if i == self.skip - 1:
                self.obs_buffer[1] = obs
            
            total_reward += reward
            if done:
                break
        max_frame = self.obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info
    
    def reset(self):
        return self.env.reset()


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))

        if done:
            self.visited_rooms.clear()
        
        return obs, r, done, info

    def reset(self):
        return self.env.reset()


class AtariEnvironment(Process):
    def __init__(self, env_id, env_idx, child_conn, is_render, max_episode_step, history_size=4, h=84, w=84, life_done=True):
        super(AtariEnvironment, self).__init__()
        self.env_id = env_id
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.max_episode_step = max_episode_step

        self.env = MaxAndSkip(gym.make(env_id), is_render)
        self.env = MontezumaInfoWrapper(self.env, room_address=3)

        self.episode_step = 0
        self.episode_reward = 0
        self.episode = 0

        self.history_size = history_size
        self.history = np.zeros(shape=[history_size, h, w])
        self.h = h
        self.w = w

        self.last_action = 0
        self.p = 0.25

        self.reset()

    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            if np.random.rand() <= self.p: #sticky action
                action = self.last_action
            self.last_action = action

            obs, r, done, info = self.env.step(action)
            
            if self.episode_step > self.max_episode_step:
                done = True

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(obs)

            self.episode_step += 1
            self.episode_reward += r
            log_r = r

            if done:
                print('[Episode {} Env {}]: episode_step {} episode_reward {} visited_room: [{}]'.format(
                    self.episode, self.env_idx, self.episode_step, self.episode_reward, info.get('episode', {}).get('visited_rooms', {})))
                self.history = self.reset()
            
            self.child_conn.send([self.history[:, :, :], r, done, info])

    def reset(self):
        self.last_action = 0
        self.episode += 1
        self.episode_reward = 0
        self.episode_step = 0
        obs = self.env.reset()
        self.get_init_state(self.pre_proc(obs))
        return self.history[:, :, :]

    def pre_proc(self, X):
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        X = cv2.resize(X, (self.h, self.w))
        return X

    def get_init_state(self, obs):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(obs)
