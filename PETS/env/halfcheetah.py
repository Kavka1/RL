import torch
from gym.envs.mujoco import HalfCheetahEnv
import numpy as np



def task_reward_for_halfcheetah(s: np.array, a: np.array, s_: np.array) -> np.array:
    reward_ctrl = -0.1 * np.linalg.norm(a, ord=2, axis=-1, keepdims=True)
    reward_run = s_[:, :1]
    return reward_run + reward_ctrl


class HalfCheetah(HalfCheetahEnv):
    def __init__(self):
        self.prev_qpos_0 = None
        self.episode_step = 0
        self.episode_len = 1000
        super().__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        
        if self.prev_qpos_0 is not None:  # Defensive coding. First reset_model needs to be called.
            self.prev_qpos_0 = self.sim.data.qpos.flat[0]

        self.episode_step += 1

        if self.episode_step > self.episode_len:
            done = True
        else:
            done = False

        return ob, reward, done, {}

    def _get_obs(self):
        # _get_obs is called also in init to setup the observation space dim. We can ignore that
        qpos_0_delta = self.sim.data.qpos.flat[0] - self.prev_qpos_0 if self.prev_qpos_0 is not None else 0

        return np.concatenate([
            [qpos_0_delta / self.dt],
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.prev_qpos_0 = self.sim.data.qpos.flat[0]
        self.episode_step = 0
        return self._get_obs()

    @staticmethod
    def is_done(states):
        bs = states.shape[0]
        return torch.zeros(size=[bs], dtype=torch.bool, device=states.device)  # Always False