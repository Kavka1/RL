from typing import List, Tuple
from numpy.core.fromnumeric import mean
import torch
import numpy as np

def soft_update(source_net: torch.nn.Module, target_net: torch.nn.Module, tau: float) -> None:
    for param, param_tar in zip(source_net.parameters(), target_net.parameters()):
        param_tar.data.copy_(tau * param.data + (1 - tau) * param_tar.data)


def evaluate_populations(swarm, envs: List, eval_episode: int) -> Tuple[float, float]:
    scores = np.zeros(shape=len(envs))
    for i, agent in enumerate(swarm.population):
        env = envs[i]
        score = 0
        for i_episode in range(eval_episode):
            obs = env.reset()
            while True:
                action = agent.choose_action(obs, use_noise=False)
                next_obs, reward, done, _ = env.step(action)
                score += reward
                if done:
                    break
                else:
                    obs = next_obs
        scores[i] = score / eval_episode

    return mean(scores), max(scores)