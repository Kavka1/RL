from typing import List
from typing import List, Tuple, Dict
import numpy as np


class BernoulliBandit:
    def __init__(self, arms: List, random_choice_bound: int) -> None:
        self.arms = arms
        self.random_choice_bound = random_choice_bound

        self.alpha = [1 for _ in range(len(self.arms))]
        self.beta = [1 for _ in range(len(self.arms))]

        self.arm_idx = 1
        self.arm = self.arms[self.arm_idx]
        self.choices = [self.arm_idx]
        self.rewards = [0]
        
    def sample(self):
        if len(self.choices) > self.random_choice_bound:
            samples = np.random.beta(a = self.alpha, b = self.beta)
            arm_idx = np.argmax(samples)
        else:
            arm_idx = np.random.randint(low=0, high=len(self.arms))
        
        self.arm_idx = arm_idx
        self.choices.append(self.arm_idx)
        self.arm = self.arms[self.arm_idx]

        return self.arm

    def update_bandit(self, reward):
        prev_reward = self.rewards[-1]
        self.rewards.append(reward)

        reward = 1 if reward > prev_reward else 0
        self.alpha[self.arm_idx] = self.alpha[self.arm_idx] + reward
        self.beta[self.arm_idx] = self.beta[self.arm_idx] + 1 - reward



if __name__ == '__main__':
    bandit = BernoulliBandit(arms=[0, 0.25, 0.5], random_choice_bound=5)

    for i in range(10):
        arm_value = bandit.sample()
        r = np.random.uniform()
        bandit.update_bandit(r)
        print(f"epoch: {i}  choosed_arm: {arm_value}  reward: {r}")

    print(f"total update process: \n choices: {bandit.choices}\n rewards: {bandit.rewards}\n alpha: {bandit.alpha}\n beta: {bandit.beta}")

