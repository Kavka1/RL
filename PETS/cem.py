from typing import Callable, List
import numpy as np
import scipy.stats as stats


class CEMOptimizer:
    def __init__(
        self,
        solution_dim: int,          # horizon * dim_a
        max_iter: int,
        population_size: int,
        num_elites: int,
        cost_function: Callable,
        upper_bound: np.array,
        lower_bound: np.array,
        epsilon: float = 1e-3,
        alpha: float = 0.1
    ) -> None:
        self.solution_dim   =       solution_dim
        self.max_iter       =       max_iter
        self.pop_size       =       population_size
        self.num_elites     =       num_elites

        self.cost_func      =       cost_function
        self.up_bound       =       upper_bound
        self.lo_bound       =       lower_bound
        self.epsilon        =       epsilon
        self.alpha          =       alpha

        assert num_elites < population_size

    def obtain_solution(self, init_mean: np.array, init_var: np.array) -> np.array:
        mean, var       =           init_mean, init_var
        dist            =           stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        for t in range(self.max_iter):
            lb_dist, ub_dist    =   mean - self.lo_bound, self.up_bound - mean
            constrained_var     =   np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples             =   dist.rvs(size=[self.pop_size, self.solution_dim]) * np.sqrt(constrained_var) + mean
            samples             =   samples.astype(np.float32)

            costs               =   self.cost_func(samples)        #    [num_solution]

            elites              =   samples[np.argsort(costs)][-self.num_elites:]

            new_mean            =   np.mean(elites, 0)
            new_var             =   np.var(elites, 0)

            mean                =   self.alpha * mean + (1 - self.alpha) * new_mean
            var                 =   self.alpha * var + (1 - self.alpha) * new_var

        return mean