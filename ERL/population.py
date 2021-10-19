from shutil import copy
import torch
import numpy as np
import copy
import random
from model import DeterministicPolicy, GaussianPolicy


class Individual():
    def __init__(self, id, o_dim, a_dim, action_bound, device, eval_episode):
        self.id = id
        self.fitness = 0.
        self.eval_episode = eval_episode
        self.device = device
        self.action_bound = action_bound

        #self.actor = DeterministicPolicy(o_dim, a_dim).to(device)
        self.actor = GaussianPolicy(o_dim, a_dim).to(self.device)

    def choose_action(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            _, _, action = self.actor(obs)
            action = action.cpu().detach().numpy()
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action

    def evaluate(self, env, memory, episode = None):
        total_reward = 0.

        if episode is not None:
            eval_ep = episode
        else:
            eval_ep = self.eval_episode

        for i_episode in range(eval_ep):
            obs = env.reset()
            done = False
            while not done:
                a = self.choose_action(obs)
                obs_, r, done, info = env.step(a)

                memory.store(obs, a, r, obs_, done)
                
                total_reward += r
                obs = obs_
        self.fitness = total_reward / eval_ep

        return copy.copy(self.fitness)


class Population():
    def __init__(self, args, env_params):
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.action_boundary = env_params['action_boundary']
        self.device = torch.device(args.device)

        self.K = args.population_K
        self.evaluate_episode = args.evaluate_episode
        self.elite_num = int(args.elite_frac * self.K)

        self.mut_prob = args.mut_prob
        self.mut_frac = args.mut_frac
        self.super_mut_prob = args.super_mut_prob
        self.reset_prob = args.reset_prob
        self.mut_strength = args.mut_strength

        self.population = []
        for id in range(self.K):
            self.population.append(Individual(id, self.o_dim, self.a_dim, self.action_boundary, self.device, self.evaluate_episode))
    
    def rank_by_fitness(self):
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

    def evaluate_population(self, env, memory, i_generation):
        print('------------Evaluating Generation {}-----------'.format(i_generation))
        average_fitness = 0.
        for id in range(self.K):
            fitness = self.population[id].evaluate(env, memory)
            average_fitness += fitness
            print('--------Individual_ID: {}   Fitness: {}'.format(id, fitness))
        return average_fitness / self.K

    def crossover(self, ind_e, ind_s):
        new_individual = Individual(random.randint(0, self.K), self.o_dim, self.a_dim, self.action_boundary, self.device, self.evaluate_episode)
        new_individual.actor.load_state_dict(ind_e.actor.state_dict())

        for param, param_s in zip(new_individual.actor.parameters(), ind_s.actor.parameters()):
            W = param.data
            W_s = param_s.data
            if len(W.shape) == 2:
                #num_crossovers = int(W.shape[0] * 0.3)
                for iter_cross_over in range(1):
                    if random.random() < 0.6:
                        index = random.randint(0, W.shape[0]-1)
                        W[index, :] = W_s[index, :]
            else:
                if random.random() < 0.8: continue
                for iter_cross_over in range(1):
                    if random.random() < 0.6:
                        index = random.randint(0, W.shape[0]-1)
                        W[index] = W_s[index]
        return new_individual

    def mutate(self, S_set):
        for ind in S_set:
            if np.random.rand() < self.mut_prob:
                for i, param in enumerate(ind.actor.parameters()):
                    W = param.data

                    if len(W.shape) == 2: #weight
                        W_len = W.shape[0] * W.shape[1]
                        for iter in range(int(self.mut_frac * W_len)):
                            x = np.random.randint(0, W.shape[0])
                            y = np.random.randint(0, W.shape[1])

                            if np.random.rand() < self.super_mut_prob:
                                W[x, y] += torch.normal(torch.tensor(0.), torch.tensor(100*self.mut_strength))
                            elif np.random.rand() < self.reset_prob:
                                W[x, y] = torch.normal(torch.tensor(0.), torch.tensor(1.))
                            else:
                                W[x,y] += torch.normal(torch.tensor(0.), torch.tensor(self.mut_strength))
                    else: #bias layer_norm
                        W_len = W.shape[0]
                        for iter in range(int(self.mut_frac * W_len)):
                            x = np.random.randint(0, W.shape[0])
                            if np.random.rand() < self.super_mut_prob:
                                W[x] += torch.normal(torch.tensor(0.), torch.tensor(100*self.mut_strength))
                            elif np.random.rand() < self.reset_prob:
                                W[x] = torch.normal(torch.tensor(0.), torch.tensor(1.))
                            else:
                                W[x] += torch.normal(torch.tensor(0.), torch.tensor(self.mut_strength))
            else:
                pass

    def population_iteration(self, env, memory, i_generation):
        average_fitness = self.evaluate_population(env, memory, i_generation)

        self.rank_by_fitness()
        champion_ind_score = self.population[0].fitness
        
        elite = []
        for idx in range(self.elite_num):
            elite.append(copy.deepcopy(self.population[idx]))

        S_set_idx = []
        S_set = []
        for _ in range(self.K - self.elite_num): 
            i, j = np.random.choice(range(self.K), size=2, replace=True)
            if self.population[i].fitness > self.population[j].fitness:
                S_set_idx.append(i)
            else:
                S_set_idx.append(j)
        S_set_idx = list(set(S_set_idx))
        for idx in S_set_idx:
            S_set.append(copy.deepcopy(self.population[idx]))

        while len(S_set) < self.K - self.elite_num:
            id_e = random.randint(0, len(elite)-1)
            id_s = random.randint(0, len(S_set)-1)
            new_individual = self.crossover(elite[id_e], S_set[id_s])
            S_set.append(new_individual)

        self.mutate(S_set)

        self.population.clear()
        for ind in elite:
            self.population.append(ind)
        for ind in S_set:
            self.population.append(ind)

        return average_fitness, champion_ind_score

    def synchronization(self, actor_model):
        self.rank_by_fitness()
        weakest_actor = self.population[-1].actor
        for param, param_source in zip(weakest_actor.parameters(), actor_model.parameters()):
            param.data.copy_(param_source.data)



