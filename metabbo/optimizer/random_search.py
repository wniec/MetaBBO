import numpy as np
import torch
from .basic_optimizer import Basic_Optimizer


class Random_search(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__fes = 0
        self.log_index = None
        self.cost = None
        self.__dim = config.dim
        self.__max_fes = config.maxFEs
        self.__NP = 100
        self.__n_logpoint = config.n_logpoint
        self.log_interval = config.log_interval

    def __reset(self, problem):
        self.__fes = 0
        self.cost = []
        self.__random_population(problem, init=True)
        self.cost.append(self.gbest)
        self.log_index = 1

    def __random_population(self, problem, init):
        rand_pos = torch.tensor(
            np.random.uniform(
                low=problem.lb, high=problem.ub, size=(self.__NP, self.__dim)
            )
        )
        if problem.optimum is None:
            cost = problem.eval(rand_pos)
        else:
            cost = problem.eval(rand_pos) - problem.optimum
        self.__fes += self.__NP
        if init:
            self.gbest = cost.min()
        else:
            self.gbest = min(self.gbest, cost.min())

    def run_episode(self, problem):
        problem.reset()
        self.__reset(problem)
        is_done = False
        while not is_done:
            self.__random_population(problem, init=False)
            if self.__fes >= self.log_index * self.log_interval:
                self.log_index += 1
                self.cost.append(self.gbest)

            if problem.optimum is None:
                is_done = self.__fes >= self.__max_fes
            else:
                is_done = self.gbest <= 1e-8 or self.__fes >= self.__max_fes

            if is_done:
                if len(self.cost) >= self.__n_logpoint + 1:
                    self.cost[-1] = self.gbest
                else:
                    self.cost.append(self.gbest)
                break

        return {"cost": self.cost, "fes": self.__fes}
