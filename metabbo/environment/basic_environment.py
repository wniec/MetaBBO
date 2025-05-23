from typing import Any

import torch
from problem.basic_problem import Basic_Problem
from optimizer.learnable_optimizer import Learnable_Optimizer


class PBO_Env:
    """
    An environment with a problem and an optimizer.
    """

    def __init__(
        self,
        problem: Basic_Problem,
        optimizer: Learnable_Optimizer,
    ):
        self.problem = problem
        self.optimizer = optimizer

    def reset(self) -> torch.Tensor:
        self.problem.reset()
        return self.optimizer.init_population(self.problem)

    def step(self, action: Any) -> tuple[torch.Tensor, int, bool]:
        return self.optimizer.update(action, self.problem)
