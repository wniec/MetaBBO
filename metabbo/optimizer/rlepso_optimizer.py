import torch
import numpy as np
from optimizer.learnable_optimizer import Learnable_Optimizer


class RLEPSO_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)

        config.w_decay = True

        config.NP = 100
        self.__config = config

        self.__dim = config.dim
        self.__w_decay = config.w_decay
        if self.__w_decay:
            self.__w = 0.9
        else:
            self.__w = 0.729

        self.__population_size = config.NP

        indices = torch.range(0, end=self.__population_size - 1)
        self.__pci = 0.05 + 0.45 * torch.exp(10 * indices / (self.__population_size - 1)) / (
            np.exp(10) - 1
        )

        self.__n_group = 5

        self.__no_improve = 0
        self.__per_no_improve = torch.zeros((self.__population_size,))
        self.function_evaluations = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval
        self.__max_function_evaluations = config.maxFEs
        self.__is_done = False
        self.name = "EPSO"

    def init_population(self, problem):
        rand_pos = torch.tensor(
            np.random.uniform(
                low=problem.lb, high=problem.ub, size=(self.__population_size, self.__dim)
            )
        )
        self.__max_velocity = 0.1 * (problem.ub - problem.lb)
        random_velocity = torch.tensor(
            np.random.uniform(
                low=-self.__max_velocity,
                high=self.__max_velocity,
                size=(self.__population_size, self.__dim),
            )
        )
        self.function_evaluations = 0

        c_cost = self.__get_costs(problem, rand_pos)  # ps

        gbest_val = c_cost.min()
        gbest_index = c_cost.argmin()
        gbest_position = rand_pos[gbest_index]
        self.__max_cost = c_cost.max()

        self.__particles: dict[str, torch.Tensor] = {
            "current_position": rand_pos.clone(),  # ps, dim
            "c_cost": c_cost.clone(),  # ps
            "pbest_position": rand_pos.clone(),  # ps, dim
            "pbest": c_cost.clone(),  # ps
            "gbest_position": gbest_position.clone(),  # dim
            "gbest_val": gbest_val,  # 1
            "velocity": random_velocity.clone(),  # ps,dim
            "gbest_index": gbest_index,  # 1
        }
        self.__no_improve -= self.__no_improve
        self.log_index = 1
        self.cost = [self.__particles["gbest_val"]]
        self.__per_no_improve -= self.__per_no_improve
        return self.__get_state()

    # calculate costs of solutions
    def __get_costs(self, problem, position):
        self.function_evaluations += self.__population_size
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum
        return cost

    def __get_v_clpso(self) -> torch.Tensor:
        rand = torch.rand(self.__population_size, self.__dim)
        mask = rand > self.__pci[:, None]
        # tournament selection 2

        target_pos = self.__tournament_selection()
        pbest_clpso = torch.where(mask, self.__particles["pbest_position"], target_pos)
        v_clpso = rand * (pbest_clpso - self.__particles["current_position"])
        return v_clpso

    def __tournament_selection(self) -> torch.Tensor:
        nsel = 2
        rand_index = torch.randint(
            low=0, high=self.__population_size, size=(self.__population_size, self.__dim, nsel)
        )

        candidate = self.__particles["pbest_position"][
            rand_index, torch.arange(self.__dim)[None, :, None]
        ]  # ps, dim, nsel
        candidate_cost = self.__particles["pbest"][rand_index]  # ps, dim, nsel
        target_pos_index = candidate_cost.argmin(dim=-1)  # shape?
        ps_index = torch.arange(self.__population_size)[:, None]
        target_pos = candidate[
            ps_index, torch.arange(self.__dim)[None, :], target_pos_index
        ]
        return target_pos

    def __get_v_fdr(self) -> torch.Tensor:
        pos = self.__particles["pbest_position"]
        distance_per_dim = torch.abs(
            pos[None, :, :].repeat_interleave(self.__population_size, dim=0)
            - pos[:, None, :].repeat_interleave(self.__population_size, dim=1)
        )
        fitness = self.__particles["pbest"]
        fitness_delta = fitness[None, :].repeat_interleave(self.__population_size, dim=0) - fitness[
            :, None
                                                                                            ].repeat_interleave(self.__population_size, dim=1)
        fdr = (fitness_delta[:, :, None]) / (distance_per_dim + 1e-5)
        target_index = fdr.argmin(dim=1)

        dim_index = torch.arange(self.__dim)[None, :]
        target_pos = pos[target_index, dim_index]

        v_fdr = torch.rand(self.__population_size, self.__dim) * (target_pos - pos)
        return v_fdr

    # return coes
    def __get__coefficients(self, actions) -> dict[str, torch.Tensor]:
        assert actions.shape[-1] == self.__n_group * 7, "actions size is not right!"
        ws = torch.zeros(self.__population_size)
        c_mutations = torch.zeros_like(ws)
        c1s, c2s, c3s, c4s = (
            torch.zeros_like(ws),
            torch.zeros_like(ws),
            torch.zeros_like(ws),
            torch.zeros_like(ws),
        )
        per_group_num = self.__population_size // self.__n_group
        for i in range(self.__n_group):
            a = torch.tensor(actions[i * self.__n_group : i * self.__n_group + 7])
            c_mutations[i * per_group_num : (i + 1) * per_group_num] = (
                a[0]
                * 0.01
                * self.__per_no_improve[i * per_group_num : (i + 1) * per_group_num]
            )
            ws[i * per_group_num : (i + 1) * per_group_num] = a[1] * 0.8 + 0.1
            scale = 1.0 / (a[3] + a[4] + a[5] + a[6] + 1e-5) * a[2] * 8
            c1s[i * per_group_num : (i + 1) * per_group_num] = scale * a[3]
            c2s[i * per_group_num : (i + 1) * per_group_num] = scale * a[4]
            c3s[i * per_group_num : (i + 1) * per_group_num] = scale * a[5]
            c4s[i * per_group_num : (i + 1) * per_group_num] = scale * a[6]
        return {
            "w": ws[:, None],
            "c_mutation": c_mutations,
            "c1": c1s[:, None],
            "c2": c2s[:, None],
            "c3": c3s[:, None],
            "c4": c4s[:, None],
        }

    def __reinit(self, mask, problem) -> None:
        if not torch.any(mask):
            return
        rand_pos = torch.tensor(
            np.random.uniform(
                low=problem.lb, high=problem.ub, size=(self.__population_size, self.__dim)
            )
        )
        rand_vel = torch.tensor(
            np.random.uniform(
                low=-self.__max_velocity,
                high=self.__max_velocity,
                size=(self.__population_size, self.__dim),
            )
        )
        new_position = torch.where(mask, rand_pos, self.__particles["current_position"])
        new_velocity = torch.where(mask, rand_vel, self.__particles["velocity"])
        pre_fes = self.function_evaluations
        new_cost = self.__get_costs(problem, new_position)
        self.function_evaluations = pre_fes + mask.sum()

        particles_mask = new_cost < self.__particles["pbest"]
        new_cbest_val = new_cost.min()
        new_cbest_index = new_cost.argmin()

        filters_best_val = new_cbest_val < self.__particles["gbest_val"]
        # update particles
        new_particles = {
            "current_position": new_position,  # bs, ps, dim
            "c_cost": new_cost,  # bs, ps
            "pbest_position": torch.where(
                particles_mask.unsqueeze(-1),
                new_position,
                self.__particles["pbest_position"],
            ),
            "pbest": torch.where(particles_mask, new_cost, self.__particles["pbest"]),
            "velocity": new_velocity,
            "gbest_val": torch.where(
                filters_best_val, new_cbest_val, self.__particles["gbest_val"]
            ),
            "gbest_position": torch.where(
                filters_best_val.unsqueeze(-1),
                new_position[new_cbest_index],
                self.__particles["gbest_position"],
            ),
            "gbest_index": torch.where(
                filters_best_val, new_cbest_index, self.__particles["gbest_index"]
            ),
        }
        self.__particles = new_particles

    def __get_state(self) -> torch.Tensor:
        return torch.tensor([self.function_evaluations / self.__max_function_evaluations])

    def update(self, action, problem) -> tuple[torch.Tensor, int, bool]:
        is_end = False

        previous_gbest = self.__particles["gbest_val"]
        # input action_dim should be : bs, ps
        # action in (0,1) the ratio to learn from pbest & gbest
        rand1 = torch.rand(self.__population_size, 1)
        rand2 = torch.rand(self.__population_size, 1)

        # update velocity
        v_clpso = self.__get_v_clpso()
        v_fdr = self.__get_v_fdr()
        v_pbest = rand1 * (
            self.__particles["pbest_position"] - self.__particles["current_position"]
        )
        v_gbest = rand2 * (
            self.__particles["gbest_position"][None, :]
            - self.__particles["current_position"]
        )
        coefficients = self.__get__coefficients(action)

        new_velocity = (
            coefficients["w"] * self.__particles["velocity"]
            + coefficients["c1"] * v_clpso
            + coefficients["c2"] * v_fdr
            + coefficients["c3"] * v_gbest
            + coefficients["c4"] * v_pbest
        )

        new_velocity = torch.clip(
            new_velocity, -self.__max_velocity, self.__max_velocity
        )

        # update position
        new_position = self.__particles["current_position"] + new_velocity
        new_position = torch.clip(new_position, problem.lb, problem.ub)

        # get new_cost
        new_cost = self.__get_costs(problem, new_position)

        particles_cost_mask = new_cost < self.__particles["pbest"]
        new_cbest_val = new_cost.min()
        new_cbest_index = new_cost.argmin()

        particles_mask_gbest_val = new_cbest_val < self.__particles["gbest_val"]
        # update particles
        new_particles = {
            "current_position": new_position,  # bs, ps, dim
            "c_cost": new_cost,  # bs, ps
            "pbest_position": torch.where(
                particles_cost_mask.unsqueeze(-1),
                new_position,
                self.__particles["pbest_position"],
            ),
            "pbest": torch.where(
                particles_cost_mask, new_cost, self.__particles["pbest"]
            ),
            "velocity": new_velocity,
            "gbest_val": torch.where(
                particles_mask_gbest_val, new_cbest_val, self.__particles["gbest_val"]
            ),
            "gbest_position": torch.where(
                particles_mask_gbest_val.unsqueeze(-1),
                new_position[new_cbest_index],
                self.__particles["gbest_position"],
            ),
            "gbest_index": torch.where(
                particles_mask_gbest_val,
                new_cbest_index,
                self.__particles["gbest_index"],
            ),
        }

        # see if any batch need to be reinitialized
        if new_particles["gbest_val"] < self.__particles["gbest_val"]:
            self.__no_improve = 0
        else:
            self.__no_improve += 1

        particles_patience_mask = new_particles["c_cost"] < self.__particles["c_cost"]
        self.__per_no_improve += 1
        tmp = torch.where(
            particles_patience_mask,
            self.__per_no_improve,
            torch.zeros_like(self.__per_no_improve),
        )
        self.__per_no_improve -= tmp

        self.__particles = new_particles
        # reinitialize according to c_mutation and per_no_improve

        mask_reinit = (
            torch.rand(self.__population_size)
            < coefficients["c_mutation"] * 0.01 * self.__per_no_improve
        )
        self.__reinit(mask_reinit[:, None], problem)

        if self.function_evaluations >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__particles["gbest_val"])

        if problem.optimum is None:
            is_end = self.function_evaluations >= self.__max_function_evaluations
        else:
            is_end = self.function_evaluations >= self.__max_function_evaluations or self.__particles["gbest_val"] <= 1e-8

        # cal the reward
        if self.__particles["gbest_val"] < previous_gbest:
            reward = 1
        else:
            reward = -1
        next_state = self.__get_state()

        if is_end:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__particles["gbest_val"]
            else:
                self.cost.append(self.__particles["gbest_val"])

        return next_state, reward, is_end
