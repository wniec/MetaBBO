import json
import os

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Normal
from agent.basic_agent import Basic_Agent
from agent.networks import MLP
from agent.utils import Memory, save_class


class Actor(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(Actor, self).__init__()
        net_config = [
            {"in": 1, "out": 64, "drop_out": 0, "activation": "ReLU"},
            {"in": 64, "out": 32, "drop_out": 0, "activation": "ReLU"},
            {"in": 32, "out": 35, "drop_out": 0, "activation": "None"},
        ]
        self.__mu_net = MLP(net_config)
        self.__sigma_net = MLP(net_config)
        self.__max_sigma = config.max_sigma
        self.__min_sigma = config.min_sigma

    def forward(
        self, x_in, fixed_action=None, require_entropy=False
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):  # x-in: bs*gs*9
        mu = (torch.tanh(self.__mu_net(x_in)) + 1.0) / 2.0
        sigma = (torch.tanh(self.__sigma_net(x_in)) + 1.0) / 2.0 * (
            self.__max_sigma - self.__min_sigma
        ) + self.__min_sigma

        policy = Normal(mu, sigma)

        if fixed_action is not None:
            action = fixed_action
        else:
            action = torch.clamp(policy.sample(), min=0, max=1)
        log_prob: torch.Tensor = policy.log_prob(action).sum()

        if require_entropy:
            entropy: torch.Tensor = policy.entropy()  # for logging only bs,ps,2

            out = (action, log_prob, entropy)
        else:
            out = (
                action,
                log_prob,
            )
        return out


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.__value_head = MLP(
            [
                {
                    "in": 1,
                    "out": 16,
                    "drop_out": 0,
                    "activation": "ReLU",
                },
                {"in": 16, "out": 8, "drop_out": 0, "activation": "ReLU"},
                {"in": 8, "out": 1, "drop_out": 0, "activation": "None"},
            ]
        )

    def forward(self, h_features):
        baseline_value = self.__value_head(h_features)
        return baseline_value.detach().squeeze(), baseline_value.squeeze()


class RLEPSO_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)

        # add specified config
        config.feature_dim = 1
        config.action_dim = 35
        config.action_shape = (35,)
        config.n_step = 10
        config.K_epochs = 3
        config.eps_clip = 0.1
        config.discount_factor = 0.999
        config.max_sigma = 0.7
        config.min_sigma = 0.01
        config.lr = 1e-5
        self.__config = config

        self.__device = self.__config.device
        self.__actor = Actor(config).to(self.__device)

        self.__critic = Critic(config).to(self.__device)

        self.__optimizer_actor = torch.optim.Adam(
            [{"params": self.__actor.parameters(), "lr": config.lr}]
        )
        self.__optimizer_critic = torch.optim.Adam(
            [{"params": self.__critic.parameters(), "lr": config.lr}]
        )

        # init learning time
        self.__learning_time = 0

        self.__cur_checkpoint = 0

        # save init agent
        if self.__cur_checkpoint == 0:
            save_class(
                self.__config.agent_save_dir,
                "checkpoint" + str(self.__cur_checkpoint),
                self,
            )
            self.__cur_checkpoint += 1

    def update_setting(self, config):
        self.__config.max_learning_step = config.max_learning_step
        self.__config.agent_save_dir = config.agent_save_dir
        self.__learning_time = 0
        save_class(self.__config.agent_save_dir, "checkpoint0", self)
        self.__config.save_interval = config.save_interval
        self.__cur_checkpoint = 1

    def train_mini_epochs(
        self,
        env,
        rollout_length,
        state,
        entropy,
        baseline_values,
        baseline_values_detached,
        memory,
    ):
        config = self.__config
        # params for training
        discount_factor = config.discount_factor
        k_epochs = config.K_epochs
        eps_clip = config.eps_clip
        rewards_sum = 0
        old_actions = torch.stack(memory.actions)
        old_states = torch.stack(memory.states).detach()
        old_log_probabilities = torch.stack(memory.logprobs).detach().view(-1)

        # Optimize PSO policy for K mini-epochs:
        old_value = None
        for epoch in range(k_epochs):
            if epoch > 0:
                # Evaluating old actions and values :
                log_probabilities = []
                entropy = []
                baseline_values_detached = []
                baseline_values = []

                for old_action, old_state in list(zip(old_actions, old_states))[
                    :rollout_length
                ]:
                    # get new action_prob
                    _, action_log_probability, policy_entropy = self.__actor(
                        old_state,
                        fixed_action=old_action,
                        require_entropy=True,  # take same action
                    )
                    log_probabilities.append(action_log_probability)
                    entropy.append(policy_entropy.detach().cpu())

                    critic_output_detached, critic_output = self.__critic(old_state)
                    baseline_values_detached.append(critic_output_detached)
                    baseline_values.append(critic_output)

            else:
                log_probabilities = memory.logprobs

            log_probabilities = torch.stack(log_probabilities).view(-1)
            entropy = torch.stack(entropy).view(-1)
            baseline_values_detached = torch.stack(baseline_values_detached).view(-1)
            baseline_values = torch.stack(baseline_values).view(-1)

            # get target value for critic
            returns = []
            # get next value
            return_value = self.__critic(state)[0]

            return_value.clone()
            for r in reversed(memory.rewards):
                return_value = return_value * discount_factor + r
                returns.append(return_value)
            # clip the target:
            returns = torch.stack(returns[::-1], 0)
            returns = returns.view(-1)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(log_probabilities - old_log_probabilities.detach())

            # Finding Surrogate Loss:
            advantages = returns - baseline_values_detached

            surrogate_loss1 = ratios * advantages
            surrogate_loss2 = (
                torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            )
            reinforce_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()

            # define baseline loss
            if old_value is None:
                baseline_loss = ((baseline_values - returns) ** 2).mean()
                old_value = baseline_values.detach()
            else:
                v_pred_clipped = old_value + torch.clamp(
                    baseline_values - old_value, -eps_clip, eps_clip
                )
                v_max = torch.max(
                    ((baseline_values - returns) ** 2),
                    ((v_pred_clipped - returns) ** 2),
                )
                baseline_loss = v_max.mean()

            # check K-L divergence (for logging only)
            approx_kl_divergence = (
                (0.5 * (old_log_probabilities.detach() - log_probabilities) ** 2)
                .mean()
                .detach()
            )
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0

            # update gradient step
            self.__optimizer_actor.zero_grad()
            self.__optimizer_critic.zero_grad()
            baseline_loss.backward()
            reinforce_loss.backward()

            # perform gradient descent
            self.__optimizer_actor.step()
            self.__optimizer_critic.step()
            self.__learning_time += 1

            if self.__learning_time >= (
                self.__config.save_interval * self.__cur_checkpoint
            ):
                save_class(
                    self.__config.agent_save_dir,
                    "checkpoint" + str(self.__cur_checkpoint),
                    self,
                )
                avg_actions_list = [i.tolist() for i in self.average_actions_taken]
                var_actions_list = [i.tolist() for i in self.actions_taken_variance]
                with open("average_actions.json", "w", encoding='utf8') as json_file:
                    json.dump(avg_actions_list, json_file)
                fig, ax = plt.subplots(7, 2, figsize=(8, 8), constrained_layout=True)

                for i in range(7):
                    ax[i, 0].plot([idx for idx, _ in enumerate(avg_actions_list)],
                                  [action[i] for action in avg_actions_list],
                                  linewidth=0.5)
                    ax[i, 1].plot([idx for idx, _ in enumerate(var_actions_list)],
                                  [action[i] for action in var_actions_list],
                                  linewidth=0.5,
                                  color="orange")
                    ax[i, 0].set_title(f"action {i} average value trough training")
                    ax[i, 1].set_title(f"action {i} value variance trough training")
                plt.savefig(os.path.join(self.__config.agent_save_dir, f"actions_checkpoint{self.__cur_checkpoint}.png"))



                self.__cur_checkpoint += 1

            if self.__learning_time >= config.max_learning_step:
                return self.__learning_time >= config.max_learning_step, {
                    "normalizer": env.optimizer.cost[0],
                    "gbest": env.optimizer.cost[-1],
                    "return": rewards_sum,
                    "learn_steps": self.__learning_time,
                }

    def train_episode(self, env: "PBO_Env"):
        actions_taken_avg = []
        actions_taken_var = []
        config = self.__config
        # setup
        memory = Memory()
        # initial instances and solutions
        state = env.reset()
        state = torch.FloatTensor(state).to(self.__device)

        # params for training
        n_steps = config.n_step

        timestep = 0
        rewards_sum = 0
        # initial_cost = obj
        done = False
        # sample trajectory
        while not done:
            step_start = timestep
            entropy = []
            baseline_values_detached = []
            baseline_values = []

            while timestep - step_start < n_steps:
                memory.states.append(state.clone())

                # get model output
                action, log_lh, policy_entropy = self.__actor(
                    state,
                    require_entropy=True,
                )
                actions_taken_avg.append(action.reshape(7, 5).mean(dim=1).detach().cpu())
                actions_taken_var.append(action.detach().cpu().tolist())
                action = action.reshape(config.action_shape)
                memory.actions.append(action.clone().detach())
                action = action.cpu()
                memory.logprobs.append(log_lh)

                entropy.append(policy_entropy.detach().cpu())

                critic_output_detached, critic_output = self.__critic(state)
                baseline_values_detached.append(critic_output_detached)
                baseline_values.append(critic_output)

                # state transient
                next_state, rewards, done = env.step(action)
                rewards_sum += rewards
                memory.rewards.append(torch.FloatTensor([rewards]).to(config.device))

                # next
                timestep += 1
                state = next_state
                state = torch.FloatTensor(state).to(config.device)
                if done:
                    break

            # store info
            rollout_length = timestep - step_start
            self.train_mini_epochs(
                env,
                rollout_length,
                state,
                entropy,
                baseline_values,
                baseline_values_detached,
                memory,
            )
            memory.clear_memory()
            self.average_actions_taken.append(sum(actions_taken_avg) / len(actions_taken_avg))
            self.actions_taken_variance.append(torch.tensor(actions_taken_var).reshape(-1, 7, 5).var(dim=-1).mean(dim=0).detach().cpu())
        return self.__learning_time >= config.max_learning_step, {
            "normalizer": env.optimizer.cost[0],
            "gbest": env.optimizer.cost[-1],
            "return": rewards_sum,
            "learn_steps": self.__learning_time,
        }

    def rollout_episode(self, env):
        is_done = False
        state = env.reset()
        reward_sum = 0
        while not is_done:
            state = torch.FloatTensor(state).to(self.__config.device)
            action = self.__actor(state)[0].cpu().numpy()
            state, reward, is_done = env.step(action)
            reward_sum += reward
        return {
            "cost": env.optimizer.cost,
            "fes": env.optimizer.function_evaluations,
            "return": reward_sum,
        }
