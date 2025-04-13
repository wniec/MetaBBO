"""
This file is used to train the agent.(for the kind of optimizer that is learnable)
"""

import pickle
from tqdm import tqdm
from environment.basic_environment import PBO_Env
from utils import  construct_problem_set
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
from agent import RLEPSO_Agent   # noqa: F401
from optimizer import RLEPSO_Optimizer   # noqa: F401

matplotlib.use("Agg")


class Trainer(object):
    def __init__(self, config):
        self.config = config
        if config.resume_dir is None:
            self.agent = eval(config.train_agent)(config)
        else:
            file_path = config.resume_dir + config.train_agent + ".pkl"
            with open(file_path, "rb") as f:
                self.agent = pickle.load(f)
            self.agent.update_setting(config)
        self.optimizer = eval(config.train_optimizer)(config)
        self.train_set, self.test_set = construct_problem_set(config)

    def save_log(self, epochs, steps, cost, returns, normalizer):
        log_dir = (
            self.config.log_dir
            + f"/train/{self.agent.__class__.__name__}/{self.config.run_time}/log/"
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return_save = np.stack((steps, returns), 0)
        np.save(log_dir + "return", return_save)
        for problem in self.train_set:
            name = problem.__str__()
            if len(cost[name]) == 0:
                continue
            while len(cost[name]) < len(epochs):
                cost[name].append(cost[name][-1])
                normalizer[name].append(normalizer[name][-1])
            cost_save = np.stack((epochs, cost[name], normalizer[name]), 0)
            np.save(log_dir + name + "_cost", cost_save)

    def draw_cost(self, Name=None, normalize=False):
        log_dir = (
            self.config.log_dir
            + f"/train/{self.agent.__class__.__name__}/{self.config.run_time}/"
        )
        if not os.path.exists(log_dir + "pic/"):
            os.makedirs(log_dir + "pic/")
        for problem in self.train_set:
            if Name is None:
                name = problem.__str__()
            elif (isinstance(Name, str) and problem.__str__() != Name) or (
                isinstance(Name, list) and problem.__str__() not in Name
            ):
                continue
            else:
                name = Name
            plt.figure()
            plt.title(name + "_cost")
            values = np.load(log_dir + "log/" + name + "_cost.npy")
            x, y, n = values
            if normalize:
                y /= n
            plt.plot(x, y)
            plt.savefig(log_dir + f"pic/{name}_cost.png")
            plt.close()

    def draw_average_cost(self, normalize=True):
        log_dir = (
            self.config.log_dir
            + f"/train/{self.agent.__class__.__name__}/{self.config.run_time}/"
        )
        if not os.path.exists(log_dir + "pic/"):
            os.makedirs(log_dir + "pic/")
        X = []
        Y = []
        for problem in self.train_set:
            name = problem.__str__()
            values = np.load(log_dir + "log/" + name + "_cost.npy")
            x, y, n = values
            if normalize:
                y /= n
            X.append(x)
            Y.append(y)
        X = np.mean(X, 0)
        Y = np.mean(Y, 0)
        plt.figure()
        plt.title("all problem cost")
        plt.plot(X, Y)
        plt.savefig(log_dir + "pic/all_problem_cost.png")
        plt.close()

    def draw_return(self):
        log_dir = (
            self.config.log_dir
            + f"/train/{self.agent.__class__.__name__}/{self.config.run_time}/"
        )
        if not os.path.exists(log_dir + "pic/"):
            os.makedirs(log_dir + "pic/")
        plt.figure()
        plt.title("return")
        values = np.load(log_dir + "log/return.npy")
        plt.plot(values[0], values[1])
        plt.savefig(log_dir + "pic/return.png")
        plt.close()

    def train(self):
        print(f"start training: {self.config.run_time}")
        exceed_max_ls = False
        epoch = 0
        cost_record = {}
        normalizer_record = {}
        return_record = []
        learn_steps = []
        epoch_steps = []
        for problem in self.train_set:
            cost_record[problem.__str__()] = []
            normalizer_record[problem.__str__()] = []
        while not exceed_max_ls:
            learn_step = 0
            self.train_set.shuffle()
            with tqdm(
                range(self.train_set.N),
                desc=f"Training {self.agent.__class__.__name__} Epoch {epoch}",
            ) as pbar:
                for problem_id, problem in enumerate(self.train_set):
                    env = PBO_Env(problem, self.optimizer)
                    exceed_max_ls, pbar_info_train = self.agent.train_episode(
                        env
                    )  # pbar_info -> dict
                    pbar.set_postfix(
                        {
                            key: (val.item() if isinstance(val, torch.Tensor) else val)
                            for key, val in pbar_info_train.items()
                        }
                    )
                    pbar.update(1)
                    name = problem.__str__()
                    learn_step = pbar_info_train["learn_steps"]
                    cost_record[name].append(pbar_info_train["gbest"])
                    normalizer_record[name].append(pbar_info_train["normalizer"])
                    return_record.append(pbar_info_train["return"])
                    learn_steps.append(learn_step)
                    if exceed_max_ls:
                        break
                self.agent.train_epoch()
            epoch_steps.append(learn_step)
            self.save_log(
                epoch_steps, learn_steps, cost_record, return_record, normalizer_record
            )
            epoch += 1
            if epoch % self.config.draw_interval == 0:
                self.draw_cost()
                self.draw_average_cost()
                self.draw_return()

        self.draw_cost()
        self.draw_average_cost()
        self.draw_return()
