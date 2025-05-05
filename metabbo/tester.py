import copy
from collections import defaultdict
from itertools import product

from utils import construct_problem_set, default_to_regular
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
from tqdm import tqdm
import os
from environment.basic_environment import PBO_Env
from logger import Logger
from optimizer.random_search import Random_search
from optimizer.rlepso_optimizer import RLEPSO_Optimizer  # noqa 403

RUNS_NO = 51


def calculate_time_0(dim, fes) -> float:
    T0 = 0
    for i in range(10):
        start = time.perf_counter()
        for _ in range(fes):
            x = np.random.rand(dim)
            x + x
            x / (x + 2)
            x * x
            np.sqrt(x)
            np.log(x)
            np.exp(x)
        end = time.perf_counter()
        T0 += (end - start) * 1000
    # ms
    return T0 / 10


def calculate_time_1(problem, dim, fes) -> float:
    T1 = 0
    for i in range(10):
        x = np.random.rand(fes, dim)
        start = time.perf_counter()
        problem.eval(x)
        end = time.perf_counter()
        T1 += (end - start) * 1000
    # ms
    return T1 / 10


class Tester(object):
    def __init__(self, config):
        agent_name = config.agent
        agent_load_dir = config.agent_load_dir
        self.agent_name_list = config.agent_for_cp
        self.agent = None
        if agent_name is not None:  # learnable optimizer
            file_path = os.path.join(agent_load_dir, f"{agent_name}.pkl")
            with open(file_path, "rb") as f:
                self.agent = pickle.load(f)
            self.agent_name_list.append(agent_name)
        if config.optimizer is not None:
            self.optimizer_name = config.optimizer
            self.optimizer = eval(config.optimizer)(copy.deepcopy(config))
        self.log_dir = config.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.config = config

        _, self.test_set = construct_problem_set(self.config)

        self.seed = range(RUNS_NO)
        # initialize the dataframe for logging
        self.test_results = {"cost": {}, "fes": {}, "T0": 0.0, "T1": {}, "T2": {}}

        # prepare experimental optimizers and agents
        self.agent_for_cp = []
        for agent in config.agent_for_cp:
            file_path = os.path.join(agent_load_dir, f"{agent}.pkl")
            with open(file_path, "rb") as f:
                self.agent_for_cp.append(pickle.load(f))

        self.learnable_optimizers = [
            eval(optimizer)(copy.deepcopy(config))
            for optimizer in config.l_optimizer_for_cp
        ]
        self.traditional_optimizers = [
            eval(optimizer)(copy.deepcopy(config))
            for optimizer in config.t_optimizer_for_cp
        ]

        if self.agent is not None:
            self.agent_for_cp.append(self.agent)
            self.learnable_optimizers.append(self.optimizer)
        elif config.optimizer is not None:
            self.traditional_optimizers.append(self.optimizer)
        # logging
        self.init_log()

        self.test_results["T1"] = defaultdict(lambda: 0.0)
        self.test_results["T2"] = defaultdict(lambda: 0.0)

        for problem in self.test_set:
            self.test_results["cost"][problem.__str__()] = defaultdict(list)
            self.test_results["fes"][problem.__str__()] = defaultdict(list)

    def init_log(self):
        if len(self.agent_for_cp) == 0:
            print("None of learnable agent")
        else:
            print(f"there are {len(self.agent_for_cp)} agent")
            for a, l_optimizer in zip(self.agent_name_list, self.learnable_optimizers):
                print(f"learnable_agent:{a},l_optimizer:{type(l_optimizer).__name__}")

        if len(self.traditional_optimizers) == 0:
            print("None of traditional optimizer")
        else:
            print(f"there are {len(self.traditional_optimizers)} traditional optimizer")
            for t_optmizer in self.traditional_optimizers:
                print(f"t_optmizer:{type(t_optmizer).__name__}")

    def test(self):
        print(f"start testing: {self.config.run_time}")
        # calculate T0
        T0 = calculate_time_0(self.config.dim, self.config.maxFEs)
        self.test_results["T0"] = T0
        pbar_len = (
            (len(self.traditional_optimizers) + len(self.agent_for_cp))
            * self.test_set.N
            * RUNS_NO
        )
        with tqdm(range(pbar_len), desc="Testing") as pbar:
            for i, problem in enumerate(self.test_set):
                # run learnable optimizer
                for agent_id, (agent, optimizer) in enumerate(
                    zip(self.agent_for_cp, self.learnable_optimizers)
                ):
                    T1, T2 = 0.0, 0.0
                    agent_name = self.agent_name_list[agent_id]
                    for run in range(RUNS_NO):
                        start = time.perf_counter()
                        np.random.seed(self.seed[run])
                        problem.reset()
                        # construct an ENV for (problem,optimizer)
                        environment = PBO_Env(problem, optimizer)
                        info = agent.rollout_episode(environment)
                        T1, T2 = self.report_test_episode(
                            agent_name, problem, info, pbar, start, T1, T2, i, run
                        )
                    if i == 0:
                        self.test_results["T1"][agent_name] = T1 / RUNS_NO
                        self.test_results["T2"][agent_name] = T2 / RUNS_NO
                # run traditional optimizer
                for optimizer in self.traditional_optimizers:
                    T1, T2 = 0.0, 0.0
                    optimizer_name = type(optimizer).__name__
                    for run in range(RUNS_NO):
                        start = time.perf_counter()
                        np.random.seed(self.seed[run])
                        problem.reset()
                        info = optimizer.run_episode(problem)
                        T1, T2 = self.report_test_episode(
                            optimizer_name, problem, info, pbar, start, T1, T2, i, run
                        )
                    if i == 0:
                        self.test_results["T1"][optimizer_name] = T1 / RUNS_NO
                        self.test_results["T2"][optimizer_name] = T2 / RUNS_NO
        with open(os.path.join(self.log_dir, "test.pkl"), "wb") as f:
            pickle.dump(default_to_regular(self.test_results), f, -1)
        random_search_results = test_for_random_search(self.config)
        with open(os.path.join(self.log_dir, "random_search_baseline.pkl"), "wb") as f:
            pickle.dump(default_to_regular(random_search_results), f, -1)

    def report_test_episode(
        self, optimizer_type: str, problem, info, pbar, start, T1, T2, i: int, run: int
    ):
        cost = info["cost"]
        cost.extend(
            [cost[-1] for _ in range(RUNS_NO - len(cost))]
        )  # extend cost to length of RUNS_NO
        fes = info["fes"]
        end = time.perf_counter()
        if i == 0:
            T1 += problem.T1
            T2 += (end - start) * 1000  # ms
        self.test_results["cost"][problem.__str__()][optimizer_type].append(cost)
        self.test_results["fes"][problem.__str__()][optimizer_type].append(fes)
        pbar_info = {
            "problem": problem.__str__(),
            "optimizer": optimizer_type,
            "run": run,
            "cost": cost[-1],
            "fes": fes,
        }
        pbar.set_postfix(pbar_info)
        pbar.update(1)
        return T1, T2


def rollout(config):
    print(f"start rollout: {config.run_time}")

    train_set, _ = construct_problem_set(config)

    agent_load_dir = config.agent_load_dir
    n_checkpoint = config.n_checkpoint

    train_rollout_results = {"cost": {}, "fes": {}, "return": {}}

    agent_for_rollout = config.agent_for_rollout

    load_agents = {}
    for agent_name in agent_for_rollout:
        load_agents[agent_name] = []
        for checkpoint in range(n_checkpoint + 1):
            file_path = os.path.join(
                agent_load_dir, agent_name, f"checkpoint{checkpoint}.pkl"
            )
            with open(file_path, "rb") as f:
                load_agents[agent_name].append(pickle.load(f))

    optimizer_for_rollout = [
        eval(optimizer_name)(copy.deepcopy(config))
        for optimizer_name in config.optimizer_for_rollout
    ]

    for problem in train_set:
        train_rollout_results["cost"][problem.__str__()] = defaultdict(
            lambda: [[] for _ in range(n_checkpoint + 1)]
        )
        train_rollout_results["fes"][problem.__str__()] = defaultdict(
            lambda: [[] for _ in range(n_checkpoint + 1)]
        )
        train_rollout_results["return"][problem.__str__()] = defaultdict(
            lambda: [[] for _ in range(n_checkpoint + 1)]
        )

    pbar_len = (len(agent_for_rollout)) * train_set.N * (n_checkpoint + 1) * 5
    with tqdm(range(pbar_len), desc="Rollouting") as pbar:
        for (agent_name, optimizer), checkpoint in product(
            zip(agent_for_rollout, optimizer_for_rollout), range(n_checkpoint + 1)
        ):
            agent = load_agents[agent_name][checkpoint]
            for problem, run in product(train_set, range(5)):
                np.random.seed(run)

                environment = PBO_Env(problem, optimizer)
                info = agent.rollout_episode(environment)
                cost = info["cost"]
                cost.extend(
                    [cost[-1] for _ in range(RUNS_NO - len(cost))]
                )  # extend cost to length of RUNS_NO
                fes = info["fes"]
                return_value = info["return"]
                for attribute, data in zip(
                    ("cost", "fes", "return"), (cost, fes, return_value)
                ):
                    train_rollout_results[attribute][problem.__str__()][agent_name][
                        checkpoint
                    ].append(data)

                pbar_info = {
                    "problem": problem.__str__(),
                    "agent": type(agent).__name__,
                    "checkpoint": checkpoint,
                    "run": run,
                    "cost": cost[-1],
                    "fes": fes,
                }
                pbar.set_postfix(pbar_info)
                pbar.update(1)

    log_dir = config.rollout_log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "rollout.pkl"), "wb") as f:
        regular_dict = default_to_regular(train_rollout_results)
        pickle.dump(regular_dict, f, -1)


def test_for_random_search(config):
    # get entire problem set
    train_set, test_set = construct_problem_set(config)
    entire_set = train_set + test_set
    # get optimizer
    optimizer = Random_search(copy.deepcopy(config))
    optimizer_type = type(optimizer).__name__
    # initialize the dataframe for logging
    test_results = {
        "cost": defaultdict(dict),
        "fes": defaultdict(dict),
        "T0": calculate_time_0(config.dim, config.maxFEs),  # calculate T0
        "T1": {optimizer_type: 0.0},
        "T2": {optimizer_type: 0.0},
    }

    for problem in entire_set:
        test_results["cost"][problem.__str__()][
            optimizer_type
        ] = []  # RUNS_NO np.arrays
        test_results["fes"][problem.__str__()][optimizer_type] = []  # RUNS_NO scalars

    # begin testing
    pbar_len = len(entire_set) * RUNS_NO
    with tqdm(range(pbar_len), desc="test for random search") as pbar:
        for i, problem in enumerate(entire_set):
            T1 = 0
            T2 = 0
            for run in range(RUNS_NO):
                start = time.perf_counter()
                np.random.seed(run)
                info = optimizer.run_episode(problem)
                cost = info["cost"]
                cost.extend(
                    [cost[-1] for _ in range(RUNS_NO - len(cost))]
                )  # extend cost to length of RUNS_NO
                fes = info["fes"]
                end = time.perf_counter()
                if i == 0:
                    T1 += problem.T1
                    T2 += (end - start) * 1000  # ms
                test_results["cost"][problem.__str__()][optimizer_type].append(cost)
                test_results["fes"][problem.__str__()][optimizer_type].append(fes)
                pbar_info = {
                    "problem": problem.__str__(),
                    "optimizer": optimizer_type,
                    "run": run,
                    "cost": cost[-1],
                    "fes": fes,
                }
                pbar.set_postfix(pbar_info)
                pbar.update(1)
            if i == 0:
                test_results["T1"][optimizer_type] = T1 / RUNS_NO
                test_results["T2"][optimizer_type] = T2 / RUNS_NO
    return test_results


def name_translate(problem: str) -> str:
    if problem == "bbob":
        return "Synthetic"
    elif problem == "bbob-noisy":
        return "Noisy-Synthetic"
    elif problem == "protein":
        return "Protein-Docking"
    else:
        raise ValueError(problem + " is not defined!")


def mgd_test(config):
    print(f"start MGD_test: {config.run_time}")
    # get test set
    _, test_set = construct_problem_set(config)
    # get agents
    with open(config.model_from, "rb") as f:
        agent_from = pickle.load(f)
    with open(config.model_to, "rb") as f:
        agent_to = pickle.load(f)
    # get optimizer
    l_optimizer = eval(config.optimizer)(copy.deepcopy(config))
    # initialize the dataframe for logging
    test_results = {"cost": {}, "fes": {}, "T0": 0.0, "T1": {}, "T2": {}}
    agent_name_list = [f"{config.agent}_from", f"{config.agent}_to"]
    for agent_name in agent_name_list:
        test_results["T1"][agent_name] = 0.0
        test_results["T2"][agent_name] = 0.0
    for problem in test_set:
        test_results["cost"][problem.__str__()] = {}
        test_results["fes"][problem.__str__()] = {}
        for agent_name in agent_name_list:
            test_results["cost"][problem.__str__()][agent_name] = []  # 51 np.arrays
            test_results["fes"][problem.__str__()][agent_name] = []  # 51 scalars
    # calculate T0
    test_results["T0"] = calculate_time_0(config.dim, config.maxFEs)
    # begin mgd_test
    seed = range(RUNS_NO)
    pbar_len = len(agent_name_list) * len(test_set) * RUNS_NO
    with tqdm(range(pbar_len), desc="MGD_Test") as pbar:
        for i, problem in enumerate(test_set):
            # run model_from and model_to
            for agent_id, agent in enumerate([agent_from, agent_to]):
                T1 = 0
                T2 = 0
                for run in range(RUNS_NO):
                    start = time.perf_counter()
                    np.random.seed(seed[run])
                    # construct an ENV for (problem,optimizer)
                    env = PBO_Env(problem, l_optimizer)
                    info = agent.rollout_episode(env)
                    cost = info["cost"]
                    cost.extend(
                        [cost[-1] for _ in range(RUNS_NO - len(cost))]
                    )  # extend cost to length of RUNS_NO
                    fes = info["fes"]
                    end = time.perf_counter()
                    if i == 0:
                        T1 += env.problem.T1
                        T2 += (end - start) * 1000  # ms
                    test_results["cost"][problem.__str__()][
                        agent_name_list[agent_id]
                    ].append(cost)
                    test_results["fes"][problem.__str__()][
                        agent_name_list[agent_id]
                    ].append(fes)
                    pbar_info = {
                        "problem": problem.__str__(),
                        "optimizer": agent_name_list[agent_id],
                        "run": run,
                        "cost": cost[-1],
                        "fes": fes,
                    }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
                if i == 0:
                    test_results["T1"][agent_name_list[agent_id]] = T1 / RUNS_NO
                    test_results["T2"][agent_name_list[agent_id]] = T2 / RUNS_NO
    if not os.path.exists(config.mgd_test_log_dir):
        os.makedirs(config.mgd_test_log_dir)
    with open(os.path.join(config.mgd_test_log_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_results, f, -1)
    random_search_results = test_for_random_search(config)
    with open(
        os.path.join(config.mgd_test_log_dir, "random_search_baseline.pkl"), "wb"
    ) as f:
        pickle.dump(default_to_regular(random_search_results), f, -1)
    logger = Logger(config)
    aei, aei_std = logger.aei_metric(test_results, random_search_results, config.maxFEs)
    print(f"AEI: {aei}")
    print(f"AEI STD: {aei_std}")
    print(
        f"MGD({name_translate(config.problem_from)}_{config.difficulty_from}, {name_translate(config.problem_to)}_{config.difficulty_to}) of {config.agent}: "
        f"{100 * (1 - aei[config.agent + '_from'] / aei[config.agent + '_to'])}%"
    )


# preprocess data for agent
def preprocess(file, agent):
    print(file, agent)
    with open(file, "rb") as f:
        data = pickle.load(f)
    # aggregate all problem's data together
    returns = data["return"]
    results = None
    for i, problem in enumerate(returns.keys()):
        if i == 0:
            results = np.array(returns[problem][agent])
        else:
            results = np.concatenate(
                [results, np.array(returns[problem][agent])], axis=1
            )
    return np.array(results)


def mte_test(config):
    print(f"start MTE_test: {config.run_time}")
    pre_train_file = config.pre_train_rollout
    scratch_file = config.scratch_rollout
    agent = config.agent
    bbob_data = preprocess(pre_train_file, agent)
    noisy_data = preprocess(scratch_file, agent)
    # calculate min_max avg
    checkpoints = np.hsplit(bbob_data, 18)
    checkpoints = np.array([checkpoint.tolist() for checkpoint in checkpoints])
    avg_bbob = bbob_data.mean(axis=-1)
    avg_bbob = savgol_filter(avg_bbob, 13, 5)
    std_bbob = np.mean(np.std(checkpoints, -1), 0) / np.sqrt(5)

    checkpoints = np.hsplit(noisy_data, 18)
    checkpoints = np.array([checkpoint.tolist() for checkpoint in checkpoints])

    std_noisy = np.mean(np.std(checkpoints, -1), 0) / np.sqrt(5)
    avg_noisy = noisy_data.mean(-1)
    avg_noisy = savgol_filter(avg_noisy, 13, 5)
    plt.figure(figsize=(40, 15))
    plt.subplot(1, 3, (2, 3))
    x = np.arange(21)
    x = (1.5e6 / x[-1]) * x
    idx = 21
    s = np.zeros(21)
    a = s[0] = avg_bbob[0]
    norm = 2
    for i in range(1, 21):
        a = a + avg_bbob[i]
        s[i] = a / norm
        norm += 1

    s_ = np.zeros(21)
    a = s_[0] = avg_noisy[0]
    norm = 2
    for i in range(1, 21):
        a = a + avg_noisy[i]
        s_[i] = a / norm
        norm += 1
    plt.plot(
        x[:idx],
        s[:idx],
        label="pre-train",
        marker="*",
        markersize=30,
        markevery=1,
        c="blue",
        linewidth=5,
    )
    plt.fill_between(
        x[:idx],
        s[:idx] - std_bbob[:idx],
        s[:idx] + std_bbob[:idx],
        alpha=0.2,
        facecolor="blue",
    )
    plt.plot(
        x[:idx],
        s_[:idx],
        label="scratch",
        marker="*",
        markersize=30,
        markevery=1,
        c="red",
        linewidth=5,
    )
    plt.fill_between(
        x[:idx],
        s_[:idx] - std_noisy[:idx],
        s_[:idx] + std_noisy[:idx],
        alpha=0.2,
        facecolor="red",
    )
    # Search MTE
    scratch = s_[:idx]
    pretrain = s[:idx]
    topx = np.argmax(scratch)
    topy = scratch[topx]
    T = topx / 21
    t = 0
    if pretrain[0] < topy:
        for i in range(1, 21):
            if pretrain[i - 1] < topy <= pretrain[i]:
                t = (
                    (topy - pretrain[i - 1]) / (pretrain[i] - pretrain[i - 1]) + i - 1
                ) / 21
                break
    if np.max(pretrain[-1]) < topy:
        t = 1
    MTE = 1 - t / T

    print(
        f"MTE({name_translate(config.problem_from)}_{config.difficulty_from}, {name_translate(config.problem_to)}_{config.difficulty_to}) of {config.agent}: "
        f"{MTE}"
    )

    ax = plt.gca()
    ax.xaxis.get_offset_text().set_fontsize(45)
    plt.xticks(
        fontsize=45,
    )
    plt.yticks(fontsize=45)
    plt.legend(loc=0, fontsize=60)
    plt.xlabel("Learning Steps", fontsize=55)
    plt.ylabel("Avg Return", fontsize=55)
    plt.title(
        f"Fine-tuning ({name_translate(config.problem_from)} $\\rightarrow$ {name_translate(config.problem_to)})",
        fontsize=60,
    )
    plt.tight_layout()
    plt.grid()
    plt.subplots_adjust(wspace=0.2)
    if not os.path.exists(config.mte_test_log_dir):
        os.makedirs(config.mte_test_log_dir)
    plt.savefig(
        os.path.join(config.mte_test_log_dir, f"MTE_{agent}.png"), bbox_inches="tight"
    )
