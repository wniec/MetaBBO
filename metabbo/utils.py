from collections import defaultdict

from torch.utils.data import Dataset
from problem import bbob, protein_docking


def construct_problem_set(config) -> tuple[Dataset, Dataset]:
    problem = config.problem
    if problem in ["bbob", "bbob-noisy"]:
        return bbob.BBOBDataset.get_datasets(
            suit=config.problem,
            dim=config.dim,
            upper_bound=config.upperbound,
            train_batch_size=1,
            test_batch_size=1,
            difficulty=config.difficulty,
        )
    elif problem == "protein":
        return protein_docking.Protein_Docking_Dataset.get_datasets(
            version=problem,
            train_batch_size=1,
            test_batch_size=1,
            difficulty=config.difficulty,
        )
    else:
        raise ValueError(problem + " is not defined!")


def default_to_regular(dictionary: dict | defaultdict):
    if isinstance(dictionary, defaultdict) or isinstance(dictionary, dict):
        dictionary = {k: default_to_regular(v) for k, v in dictionary.items()}
    return dictionary
