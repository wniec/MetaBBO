import numpy as np
from typing import Union


def generate_random_individuals(
    population_size: int, cols: int, followed_individual: int
) -> np.ndarray:
    generated_population = np.random.randint(low=0, high=population_size, size=cols)
    while followed_individual in generated_population:
        generated_population = np.random.randint(low=0, high=population_size, size=cols)
    return generated_population


def generate_mutation_matrix(population_size: int, cols: int) -> np.ndarray:
    """
    Generate a matrix of random integers used for mutation.

    Each row contains 'cols' distinct integers in the range [0, population_size - 1],
    excluding the row index itself.

    :param population_size: Number of individuals in the population.
    :param cols: Number of random integers generated per individual.
    :return: A (population_size x cols) matrix with required constraints.
    """
    mutation_matrix = np.random.randint(
        low=0, high=population_size, size=(population_size, cols)
    )

    for col in range(cols):
        while True:
            # Check for duplicates in previous columns and identity with row index
            conflicts = np.zeros(population_size, dtype=bool)
            for prev_col in range(col):
                conflicts |= mutation_matrix[:, col] == mutation_matrix[:, prev_col]
            conflicts |= mutation_matrix[:, col] == np.arange(population_size)

            if not np.any(conflicts):
                break

            mutation_matrix[conflicts, col] = np.random.randint(
                0, population_size, size=np.sum(conflicts)
            )
    return mutation_matrix


def rand_1_individual(
    population: np.ndarray,
    mutation_factor: float,
    pointer: int,
    individuals: np.ndarray = None,
) -> np.ndarray:
    if individuals is None:
        individuals = generate_random_individuals(population.shape[0], 3, pointer)
    return population[individuals[0]] + mutation_factor * (
        population[individuals[1]] - population[individuals[2]]
    )


def rand_1(
    population: np.ndarray, mutation_factor: Union[np.ndarray, float]
) -> np.ndarray:
    """
    :param population: The 2-D population matrix of shape [NP, dim].
    :param mutation_factor: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(mutation_factor, np.ndarray) and mutation_factor.ndim == 1:
        mutation_factor = mutation_factor.reshape(-1, 1)
    mutation_matrix = generate_mutation_matrix(population.shape[0], 3)
    return population[mutation_matrix[:, 0]] + mutation_factor * (
        population[mutation_matrix[:, 1]] - population[mutation_matrix[:, 2]]
    )


def rand_2_single(
    population: np.ndarray,
    mutation_factor: float,
    pointer: int,
    individuals: np.ndarray = None,
) -> np.ndarray:
    if individuals is None:
        individuals = generate_random_individuals(population.shape[0], 5, pointer)
    return population[individuals[0]] + mutation_factor * (
        population[individuals[1]]
        - population[individuals[2]]
        + population[individuals[3]]
        - population[individuals[4]]
    )


def rand_2(
    population: np.ndarray, mutation_factor: Union[np.ndarray, float]
) -> np.ndarray:
    """
    :param population: The 2-D population matrix of shape [NP, dim].
    :param mutation_factor: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(mutation_factor, np.ndarray) and mutation_factor.ndim == 1:
        mutation_factor = mutation_factor.reshape(-1, 1)
    individuals = generate_mutation_matrix(population.shape[0], 5)
    return population[individuals[:, 0]] + mutation_factor * (
        population[individuals[:, 1]]
        - population[individuals[:, 2]]
        + population[individuals[:, 3]]
        - population[individuals[:, 4]]
    )


def best_1_single(
    population: np.ndarray,
    best: np.ndarray,
    mutation_factor: float,
    pointer: int,
    individuals: np.ndarray = None,
) -> np.ndarray:
    if individuals is None:
        individuals = generate_random_individuals(population.shape[0], 2, pointer)
    return best + mutation_factor * (
        population[individuals[0]] - population[individuals[1]]
    )


def best_1(
    population: np.ndarray, best: np.ndarray, mutation_factor: Union[np.ndarray, float]
) -> np.ndarray:
    """
    :param population: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param mutation_factor: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(mutation_factor, np.ndarray) and mutation_factor.ndim == 1:
        mutation_factor = mutation_factor.reshape(-1, 1)
    r = generate_mutation_matrix(population.shape[0], 2)
    return best + mutation_factor * (population[r[:, 0]] - population[r[:, 1]])


def best_2_single(
    population: np.ndarray,
    best: np.ndarray,
    mutation_factor: float,
    pointer: int,
    individuals: np.ndarray = None,
) -> np.ndarray:
    if individuals is None:
        individuals = generate_random_individuals(population.shape[0], 4, pointer)
    return best + mutation_factor * (
        population[individuals[0]]
        - population[individuals[1]]
        + population[individuals[2]]
        - population[individuals[3]]
    )


def best_2(
    population: np.ndarray, best: np.ndarray, mutation_factor: Union[np.ndarray, float]
) -> np.ndarray:
    """
    :param population: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param mutation_factor: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(mutation_factor, np.ndarray) and mutation_factor.ndim == 1:
        mutation_factor = mutation_factor.reshape(-1, 1)
    r = generate_mutation_matrix(population.shape[0], 4)
    return best + mutation_factor * (
        population[r[:, 0]]
        - population[r[:, 1]]
        + population[r[:, 2]]
        - population[r[:, 3]]
    )


def rand_to_best_1_single(
    population: np.ndarray,
    best: np.ndarray,
    mutation_factor: float,
    pointer: int,
    r: np.ndarray = None,
) -> np.ndarray:
    if r is None:
        r = generate_random_individuals(population.shape[0], 3, pointer)
    return population[r[0]] + mutation_factor * (
        best - population[r[0]] + population[r[1]] - population[r[2]]
    )


def rand_to_best_1(
    population: np.ndarray, best: np.ndarray, mutation_factor: Union[np.ndarray, float]
) -> np.ndarray:
    """
    :param population: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param mutation_factor: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(mutation_factor, np.ndarray) and mutation_factor.ndim == 1:
        mutation_factor = mutation_factor.reshape(-1, 1)
    individuals = generate_mutation_matrix(population.shape[0], 3)
    return population[individuals[:, 0]] + mutation_factor * (
        best
        - population[individuals[:, 0]]
        + population[individuals[:, 1]]
        - population[individuals[:, 2]]
    )


def rand_to_best_2_single(
    population: np.ndarray,
    best: np.ndarray,
    mutation_factor: float,
    pointer: int,
    individuals: np.ndarray = None,
) -> np.ndarray:
    if individuals is None:
        individuals = generate_random_individuals(population.shape[0], 5, pointer)
    return population[individuals[0]] + mutation_factor * (
        best
        - population[individuals[0]]
        + population[individuals[1]]
        - population[individuals[2]]
        + population[individuals[3]]
        - population[individuals[4]]
    )


def rand_to_best_2(
    population: np.ndarray, best: np.ndarray, mutation_factor: Union[np.ndarray, float]
) -> np.ndarray:
    """
    :param population: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param mutation_factor: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(mutation_factor, np.ndarray) and mutation_factor.ndim == 1:
        mutation_factor = mutation_factor.reshape(-1, 1)
    individuals = generate_mutation_matrix(population.shape[0], 5)
    return population[individuals[:, 0]] + mutation_factor * (
        best
        - population[individuals[:, 0]]
        + population[individuals[:, 1]]
        - population[individuals[:, 2]]
        + population[individuals[:, 3]]
        - population[individuals[:, 4]]
    )


def cur_to_best_1_single(
    population: np.ndarray,
    best: np.ndarray,
    mutation_factor: float,
    pointer: int,
    individuals: np.ndarray = None,
) -> np.ndarray:
    if individuals is None:
        individuals = generate_random_individuals(population.shape[0], 2, pointer)
    return population[pointer] + mutation_factor * (
        best
        - population[pointer]
        + population[individuals[0]]
        - population[individuals[1]]
    )


def cur_to_best_1(
    population: np.ndarray, best: np.ndarray, mutation_factor: Union[np.ndarray, float]
) -> np.ndarray:
    """
    :param population: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param mutation_factor: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(mutation_factor, np.ndarray) and mutation_factor.ndim == 1:
        mutation_factor = mutation_factor.reshape(-1, 1)
    r = generate_mutation_matrix(population.shape[0], 2)
    return population + mutation_factor * (
        best - population + population[r[:, 0]] - population[r[:, 1]]
    )


def cur_to_best_2_single(
    population: np.ndarray,
    best: np.ndarray,
    mutation_factor: float,
    pointer: int,
    individuals: np.ndarray = None,
) -> np.ndarray:
    if individuals is None:
        individuals = generate_random_individuals(population.shape[0], 4, pointer)
    return population[pointer] + mutation_factor * (
        best
        - population[pointer]
        + population[individuals[0]]
        - population[individuals[1]]
        + population[individuals[2]]
        - population[individuals[3]]
    )


def cur_to_best_2(
    population: np.ndarray, best: np.ndarray, mutation_factor: Union[np.ndarray, float]
) -> np.ndarray:
    """
    :param population: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param mutation_factor: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(mutation_factor, np.ndarray) and mutation_factor.ndim == 1:
        mutation_factor = mutation_factor.reshape(-1, 1)
    r = generate_mutation_matrix(population.shape[0], 4)
    return population + mutation_factor * (
        best
        - population
        + population[r[:, 0]]
        - population[r[:, 1]]
        + population[r[:, 2]]
        - population[r[:, 3]]
    )


def cur_to_rand_1_single(
    x: np.ndarray, mutation_factor: float, pointer: int, r: np.ndarray = None
) -> np.ndarray:
    if r is None:
        r = generate_random_individuals(x.shape[0], 3, pointer)
    return x[pointer] + mutation_factor * (x[r[0]] - x[pointer] + x[r[1]] - x[r[2]])


def cur_to_rand_1(
    population: np.ndarray, mutation_factor: Union[np.ndarray, float]
) -> np.ndarray:
    if isinstance(mutation_factor, np.ndarray) and mutation_factor.ndim == 1:
        mutation_factor = mutation_factor.reshape(-1, 1)
    r = generate_mutation_matrix(population.shape[0], 3)
    return population + mutation_factor * (
        population[r[:, 0]] - population + population[r[:, 1]] - population[r[:, 2]]
    )


def cur_to_rand_2_single(
    x: np.ndarray, mutation_factor: float, pointer: int, r: np.ndarray = None
) -> np.ndarray:
    if r is None:
        r = generate_random_individuals(x.shape[0], 5, pointer)
    return x[pointer] + mutation_factor * (
        x[r[0]] - x[pointer] + x[r[1]] - x[r[2]] + x[r[3]] - x[r[4]]
    )


def cur_to_rand_2(
    x: np.ndarray, mutation_factor: Union[np.ndarray, float]
) -> np.ndarray:
    if isinstance(mutation_factor, np.ndarray) and mutation_factor.ndim == 1:
        mutation_factor = mutation_factor.reshape(-1, 1)
    r = generate_mutation_matrix(x.shape[0], 5)
    return x + mutation_factor * (
        x[r[:, 0]] - x + x[r[:, 1]] - x[r[:, 2]] - x[r[:, 3]] + x[r[:, 4]]
    )
