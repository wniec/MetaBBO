import numpy as np
from typing import Union
import copy


def binomial(
    population_1: np.ndarray,
    population_2: np.ndarray,
    crossover_rate: Union[np.ndarray, float],
) -> np.ndarray:
    if population_1.ndim == 1:
        population_1 = population_1.reshape(1, -1)
        population_2 = population_2.reshape(1, -1)
    population_size, dim = population_1.shape
    random_dimension_idx = np.random.randint(dim, size=population_size)
    if isinstance(crossover_rate, np.ndarray) and crossover_rate.ndim == 1:
        crossover_rate = crossover_rate.reshape(-1, 1)
    new_population = np.where(
        np.random.rand(population_size, dim) < crossover_rate,
        population_2,
        population_1,
    )
    new_population[np.arange(population_size), random_dimension_idx] = population_2[
        np.arange(population_size), random_dimension_idx
    ]
    if new_population.shape[0] == 1:
        new_population = new_population.squeeze(axis=0)
    return new_population


def exponential(
    population1: np.ndarray,
    population2: np.ndarray,
    crossover_rate: Union[np.ndarray, float],
) -> np.ndarray:
    """
    Perform exponential crossover between two populations.

    :param population1: Original individuals, shape (population_size, dimension) or (dimension,)
    :param population2: Mutated individuals (donors), same shape as parents
    :param crossover_rate: Scalar or (population_size,) array of crossover probabilities
    :return: Offspring after crossover, same shape as parents
    """
    if population1.ndim == 1:
        population1 = population1[np.newaxis, :]
        population2 = population2[np.newaxis, :]

    population_size, dimension = population1.shape
    offspring = population1.copy()

    # Randomly select starting index for crossover for each individual
    start_indices = np.random.randint(0, dimension, size=(population_size, 1))

    # Random values to decide whether crossover happens at each gene
    random_values = np.random.rand(population_size, dimension)

    # Ensure crossover_rate is shaped (population_size, 1)
    if isinstance(crossover_rate, np.ndarray) and crossover_rate.ndim == 1:
        crossover_rate = crossover_rate[:, np.newaxis]
    elif isinstance(crossover_rate, float):
        crossover_rate = np.full((population_size, 1), crossover_rate)

    # Where crossover is allowed
    allowed_mask = random_values <= crossover_rate

    # Build a mask that defines a contiguous block of crossover for each individual
    crossover_mask = np.zeros_like(allowed_mask, dtype=bool)

    for i in range(population_size):
        start = start_indices[i, 0]
        length = 0
        while length < dimension and allowed_mask[i, (start + length) % dimension]:
            crossover_mask[i, (start + length) % dimension] = True
            length += 1

    # Apply crossover from donors to offspring where allowed
    offspring[crossover_mask] = population2[crossover_mask]

    return offspring.squeeze(axis=0) if offspring.shape[0] == 1 else offspring
