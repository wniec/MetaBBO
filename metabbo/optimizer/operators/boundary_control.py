import numpy as np
from typing import Union, Iterable


def clipping(
    x: Union[np.ndarray, Iterable],
    lower_boundary: Union[np.ndarray, Iterable, int, float, None],
    upper_boundary: Union[np.ndarray, Iterable, int, float, None],
) -> np.ndarray:
    return np.clip(x, lower_boundary, upper_boundary)


def random(
    x: Union[np.ndarray, Iterable],
    lower_bound: Union[np.ndarray, Iterable, int, float],
    upper_bound: Union[np.ndarray, Iterable, int, float],
) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(upper_bound, np.ndarray):
        upper_bound = np.array(upper_bound)
    outside_boundary = (x < lower_bound) or (x > upper_bound)
    return int(outside_boundary) * x + outside_boundary * (
        np.random.rand(*x.shape) * (upper_bound - lower_bound) + lower_bound
    )


def reflection(
    x: Union[np.ndarray, Iterable],
    lower_bound: Union[np.ndarray, Iterable, int, float],
    upper_bound: Union[np.ndarray, Iterable, int, float],
) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    under_lower_bound = x < lower_bound
    over_upper_bound = x > upper_bound
    in_bounds = not (under_lower_bound or over_upper_bound)
    return (
        in_bounds * x
        + under_lower_bound * (2 * lower_bound - x)
        + over_upper_bound * (2 * upper_bound - x)
    )


def periodic(
    x: Union[np.ndarray, Iterable],
    lower_bound: Union[np.ndarray, Iterable, int, float],
    upper_bound: Union[np.ndarray, Iterable, int, float],
) -> np.ndarray:
    if not isinstance(upper_bound, np.ndarray):
        upper_bound = np.array(upper_bound)
    return (x - upper_bound) % (upper_bound - lower_bound) + lower_bound


def halving(
    x: Union[np.ndarray, Iterable],
    lower_bound: Union[np.ndarray, Iterable, int, float],
    upper_bound: Union[np.ndarray, Iterable, int, float],
) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    under_lower_bound = x < lower_bound
    over_upper_bound = x > upper_bound
    in_bounds = not (under_lower_bound or over_upper_bound)
    return (
        in_bounds * x
        + under_lower_bound * (x + lower_bound) / 2
        + over_upper_bound * (x + upper_bound) / 2
    )


def parent(
    x: Union[np.ndarray, Iterable],
    lower_bound: Union[np.ndarray, Iterable, int, float],
    upper_bound: Union[np.ndarray, Iterable, int, float],
    parent: Union[np.ndarray, Iterable],
) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(parent, np.ndarray):
        parent = np.array(parent)
    under_lower_bound = x < lower_bound
    over_upper_bound = x > upper_bound
    in_bounds = not (under_lower_bound or over_upper_bound)
    return (
        in_bounds * x
        + under_lower_bound * (parent + lower_bound) / 2
        + over_upper_bound * (parent + upper_bound) / 2
    )
