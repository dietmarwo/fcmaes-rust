"""Small optimization test functions adapted from ``fcmaes.testfun``.

The original module also contains multiprocessing-aware monitoring and SciPy
``Bounds`` objects.  This example keeps only the pieces needed to demonstrate
the low-level PyO3 API and has no dependency beyond NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


Objective = Callable[[ArrayLike], float]


@dataclass(frozen=True)
class TestFunction:
    """An objective together with its box bounds."""

    name: str
    fun: Objective
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


class ObjectiveMonitor:
    """Thread-safe evaluation count and best-observation monitor."""

    def __init__(self, fun: Objective, dim: int) -> None:
        self._fun = fun
        self._lock = Lock()
        self._best_x = np.zeros(dim, dtype=np.float64)
        self._best_y = float("inf")
        self._count = 0

    def __call__(self, x: ArrayLike) -> float:
        point = np.asarray(x, dtype=np.float64)
        value = float(self._fun(point))
        with self._lock:
            self._count += 1
            if value < self._best_y:
                self._best_y = value
                self._best_x = point.copy()
        return value

    @property
    def best_x(self) -> NDArray[np.float64]:
        with self._lock:
            return self._best_x.copy()

    @property
    def best_y(self) -> float:
        with self._lock:
            return self._best_y

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


def rosenbrock(x: ArrayLike) -> float:
    """Rosenbrock's function, with its minimum at the all-ones vector."""

    values = np.asarray(x, dtype=np.float64)
    return float(
        np.sum(
            100.0 * (values[:-1] * values[:-1] - values[1:]) ** 2
            + (1.0 - values[:-1]) ** 2
        )
    )


def rastrigin(x: ArrayLike) -> float:
    """Rastrigin's function, with its global minimum at the origin."""

    values = np.asarray(x, dtype=np.float64)
    return float(
        10.0 * values.size
        + np.sum(values * values - 10.0 * np.cos(2.0 * np.pi * values))
    )


def rosen(dim: int) -> TestFunction:
    if dim < 2:
        raise ValueError("Rosenbrock requires at least two dimensions")
    return TestFunction(
        "rosen",
        rosenbrock,
        np.full(dim, -5.0, dtype=np.float64),
        np.full(dim, 5.0, dtype=np.float64),
    )


def rastrigin_problem(dim: int) -> TestFunction:
    if dim == 0:
        raise ValueError("Rastrigin requires at least one dimension")
    return TestFunction(
        "rastrigin",
        rastrigin,
        np.full(dim, -5.12, dtype=np.float64),
        np.full(dim, 5.12, dtype=np.float64),
    )
