"""Active CMA-ES through the native Rust PyO3 extension.

Run this file directly for a compact example:

    .venv/bin/python examples/python/test_cma.py

It is also a pytest module, so the same code checks the one-shot and ask/tell
binding contracts:

    .venv/bin/python -m pytest examples/python/test_cma.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import _fcmaes_ext as fcmaes

try:
    from .testfun import ObjectiveMonitor, rosen
except ImportError:  # Direct execution instead of package/pytest import.
    from testfun import ObjectiveMonitor, rosen


DIM = 5
POPSIZE = 31
MAX_EVALUATIONS = 20_000
STOP_FITNESS = 1e-10
SEED = 42


@dataclass(frozen=True)
class CmaResult:
    x: NDArray[np.float64]
    fun: float
    nfev: int
    nit: int
    status: int

    @classmethod
    def from_tuple(cls, result: tuple) -> "CmaResult":
        x, fun, nfev, nit, status = result
        return cls(np.asarray(x), float(fun), int(nfev), int(nit), int(status))


def solve_one_shot() -> tuple[CmaResult, ObjectiveMonitor]:
    """Minimize Rosenbrock with the one-shot Rust CMA-ES binding."""

    problem = rosen(DIM)
    monitor = ObjectiveMonitor(problem.fun, DIM)
    result = fcmaes.optimize_acma(
        monitor,
        None,
        np.zeros(DIM, dtype=np.float64),
        problem.lower,
        problem.upper,
        np.array([0.3], dtype=np.float64),
        seed=SEED,
        max_evaluations=MAX_EVALUATIONS,
        stop_fitness=STOP_FITNESS,
        popsize=POPSIZE,
        workers=1,
    )
    return CmaResult.from_tuple(result), monitor


def solve_ask_tell() -> tuple[CmaResult, ObjectiveMonitor]:
    """Minimize Rosenbrock while Python evaluates each requested population."""

    problem = rosen(DIM)
    monitor = ObjectiveMonitor(problem.fun, DIM)
    optimizer = fcmaes.ACMA(
        np.zeros(DIM, dtype=np.float64),
        problem.lower,
        problem.upper,
        np.array([0.3], dtype=np.float64),
        seed=SEED,
        max_evaluations=MAX_EVALUATIONS,
        stop_fitness=STOP_FITNESS,
        popsize=POPSIZE,
    )
    while optimizer.stop == 0:
        xs = optimizer.ask()
        ys = np.asarray([monitor(x) for x in xs], dtype=np.float64)
        optimizer.tell(ys)
    return CmaResult.from_tuple(optimizer.result()), monitor


def assert_valid(result: CmaResult, monitor: ObjectiveMonitor) -> None:
    assert result.fun < STOP_FITNESS
    assert result.nfev == monitor.count
    assert result.nfev <= MAX_EVALUATIONS
    assert result.nit > 0
    assert result.status == 1
    np.testing.assert_allclose(result.x, monitor.best_x, rtol=0.0, atol=0.0)
    assert result.fun == monitor.best_y


def test_cma_one_shot() -> None:
    result, monitor = solve_one_shot()
    assert_valid(result, monitor)


def test_cma_ask_tell() -> None:
    result, monitor = solve_ask_tell()
    assert_valid(result, monitor)


def main() -> None:
    one_shot, _ = solve_one_shot()
    ask_tell, _ = solve_ask_tell()
    print(
        "one-shot:"
        f" fun={one_shot.fun:.3e} nfev={one_shot.nfev}"
        f" iterations={one_shot.nit} status={one_shot.status}"
    )
    print(
        "ask/tell:"
        f" fun={ask_tell.fun:.3e} nfev={ask_tell.nfev}"
        f" iterations={ask_tell.nit} status={ask_tell.status}"
    )


if __name__ == "__main__":
    main()
