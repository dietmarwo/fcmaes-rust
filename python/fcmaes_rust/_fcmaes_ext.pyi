"""Type stubs for the native ``fcmaes_rust._fcmaes_ext`` extension.

The stubs are the authoritative signature reference for editors and type
checkers. They exist for a second reason too: PyO3 cannot render non-literal
Rust defaults in ``__text_signature__``, so ``inspect.signature`` reports
``Ellipsis`` for parameters whose real default is ``-inf`` or ``inf``
(``stop_fitness``, ``value_limit``) and for negative literals
(``stop_hist=-1.0``, ``update_gap=-1``). The defaults written here are the
values the Rust code actually uses.

Array parameters accept anything NumPy converts to a contiguous ``float64``
array; a non-convertible dtype raises ``TypeError``. Invalid dimensions,
bounds, population sizes and out-of-order ``ask``/``tell`` calls raise
``ValueError``.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

_F64 = NDArray[np.float64]
_ArrayLike = Sequence[float] | _F64

# (x, fun, evaluations, iterations, stop)
_Result = tuple[_F64, float, int, int, int]

def phase1_build_info() -> dict[str, Any]:
    """Return native-extension build information.

    Keys: ``module``, ``phase``, ``backend``, ``nanobind``, ``core_version``,
    ``binding_version``.
    """

def _phase1_probe_sum(values: _F64) -> float:
    """Installation probe. Not a numerical reduction API."""

# --------------------------------------------------------------------------
# One-shot optimizers
# --------------------------------------------------------------------------

def optimize_acma(
    fun: Callable[[_F64], float],
    batch_fun: Callable[[_F64], _ArrayLike] | None,
    guess: _ArrayLike,
    lower: _ArrayLike,
    upper: _ArrayLike,
    sigma: _ArrayLike,
    *,
    seed: int,
    runid: int = 0,
    max_evaluations: int = 100_000,
    stop_fitness: float = -np.inf,
    stop_hist: float = -1.0,
    mu: int = 0,
    popsize: int = 31,
    accuracy: float = 1.0,
    normalize: bool = True,
    delayed_update: bool = True,
    update_gap: int = -1,
    workers: int = 1,
) -> _Result: ...
def optimize_bite(
    fun: Callable[[_F64], float],
    guess: _ArrayLike,
    lower: _ArrayLike,
    upper: _ArrayLike,
    *,
    seed: int,
    runid: int = 0,
    max_evaluations: int = 100_000,
    stop_fitness: float = -np.inf,
    M: int = 1,
    popsize: int = 0,
    stall_criterion: int = 0,
) -> _Result: ...
def optimize_crfmnes(
    batch_fun: Callable[[_F64], _ArrayLike],
    guess: _ArrayLike,
    lower: _ArrayLike,
    upper: _ArrayLike,
    sigma: float = 0.3,
    *,
    seed: int,
    runid: int = 0,
    max_evaluations: int = 100_000,
    stop_fitness: float = -np.inf,
    popsize: int = 32,
    penalty_coef: float = 100_000.0,
    use_constraint_violation: bool = True,
    normalize: bool = False,
) -> _Result: ...
def optimize_da(
    fun: Callable[[_F64], float],
    guess: _ArrayLike,
    lower: _ArrayLike,
    upper: _ArrayLike,
    *,
    seed: int,
    runid: int = 0,
    max_evaluations: int = 100_000,
    use_local_search: bool = True,
) -> _Result: ...
def optimize_de(
    fun: Callable[[_F64], float],
    dim: int,
    lower: _ArrayLike,
    upper: _ArrayLike,
    guess: _ArrayLike,
    sigma: _ArrayLike,
    ints: Sequence[bool] | NDArray[np.bool_],
    *,
    seed: int,
    runid: int = 0,
    max_evaluations: int = 100_000,
    keep: float = 200.0,
    stop_fitness: float = -np.inf,
    popsize: int = 31,
    F: float = 0.5,
    CR: float = 0.9,
    min_sigma: float = 0.0,
    min_mutate: float = 0.1,
    max_mutate: float = 0.5,
    workers: int = 1,
    terminate: Callable[[], bool] | None = None,
) -> _Result: ...
def optimize_pgpe(
    batch_fun: Callable[[_F64], _ArrayLike],
    guess: _ArrayLike,
    lower: _ArrayLike,
    upper: _ArrayLike,
    sigma: _ArrayLike,
    *,
    seed: int,
    runid: int = 0,
    max_evaluations: int = 100_000,
    stop_fitness: float = -np.inf,
    popsize: int = 32,
    lr_decay_steps: int = 1000,
    use_ranking: bool = True,
    center_learning_rate: float = 0.15,
    stdev_learning_rate: float = 0.1,
    stdev_max_change: float = 0.2,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-08,
    decay_coef: float = 1.0,
    normalize: bool = True,
) -> _Result: ...

# --------------------------------------------------------------------------
# Retry drivers
# --------------------------------------------------------------------------

def minimize_retry(
    fun: Callable[[_F64], float],
    optimize: Callable[..., tuple[_F64, float, int]],
    lower: _ArrayLike,
    upper: _ArrayLike,
    num_retries: int = 1024,
    workers: int = 0,
    capacity: int = 500,
    value_limit: float = np.inf,
    stop_fitness: float = -np.inf,
    max_evaluations: int = 50_000,
    statistic_num: int = 0,
    seed: int = 0,
) -> dict[str, Any]: ...
def minimize_advanced_retry(
    fun: Callable[[_F64], float],
    optimize: Callable[..., tuple[_F64, float, int]],
    lower: _ArrayLike,
    upper: _ArrayLike,
    num_retries: int = 5000,
    workers: int = 0,
    capacity: int = 500,
    value_limit: float = np.inf,
    stop_fitness: float = -np.inf,
    min_evaluations: int = 1500,
    max_eval_fac: float = 50.0,
    check_interval: int = 100,
    statistic_num: int = 0,
    seed: int = 0,
) -> dict[str, Any]: ...
def minimize_moretry(
    fun: Callable[[_F64], _ArrayLike],
    optimize: Callable[..., tuple[_F64, float, int]],
    lower: _ArrayLike,
    upper: _ArrayLike,
    weight_lower: _ArrayLike,
    weight_upper: _ArrayLike,
    ncon: int = 0,
    value_exp: float = 2.0,
    value_limits: _ArrayLike | None = None,
    num_retries: int = 1024,
    workers: int = 0,
    capacity: int = 1024,
    value_limit: float = np.inf,
    stop_fitness: float = -np.inf,
    max_evaluations: int = 50_000,
    statistic_num: int = 0,
    seed: int = 0,
) -> dict[str, Any]: ...

# --------------------------------------------------------------------------
# Stateful ask/tell optimizers
# --------------------------------------------------------------------------

class ACMA:
    """Stateful active CMA-ES driven by external ``ask``/``tell`` calls."""

    def __init__(
        self,
        guess: _ArrayLike,
        lower: _ArrayLike,
        upper: _ArrayLike,
        sigma: _ArrayLike,
        *,
        max_evaluations: int = 100_000,
        stop_fitness: float = -np.inf,
        stop_hist: float = -1.0,
        mu: int = 0,
        popsize: int = 31,
        accuracy: float = 1.0,
        seed: int,
        runid: int = 0,
        normalize: bool = True,
        delayed_update: bool = True,
        update_gap: int = -1,
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def popsize(self) -> int: ...
    @property
    def stop(self) -> int: ...
    def ask(self) -> _F64: ...
    def population(self) -> _F64: ...
    def result(self) -> _Result: ...
    def tell(self, ys: _ArrayLike) -> int: ...
    def tell_x(self, ys: _ArrayLike, xs: _ArrayLike) -> int: ...

class Bite:
    """Stateful BiteOpt driven by external ``ask``/``tell`` calls."""

    def __init__(
        self,
        guess: _ArrayLike,
        lower: _ArrayLike,
        upper: _ArrayLike,
        M: int = 1,
        popsize: int = 0,
        batch_size: int = 8,
        *,
        max_evaluations: int = 100_000,
        stop_fitness: float = -np.inf,
        stall_criterion: int = 0,
        seed: int,
        runid: int = 0,
    ) -> None: ...
    @property
    def current_batch_size(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def popsize(self) -> int: ...
    @property
    def population_size(self) -> int: ...
    @property
    def stop(self) -> int: ...
    def ask(self) -> _F64: ...
    def result(self) -> _Result: ...
    def tell(self, ys: _ArrayLike) -> int: ...

class CRFMNES:
    """Stateful CR-FM-NES driven by external ``ask``/``tell`` calls."""

    def __init__(
        self,
        guess: _ArrayLike,
        lower: _ArrayLike,
        upper: _ArrayLike,
        sigma: float = 0.3,
        popsize: int = 32,
        *,
        seed: int,
        runid: int = 0,
        penalty_coef: float = 100_000.0,
        use_constraint_violation: bool = True,
        normalize: bool = False,
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def popsize(self) -> int: ...
    @property
    def stop(self) -> int: ...
    def ask(self) -> _F64: ...
    def population(self) -> _F64: ...
    def result(self) -> _Result: ...
    def tell(self, ys: _ArrayLike) -> int: ...

class DE:
    """Stateful Differential Evolution driven by external ``ask``/``tell``."""

    def __init__(
        self,
        dim: int,
        lower: _ArrayLike,
        upper: _ArrayLike,
        guess: _ArrayLike,
        sigma: _ArrayLike,
        ints: Sequence[bool] | NDArray[np.bool_],
        popsize: int = 31,
        keep: float = 200.0,
        F: float = 0.5,
        CR: float = 0.9,
        min_sigma: float = 0.0,
        min_mutate: float = 0.1,
        max_mutate: float = 0.5,
        *,
        seed: int,
        runid: int = 0,
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def popsize(self) -> int: ...
    @property
    def stop(self) -> int: ...
    def ask(self) -> _F64: ...
    def population(self) -> _F64: ...
    def result(self) -> _Result: ...
    def tell(self, ys: _ArrayLike) -> int: ...

class MODE:
    """Stateful multi-objective DE with NSGA-II style population update."""

    def __init__(
        self,
        dim: int,
        nobj: int,
        ncon: int,
        lower: _ArrayLike,
        upper: _ArrayLike,
        ints: Sequence[bool] | NDArray[np.bool_],
        popsize: int = 64,
        F: float = 0.5,
        CR: float = 0.9,
        pro_c: float = 0.5,
        dis_c: float = 15.0,
        pro_m: float = 0.9,
        dis_m: float = 20.0,
        nsga_update: bool = True,
        pareto_update: float = 0.0,
        min_mutate: float = 0.1,
        max_mutate: float = 0.5,
        *,
        seed: int,
        runid: int = 0,
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def ncon(self) -> int: ...
    @property
    def nobj(self) -> int: ...
    @property
    def popsize(self) -> int: ...
    @property
    def stop(self) -> int: ...
    def ask(self) -> _F64: ...
    def population(self) -> _F64: ...
    def set_population(self, xs: _ArrayLike, ys: _ArrayLike) -> int: ...
    def tell(self, ys: _ArrayLike) -> int: ...
    def tell_switch(
        self,
        ys: _ArrayLike,
        nsga_update: bool = True,
        pareto_update: float = 0.0,
    ) -> int: ...

class PGPE:
    """Stateful PGPE driven by external ``ask``/``tell`` calls."""

    def __init__(
        self,
        guess: _ArrayLike,
        lower: _ArrayLike,
        upper: _ArrayLike,
        sigma: _ArrayLike,
        popsize: int = 32,
        *,
        seed: int,
        runid: int = 0,
        lr_decay_steps: int = 1000,
        use_ranking: bool = True,
        center_learning_rate: float = 0.15,
        stdev_learning_rate: float = 0.1,
        stdev_max_change: float = 0.2,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-08,
        decay_coef: float = 1.0,
        normalize: bool = True,
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def popsize(self) -> int: ...
    @property
    def stop(self) -> int: ...
    def ask(self) -> _F64: ...
    def population(self) -> _F64: ...
    def result(self) -> _Result: ...
    def tell(self, ys: _ArrayLike) -> int: ...

# --------------------------------------------------------------------------
# Quality diversity
# --------------------------------------------------------------------------

class Archive:
    """CVT-MAP-Elites archive over a bounded behavior space."""

    def __init__(
        self,
        dim: int,
        lower: _ArrayLike,
        upper: _ArrayLike,
        qd_lower: _ArrayLike,
        qd_upper: _ArrayLike,
        capacity: int = 4000,
        samples_per_niche: int = 20,
        *,
        seed: int,
        seed_parents: bool = True,
    ) -> None: ...
    @property
    def best_y(self) -> float: ...
    @property
    def capacity(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def occupied(self) -> int: ...
    @property
    def qd_dim(self) -> int: ...
    @property
    def qd_score(self) -> float: ...
    def descriptors(self) -> _F64: ...
    def xs(self) -> _F64: ...
    def ys(self) -> _F64: ...
    def diversify(
        self,
        qd_fitness: Callable[[_F64], tuple[float, _ArrayLike]],
        max_evaluations: int = 100_000,
        popsize: int = 31,
        stall_criterion: int = 20,
    ) -> None: ...
    def optimize_map_elites(
        self,
        qd_fitness: Callable[[_F64], tuple[float, _ArrayLike]],
        generations: int = 100,
        chunk_size: int = 20,
        use_sbx: bool = True,
        dis_c: float = 20.0,
        dis_m: float = 20.0,
        iso_sigma: float = 0.02,
        line_sigma: float = 0.2,
        cma_generations: int = 0,
    ) -> None: ...

# --------------------------------------------------------------------------
# GTOP benchmark objectives
# --------------------------------------------------------------------------

def gtop_cassini1(x: _ArrayLike) -> float: ...
def gtop_cassini1_minlp(x: _ArrayLike) -> float: ...
def gtop_cassini2(x: _ArrayLike) -> float: ...
def gtop_cassini2_minlp(x: _ArrayLike) -> float: ...
def gtop_gtoc1(x: _ArrayLike) -> float: ...
def gtop_messenger(x: _ArrayLike) -> float: ...
def gtop_messengerfull(x: _ArrayLike) -> float: ...
def gtop_rosetta(x: _ArrayLike) -> float: ...
def gtop_sagas(x: _ArrayLike) -> float: ...
def gtop_tandem(x: _ArrayLike, sequence: Sequence[int]) -> float: ...
def gtop_tandem_unconstrained(x: _ArrayLike, sequence: Sequence[int]) -> float: ...
