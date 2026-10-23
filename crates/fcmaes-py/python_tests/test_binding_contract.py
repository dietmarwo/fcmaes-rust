"""The published Python contract: array inputs, evaluation counts, failures.

fcmaes-rust 0.2.0 typed its array parameters as ``Sequence[float] | NDArray``
(``Sequence[bool] | NDArray[np.bool_]`` for ``ints``) but the binding accepted
only NumPy arrays of the exact dtype, so a Python list raised ``TypeError``.
These tests pin the corrected behaviour: sequences give the same results as
the equivalent ``float64``/``bool`` arrays.

They also pin what ``evaluations`` means (objective calls actually made, which
population optimizers can exceed ``max_evaluations`` by a bounded amount) and
that a raising objective is a failed evaluation, not an aborted run.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import fcmaes_rust

DIM = 3
LOWER = [-2.0, -1.0, 0.0]
UPPER = (2.0, 3.0, 4.0)
GUESS = [0.5, 0.5, 1.0]
SIGMA = [0.3, 0.3, 0.3]


def shifted_sphere(x):
    values = np.asarray(x, dtype=np.float64)
    return float(np.sum((values - 0.25) ** 2))


def batch_sphere(xs):
    values = np.asarray(xs, dtype=np.float64)
    return np.sum((values - 0.25) ** 2, axis=1)


def as_arrays(*values):
    return [np.asarray(v, dtype=np.float64) for v in values]


def same_result(a, b):
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]


ONE_SHOT = {
    "optimize_de": lambda lo, up, guess, sigma, ints: fcmaes_rust.optimize_de(
        shifted_sphere, DIM, lo, up, guess, sigma, ints, seed=3, max_evaluations=600
    ),
    "optimize_acma": lambda lo, up, guess, sigma, ints: fcmaes_rust.optimize_acma(
        shifted_sphere, None, guess, lo, up, sigma, seed=3, max_evaluations=600
    ),
    "optimize_bite": lambda lo, up, guess, sigma, ints: fcmaes_rust.optimize_bite(
        shifted_sphere, guess, lo, up, seed=3, max_evaluations=600
    ),
    "optimize_crfmnes": lambda lo, up, guess, sigma, ints: fcmaes_rust.optimize_crfmnes(
        batch_sphere, guess, lo, up, 0.3, seed=3, max_evaluations=640
    ),
    "optimize_da": lambda lo, up, guess, sigma, ints: fcmaes_rust.optimize_da(
        shifted_sphere, guess, lo, up, seed=3, max_evaluations=600
    ),
    "optimize_pgpe": lambda lo, up, guess, sigma, ints: fcmaes_rust.optimize_pgpe(
        batch_sphere, guess, lo, up, sigma, seed=3, max_evaluations=640
    ),
}


@pytest.mark.parametrize("name", sorted(ONE_SHOT))
def test_one_shot_functions_accept_sequences(name):
    run = ONE_SHOT[name]
    reference = run(*as_arrays(LOWER, UPPER, GUESS, SIGMA), np.zeros(DIM, dtype=np.bool_))
    from_lists = run(LOWER, UPPER, GUESS, SIGMA, [False] * DIM)
    from_tuples = run(tuple(LOWER), tuple(UPPER), tuple(GUESS), tuple(SIGMA), (False,) * DIM)
    same_result(reference, from_lists)
    same_result(reference, from_tuples)


def test_de_integer_mask_accepts_bool_and_zero_one_sequences():
    lower, upper = [-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]

    def run(ints):
        return fcmaes_rust.optimize_de(
            shifted_sphere, DIM, lower, upper, [], [], ints, seed=5, max_evaluations=800
        )

    reference = run(np.array([False, True, False]))
    for ints in ([False, True, False], (False, True, False), [0, 1, 0], np.array([0, 1, 0])):
        same_result(reference, run(ints))
    # The mask changes the search, so it is really read.
    assert not np.array_equal(run([False, False, False])[0], reference[0])


def test_integer_mask_length_is_still_validated():
    with pytest.raises(ValueError):
        fcmaes_rust.optimize_de(
            shifted_sphere, DIM, LOWER, UPPER, [], [], [True], seed=1, max_evaluations=100
        )


def test_conversion_errors_are_typed():
    with pytest.raises(TypeError):
        fcmaes_rust.optimize_bite(shifted_sphere, GUESS, [LOWER], UPPER, seed=1)
    with pytest.raises(ValueError):
        fcmaes_rust.optimize_bite(shifted_sphere, GUESS, ["a", "b", "c"], UPPER, seed=1)


def drive(optimizer, rounds=3):
    for _ in range(rounds):
        xs = optimizer.ask()
        optimizer.tell([shifted_sphere(x) for x in xs])
    return optimizer.population()


def test_ask_tell_classes_accept_sequences():
    for make in (
        lambda lo, up, g, s: fcmaes_rust.DE(DIM, lo, up, g, s, [False, True, False], popsize=8, seed=2),
        lambda lo, up, g, s: fcmaes_rust.ACMA(g, lo, up, s, popsize=8, seed=2),
        lambda lo, up, g, s: fcmaes_rust.CRFMNES(g, lo, up, 0.3, 8, seed=2),
        lambda lo, up, g, s: fcmaes_rust.PGPE(g, lo, up, s, 8, seed=2),
    ):
        reference = drive(make(*as_arrays(LOWER, UPPER, GUESS, SIGMA)))
        np.testing.assert_array_equal(reference, drive(make(LOWER, UPPER, GUESS, SIGMA)))

    bite = fcmaes_rust.Bite(GUESS, LOWER, UPPER, seed=2)
    xs = bite.ask()
    assert bite.tell([shifted_sphere(x) for x in xs]) == 0


def test_acma_tell_x_accepts_nested_sequences():
    def run(convert):
        acma = fcmaes_rust.ACMA(GUESS, LOWER, UPPER, SIGMA, popsize=8, seed=4)
        for _ in range(3):
            xs = acma.ask()
            acma.tell_x(convert([shifted_sphere(x) for x in xs]), convert(xs.tolist()))
        return acma.population()

    np.testing.assert_array_equal(run(np.asarray), run(lambda v: v))


def test_mode_accepts_sequences_and_nested_sequences():
    def objectives(x):
        return [float(x[0] ** 2 + x[1] ** 2), float((x[0] - 1) ** 2 + x[1] ** 2)]

    def run(convert):
        mode = fcmaes_rust.MODE(
            2, 2, 0, convert([-2.0, -2.0]), convert([2.0, 2.0]), [False, False], popsize=8, seed=6
        )
        for _ in range(3):
            xs = mode.ask()
            mode.tell(convert([objectives(x) for x in xs]))
        population = mode.population()
        mode.set_population(convert(population.tolist()), convert([objectives(x) for x in population]))
        return mode.population()

    np.testing.assert_array_equal(run(np.asarray), run(lambda v: v))


def test_archive_and_retry_drivers_accept_sequences():
    archive = fcmaes_rust.Archive(DIM, LOWER, UPPER, [0.0, 0.0], [1.0, 1.0], 16, seed=1)
    archive.optimize_map_elites(
        lambda x: (shifted_sphere(x), [abs(x[0]) / 2, abs(x[1]) / 3]), generations=2, chunk_size=4
    )
    assert np.isfinite(archive.ys()).any()

    def optimize(fun, bounds, guess, sdev, rng, store):
        x, y, evaluations, _, _ = fcmaes_rust.optimize_bite(
            fun, (bounds.lb + bounds.ub) / 2, bounds.lb, bounds.ub,
            seed=int(rng.integers(1 << 31)), max_evaluations=200,
        )
        return x, y, evaluations

    result = fcmaes_rust.minimize_retry(shifted_sphere, optimize, LOWER, UPPER, num_retries=2, workers=1)
    assert result["fun"] < 1.0

    def two_objectives(x):
        return np.array([shifted_sphere(x), float(np.sum(x))])

    result = fcmaes_rust.minimize_moretry(
        two_objectives, optimize, LOWER, UPPER, [0.0, 0.0], [1.0, 1.0],
        value_limits=[100.0, 100.0], num_retries=2, workers=1,
    )
    assert result["nfev"] > 0


def test_gtop_functions_accept_sequences():
    x = [-789.8, 158.3, 449.4, 54.7, 1024.4, 4552.3]
    assert fcmaes_rust.gtop_cassini1(x) == fcmaes_rust.gtop_cassini1(np.asarray(x))
    value, launch = fcmaes_rust.gtop_cassini1_minlp(x + [2, 2, 3, 5])
    assert np.isfinite(value) and np.isfinite(launch)


class CountingObjective:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, x):
        with self.lock:
            self.calls += 1
        return shifted_sphere(x) + 10.0  # never reaches a stop_fitness

    def batch(self, xs):
        with self.lock:
            self.calls += len(xs)
        return batch_sphere(xs) + 10.0


BUDGETED = {
    # name: (run, maximum overshoot of max_evaluations)
    "optimize_de": (
        lambda f, budget: fcmaes_rust.optimize_de(
            f, DIM, LOWER, UPPER, [], [], [], seed=9, max_evaluations=budget, popsize=11
        ),
        2 * 11 - 1,
    ),
    "optimize_acma": (
        lambda f, budget: fcmaes_rust.optimize_acma(
            f, None, GUESS, LOWER, UPPER, SIGMA, seed=9, max_evaluations=budget, popsize=11
        ),
        11 - 1,
    ),
    "optimize_crfmnes": (
        lambda f, budget: fcmaes_rust.optimize_crfmnes(
            f.batch, GUESS, LOWER, UPPER, 0.3, seed=9, max_evaluations=budget, popsize=12
        ),
        12 - 1,
    ),
    "optimize_pgpe": (
        lambda f, budget: fcmaes_rust.optimize_pgpe(
            f.batch, GUESS, LOWER, UPPER, SIGMA, seed=9, max_evaluations=budget, popsize=12
        ),
        12 - 1,
    ),
    "optimize_bite": (
        lambda f, budget: fcmaes_rust.optimize_bite(f, GUESS, LOWER, UPPER, seed=9, max_evaluations=budget),
        0,
    ),
    "optimize_da": (
        lambda f, budget: fcmaes_rust.optimize_da(f, GUESS, LOWER, UPPER, seed=9, max_evaluations=budget),
        0,
    ),
}


@pytest.mark.parametrize("name", sorted(BUDGETED))
def test_reported_evaluations_are_actual_calls_within_documented_overshoot(name):
    run, overshoot = BUDGETED[name]
    worst = 0
    for budget in range(150, 1200, 53):
        objective = CountingObjective()
        result = run(objective, budget)
        assert result[2] == objective.calls, "evaluations must count actual calls"
        assert result[2] <= budget + overshoot
        worst = max(worst, result[2] - budget)
    if overshoot:
        # The overshoot is real, so requested evaluations are not actual work.
        assert worst > 0


def test_raising_objective_is_a_failed_evaluation():
    def fails_on_the_left(x):
        if x[0] < 0:
            raise RuntimeError("simulator failed")
        return shifted_sphere(x)

    x, y, evaluations, _, _ = fcmaes_rust.optimize_de(
        fails_on_the_left, DIM, LOWER, UPPER, [], [], [], seed=2, max_evaluations=1500
    )
    assert x[0] >= 0 and y < 0.1 and evaluations >= 1500


@pytest.mark.parametrize("value", [1e300, float("inf"), float("nan")])
def test_dual_annealing_without_start_point_reports_failure(value):
    objective = CountingObjective()

    def rejected(x):
        objective(x)
        return value

    x, y, evaluations, _, stop = fcmaes_rust.optimize_da(
        rejected, GUESS, LOWER, UPPER, seed=1, max_evaluations=300
    )
    assert stop == -1
    assert evaluations == objective.calls <= 300
    assert np.all(np.asarray(LOWER) <= x) and np.all(x <= np.asarray(UPPER))
    assert y == (np.finfo(np.float64).max if np.isnan(value) else value)


def _invalid_inputs():
    f, empty, no_ints = shifted_sphere, [], []
    good_lo, good_up = [-1.0, -1.0], [1.0, 1.0]
    boxes = {
        "unequal bounds": ([-1.0, -1.0], [1.0]),
        "lower >= upper": ([1.0, -1.0], [-1.0, 1.0]),
        "non-finite bound": ([-np.inf, -1.0], [1.0, 1.0]),
    }
    for label, (lo, up) in boxes.items():
        yield f"optimize_de {label}", lambda lo=lo, up=up: fcmaes_rust.optimize_de(
            f, 2, lo, up, empty, empty, no_ints, seed=1, max_evaluations=50)
        yield f"DE {label}", lambda lo=lo, up=up: fcmaes_rust.DE(2, lo, up, empty, empty, no_ints, seed=1)
        yield f"optimize_acma {label}", lambda lo=lo, up=up: fcmaes_rust.optimize_acma(
            f, None, [0.0, 0.0], lo, up, [0.3, 0.3], seed=1, max_evaluations=50)
        yield f"ACMA {label}", lambda lo=lo, up=up: fcmaes_rust.ACMA([0.0, 0.0], lo, up, [0.3, 0.3], seed=1)
        yield f"optimize_crfmnes {label}", lambda lo=lo, up=up: fcmaes_rust.optimize_crfmnes(
            batch_sphere, [0.0, 0.0], lo, up, 0.3, seed=1, max_evaluations=64)
        yield f"CRFMNES {label}", lambda lo=lo, up=up: fcmaes_rust.CRFMNES([0.0, 0.0], lo, up, 0.3, 8, seed=1)
        yield f"optimize_pgpe {label}", lambda lo=lo, up=up: fcmaes_rust.optimize_pgpe(
            batch_sphere, [0.0, 0.0], lo, up, [0.3, 0.3], seed=1, max_evaluations=64)
        yield f"PGPE {label}", lambda lo=lo, up=up: fcmaes_rust.PGPE([0.0, 0.0], lo, up, [0.3, 0.3], 8, seed=1)
        yield f"optimize_da {label}", lambda lo=lo, up=up: fcmaes_rust.optimize_da(
            f, [0.0, 0.0], lo, up, seed=1, max_evaluations=50)
    yield "optimize_de without bounds or guess", lambda: fcmaes_rust.optimize_de(
        f, 2, empty, empty, empty, empty, no_ints, seed=1, max_evaluations=50)
    yield "optimize_de guess without sigma", lambda: fcmaes_rust.optimize_de(
        f, 2, good_lo, good_up, [0.0, 0.0], empty, no_ints, seed=1, max_evaluations=50)
    yield "DE short ints", lambda: fcmaes_rust.DE(2, good_lo, good_up, empty, empty, [True], seed=1)
    yield "optimize_acma short guess", lambda: fcmaes_rust.optimize_acma(
        f, None, [0.0], good_lo, good_up, [0.3], seed=1, max_evaluations=50)
    yield "optimize_acma empty sigma", lambda: fcmaes_rust.optimize_acma(
        f, None, [0.0, 0.0], good_lo, good_up, empty, seed=1, max_evaluations=50)
    yield "optimize_pgpe short guess", lambda: fcmaes_rust.optimize_pgpe(
        batch_sphere, [0.0], good_lo, good_up, [0.3], seed=1, max_evaluations=64)
    yield "optimize_da empty guess", lambda: fcmaes_rust.optimize_da(
        f, empty, empty, empty, seed=1, max_evaluations=50)


INVALID = dict(_invalid_inputs())


@pytest.mark.parametrize("label", sorted(INVALID))
def test_invalid_inputs_raise_value_error_instead_of_panicking(label):
    # 0.2.0 raised pyo3's PanicException (a BaseException) for most of these.
    with pytest.raises(ValueError):
        INVALID[label]()


def test_unbounded_search_with_empty_bounds_still_works():
    start = [0.0, 0.0]
    assert fcmaes_rust.optimize_acma(shifted_sphere, None, start, [], [], [0.3], seed=1, max_evaluations=600)[1] < 1e-3
    assert fcmaes_rust.optimize_crfmnes(batch_sphere, start, [], [], 0.3, seed=1, max_evaluations=640)[1] < 1e-3
    assert fcmaes_rust.optimize_pgpe(batch_sphere, start, [], [], [0.3], seed=1, max_evaluations=2000)[1] < 1e-2
    assert fcmaes_rust.optimize_da(shifted_sphere, start, [], [], seed=1, max_evaluations=600)[1] < 1e-3
    result = fcmaes_rust.optimize_de(
        shifted_sphere, 2, [], [], start, [0.3, 0.3], [], seed=1, max_evaluations=600
    )
    assert result[1] < 1e-3
