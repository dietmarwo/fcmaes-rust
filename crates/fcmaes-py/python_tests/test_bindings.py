"""Integration tests for the installed ``fcmaes_rust`` distribution."""

import inspect
from importlib.metadata import version

import numpy as np
import pytest

import fcmaes_rust


ext = fcmaes_rust.native


EMPTY_FLOAT = np.empty(0, dtype=np.float64)
EMPTY_BOOL = np.empty(0, dtype=np.bool_)

PUBLIC_CLASSES = ("DE", "ACMA", "CRFMNES", "PGPE", "Bite", "MODE", "Archive")


def sphere(x):
    values = np.asarray(x, dtype=np.float64)
    return float(np.dot(values, values))


def test_public_api_has_runtime_documentation():
    """Every callable and descriptor shown by help() must explain itself."""

    assert inspect.getdoc(fcmaes_rust)
    assert inspect.getdoc(ext)
    assert inspect.getdoc(ext._phase1_probe_sum)

    missing = []
    for name in fcmaes_rust.__all__:
        value = getattr(fcmaes_rust, name)
        if callable(value) and not inspect.getdoc(value):
            missing.append(name)

    for class_name in PUBLIC_CLASSES:
        cls = getattr(fcmaes_rust, class_name)
        for name, value in vars(cls).items():
            if name.startswith("_"):
                continue
            if (callable(value) or inspect.isdatadescriptor(value)) and not inspect.getdoc(value):
                missing.append(f"{class_name}.{name}")

    assert not missing, f"public Python API without runtime docstrings: {missing}"


def test_module_metadata_and_numpy_probe():
    info = ext.phase1_build_info()
    assert info["module"] == "_fcmaes_ext"
    assert info["phase"] == 0
    assert info["backend"] == "rust"
    assert info["nanobind"] is False
    assert info["core_version"] == info["binding_version"]
    assert info["binding_version"] == version("fcmaes-rust")
    assert fcmaes_rust.__version__ == info["binding_version"]
    assert fcmaes_rust.phase1_build_info is ext.phase1_build_info
    assert fcmaes_rust.optimize_de is ext.optimize_de
    assert fcmaes_rust.DE is ext.DE
    assert "native" in fcmaes_rust.__all__

    values = np.ascontiguousarray([1.0, -2.0, 3.5], dtype=np.float64)
    assert ext._phase1_probe_sum(values) == pytest.approx(2.5)
    with pytest.raises(TypeError):
        ext._phase1_probe_sum(values.astype(np.float32))


def test_optimize_de_minimizes_sphere():
    dim = 4
    lower = np.full(dim, -5.0, dtype=np.float64)
    upper = np.full(dim, 5.0, dtype=np.float64)
    x, value, evaluations, iterations, stop = ext.optimize_de(
        sphere,
        dim,
        lower,
        upper,
        EMPTY_FLOAT,
        EMPTY_FLOAT,
        EMPTY_BOOL,
        seed=7,
        max_evaluations=4_000,
        popsize=20,
    )

    assert np.asarray(x).shape == (dim,)
    assert value == pytest.approx(sphere(x), rel=1e-12, abs=1e-14)
    assert value < 1e-8
    assert evaluations >= 4_000
    assert iterations > 0
    assert stop in (0, 1)


def test_bite_ask_tell_contract_and_validation():
    lower = np.array([-1.0, -1.0], dtype=np.float64)
    upper = np.array([1.0, 1.0], dtype=np.float64)
    bite = ext.Bite(
        EMPTY_FLOAT,
        lower,
        upper,
        1,
        4,
        3,
        seed=11,
        max_evaluations=5,
    )

    with pytest.raises(RuntimeError, match="before ask"):
        bite.tell(np.array([1.0], dtype=np.float64))

    xs = bite.ask()
    assert xs.shape == (3, 2)
    with pytest.raises(RuntimeError, match="twice"):
        bite.ask()
    with pytest.raises(ValueError, match="batch size"):
        bite.tell(np.array([1.0, 2.0], dtype=np.float64))
    bite.tell(np.asarray([sphere(x) for x in xs], dtype=np.float64))

    while True:
        xs = bite.ask()
        if len(xs) == 0:
            break
        bite.tell(np.asarray([sphere(x) for x in xs], dtype=np.float64))

    x, value, evaluations, iterations, _stop = bite.result()
    assert np.asarray(x).shape == (2,)
    assert np.isfinite(value)
    assert (evaluations, iterations) == (5, 5)

    with pytest.raises(ValueError):
        ext.Bite(
            EMPTY_FLOAT,
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            seed=1,
        )


def sample_optimizer(fun, bounds, guess, _sdev, rng, store):
    assert store.eval_num(3) == 3
    assert store.get_count_runs() >= 0
    x = rng.uniform(bounds.lb, bounds.ub) if guess is None else np.asarray(guess)
    return x, float(fun(x)), 3


def run_retry():
    return ext.minimize_retry(
        sphere,
        sample_optimizer,
        np.array([-2.0, -2.0], dtype=np.float64),
        np.array([2.0, 2.0], dtype=np.float64),
        num_retries=6,
        workers=1,
        capacity=6,
        max_evaluations=3,
        statistic_num=6,
        seed=123,
    )


def test_single_worker_retry_is_deterministic():
    first = run_retry()
    second = run_retry()

    assert first["success"]
    assert first["runs"] == 6
    assert first["nfev"] == 18
    assert first["xs"].shape == (6, 2)
    assert first["improvements"].shape[1] == 3
    assert first["fun"] == pytest.approx(sphere(first["x"]))
    np.testing.assert_array_equal(first["x"], second["x"])
    np.testing.assert_array_equal(first["xs"], second["xs"])
    np.testing.assert_array_equal(first["ys"], second["ys"])


def test_mode_ask_tell_shapes_and_errors():
    dim = 3
    popsize = 6
    lower = np.zeros(dim, dtype=np.float64)
    upper = np.ones(dim, dtype=np.float64)
    mode = ext.MODE(
        dim,
        2,
        0,
        lower,
        upper,
        EMPTY_BOOL,
        popsize=popsize,
        seed=17,
    )

    xs = mode.ask()
    assert xs.shape == (popsize, dim)
    with pytest.raises(ValueError):
        mode.ask()
    with pytest.raises(ValueError):
        mode.tell(np.zeros((popsize, 1), dtype=np.float64))

    ys = np.column_stack(
        (np.sum(xs * xs, axis=1), np.sum((xs - 1.0) ** 2, axis=1))
    )
    assert mode.tell(np.ascontiguousarray(ys)) == 0
    assert mode.population().shape == (popsize, dim)
    assert (mode.dim, mode.nobj, mode.ncon, mode.popsize) == (dim, 2, 0, popsize)

    with pytest.raises(ValueError):
        ext.MODE(2, 1, 0, lower, upper, EMPTY_BOOL, seed=1)


def qd_fitness(x):
    values = np.asarray(x, dtype=np.float64)
    return sphere(values), np.array([values[0], values[1]], dtype=np.float64)


def test_map_elites_archive_shapes_and_updates():
    archive = ext.Archive(
        2,
        np.array([-1.0, -1.0], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([-1.0, -1.0], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        capacity=16,
        samples_per_niche=0,
        seed=19,
    )

    archive.optimize_map_elites(qd_fitness, generations=4, chunk_size=4)
    assert (archive.dim, archive.qd_dim, archive.capacity) == (2, 2, 16)
    assert archive.xs().shape == (16, 2)
    assert archive.ys().shape == (16,)
    assert archive.descriptors().shape == (16, 2)
    assert archive.occupied > 0
    assert np.isfinite(archive.ys()).sum() == archive.occupied
    assert np.isfinite(archive.best_y)


ONE_SHOT_AND_STATEFUL = [
    ("optimize_acma", "ACMA"),
    ("optimize_bite", "Bite"),
    ("optimize_crfmnes", "CRFMNES"),
    ("optimize_de", "DE"),
    ("optimize_pgpe", "PGPE"),
]


def test_one_shot_and_stateful_optimizers_share_defaults():
    """A parameter must not mean one thing in the function and another in the class.

    ``PGPE`` once defaulted ``use_ranking`` to False while ``optimize_pgpe``
    and the Rust ``PgpeParams::default()`` both used True, so the same
    algorithm behaved differently depending on which entry point was called.
    """

    divergent = []
    for function_name, class_name in ONE_SHOT_AND_STATEFUL:
        function_parameters = inspect.signature(getattr(fcmaes_rust, function_name)).parameters
        class_parameters = inspect.signature(getattr(fcmaes_rust, class_name)).parameters
        for name, parameter in function_parameters.items():
            if parameter.default is inspect.Parameter.empty or name not in class_parameters:
                continue
            other = class_parameters[name].default
            if other is inspect.Parameter.empty or other == parameter.default:
                continue
            divergent.append(
                f"{name}: {function_name}={parameter.default!r} {class_name}={other!r}"
            )
    assert not divergent, "shared parameter defaults diverged:\n" + "\n".join(divergent)
