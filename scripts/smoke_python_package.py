"""Smoke-test an installed fcmaes-rust wheel or source distribution."""

from __future__ import annotations

import os
import tempfile
from importlib.metadata import version

import numpy as np

import fcmaes_rust


def sphere(x: np.ndarray) -> float:
    return float(np.dot(x, x))


def main() -> None:
    # Ensure the test does not succeed by importing package sources through
    # the repository working directory.
    os.chdir(tempfile.mkdtemp(prefix="fcmaes-rust-smoke-"))

    installed_version = version("fcmaes-rust")
    expected_version = os.environ.get("EXPECTED_VERSION")
    if expected_version is not None:
        assert installed_version == expected_version
    assert fcmaes_rust.__version__ == installed_version

    info = fcmaes_rust.phase1_build_info()
    assert info["backend"] == "rust"
    assert info["core_version"] == installed_version
    assert info["binding_version"] == installed_version

    dim = 4
    empty_float = np.empty(0, dtype=np.float64)
    x, value, evaluations, iterations, stop = fcmaes_rust.optimize_de(
        sphere,
        dim,
        np.full(dim, -5.0, dtype=np.float64),
        np.full(dim, 5.0, dtype=np.float64),
        empty_float,
        empty_float,
        np.empty(0, dtype=np.bool_),
        seed=7,
        max_evaluations=4_000,
        popsize=20,
    )
    assert np.asarray(x).shape == (dim,)
    assert value < 1e-8
    assert evaluations >= 4_000
    assert iterations > 0
    assert stop in (0, 1)
    print(
        f"fcmaes-rust {installed_version}: value={value:.3e}, "
        f"evaluations={evaluations}, backend={info['backend']}"
    )


if __name__ == "__main__":
    main()
