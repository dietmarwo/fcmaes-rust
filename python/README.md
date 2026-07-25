# fcmaes-rust for Python

`fcmaes-rust` provides Python access to fast, parallel, gradient-free
optimization algorithms implemented in native Rust. The optimizer core is
pure Rust and does not compile, link, or load the historical C++ backend.

## Installation

```bash
python -m pip install fcmaes-rust
```

The initial release supports CPython 3.11 through 3.13 on the platforms for
which wheels are listed on PyPI.

## Quick check

```python
import fcmaes_rust

print(fcmaes_rust.__version__)
print(fcmaes_rust.phase1_build_info())
```

## Optimizers

The package exposes one-shot and ask/tell interfaces for Differential
Evolution, active CMA-ES, CR-FM-NES, PGPE and BiteOpt, plus Dual Annealing,
MODE, MAP-Elites, the Diversifier, independent retry, coordinated retry and
weighted multi-objective retry.

The functions currently return low-level NumPy arrays, tuples or dictionaries.
See the [Python binding guide](https://dietmarwo.github.io/fcmaes-rust/docs/python-bindings.html)
for signatures, examples, parallelism behavior and result layouts.
Every exported function, stateful method, and property also provides an
installed runtime docstring, for example `help(fcmaes_rust.optimize_de)` or
`help(fcmaes_rust.MODE)`.
SciPy is a runtime dependency because retry callbacks receive a public
`scipy.optimize.Bounds` object.

```python
import numpy as np
import fcmaes_rust

def sphere(x):
    return float(np.dot(x, x))

dim = 4
empty = np.empty(0, dtype=np.float64)
x, value, evaluations, iterations, stop = fcmaes_rust.optimize_de(
    sphere,
    dim,
    np.full(dim, -5.0),
    np.full(dim, 5.0),
    empty,
    empty,
    np.empty(0, dtype=np.bool_),
    seed=7,
    max_evaluations=4_000,
    popsize=20,
)
print(value, evaluations, x)
```

Python callbacks must reacquire the GIL. Native Rust objectives offer the
strongest parallel scaling; cheap pure-Python callbacks are usually
callback-bound.

## Project links

- [Source repository](https://github.com/dietmarwo/fcmaes-rust)
- [Documentation](https://dietmarwo.github.io/fcmaes-rust/)
- [Issue tracker](https://github.com/dietmarwo/fcmaes-rust/issues)
- [Release history](https://github.com/dietmarwo/fcmaes-rust/releases)

## License

MIT
