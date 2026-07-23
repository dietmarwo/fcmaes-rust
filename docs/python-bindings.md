# Optional PyO3 bindings

The `fcmaes-py` crate is an optional low-level embedding surface for Python
packages. It builds `_fcmaes_ext`; no Python facade, SciPy result adapter, or
plotting package is bundled in this Rust-only repository. The repository-local
`pyproject.toml` keeps this extension package independent from any surrounding
Python checkout.

## Build and import

Using maturin in a virtual environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install "maturin[patchelf]>=1.7,<2"
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" \
  PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/maturin develop --release \
  --manifest-path crates/fcmaes-py/Cargo.toml
.venv/bin/python -c \
  'from _fcmaes_ext import phase1_build_info; print(phase1_build_info())'
```

The extension functions return low-level tuples or dictionaries. Downstream
packages can wrap these in their preferred public result types.

## Runnable Python example

[`examples/python/test_cma.py`](../examples/python/test_cma.py) adapts the
Rosenbrock tests from the original `fcmaes.testfun` and `fcmaes.test_cma`
modules. It demonstrates both the one-shot `optimize_acma` function and the
`ACMA` ask/tell class:

```bash
.venv/bin/python examples/python/test_cma.py
.venv/bin/python -m pytest examples/python/test_cma.py
```

The objective monitor verifies that the result returned by Rust matches the
best point observed by Python and that the evaluation counts agree.

## Optimizer surface

| Algorithm | One-shot function | Stateful class |
|---|---|---|
| Differential Evolution | `optimize_de` | `DE` |
| Active CMA-ES | `optimize_acma` | `ACMA` |
| CR-FM-NES | `optimize_crfmnes` | `CRFMNES` |
| PGPE | `optimize_pgpe` | `PGPE` |
| Dual Annealing | `optimize_da` | — |
| BiteOpt | `optimize_bite` | `Bite` |
| MODE | Ask/tell only | `MODE` |
| MAP-Elites / Diversifier | Archive methods | `Archive` |

Stateful scalar classes expose `ask`, `tell`, `population`, and `result` where
the underlying optimizer supports them. ACMA also exposes `tell_x`. BiteOpt
enforces pending-batch call order and exact feedback lengths. The authoritative
callable signatures are the `#[pyo3(signature = ...)]` declarations under
`crates/fcmaes-py/src/`.

## MODE and quality diversity

`MODE.ask()` returns a `(popsize, dim)` matrix. `tell()` accepts a
`(popsize, nobj + ncon)` matrix in which minimized objectives precede
constraints; constraints are feasible at values less than or equal to zero.
`tell_switch()` changes the update mode for one batch, and `set_population()`
installs validated decision and objective matrices.

The native QD `Archive` constructor receives decision bounds, descriptor
bounds, capacity, CVT sampling density, and a seed. For two descriptors,
`samples_per_niche=0` selects constant-time grid lookup; positive values build
CVT centers.

`Archive.optimize_map_elites()` runs the SBX/mutation or Iso+LineDD emitter and
optional CMA emitter generations. `Archive.diversify()` runs the CMA-ME-style
improvement search. `xs()`, `ys()`, and `descriptors()` expose archive arrays;
`occupied`, `best_y`, and `qd_score` expose summary values.

Persistence, archive joins, shared-memory statistics, and plotting are not
part of this binding crate.

## Retry surface

The extension exports `minimize_retry`, `minimize_advanced_retry`, and
`minimize_moretry`. Returned dictionaries contain:

- `x`, `fun`, `nfev`, `nit`, and `success`;
- `retry_xs` and `retry_ys` for retained results;
- `improvements` for completed-retry best-value samples.

The advanced-retry optimizer callback receives local bounds, an optional
guess, step-size information, its evaluation budget, and an independently
spawned seed. The moretry callback receives sampled scalarization weights and
retains the original vector-valued evaluations.

## GTOP surface

The extension exposes:

- `gtop_gtoc1`
- `gtop_cassini1` and `gtop_cassini1_minlp`
- `gtop_cassini2` and `gtop_cassini2_minlp`
- `gtop_messenger` and `gtop_messengerfull`
- `gtop_rosetta`
- `gtop_sagas`
- `gtop_tandem` and `gtop_tandem_unconstrained`

Wrong input dimensions return the GTOP penalty result instead of accessing
outside native arrays.

## GIL and parallelism

Optimizer loops execute under `py.allow_threads`. Each Python objective call
constructs a NumPy vector and reacquires the GIL. Consequently:

- Native Rust objectives scale across retry workers without the GIL.
- Python objectives dominated by an extension may scale if that extension
  releases the GIL.
- Cheap Python callbacks are normally callback/GIL limited.
- Combining retry-level workers with population-level workers can
  oversubscribe the machine and should be measured.
