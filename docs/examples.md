# Native Rust examples

## Binaries

The `fcmaes-examples` crate provides:

| Binary | Execution model | Optimizer or purpose |
|---|---|---|
| `gtop-examples` | Independent basic retry | 40% DE, then 60% active CMA-ES |
| `gtop-advexamples` | Coordinated advanced retry | 40% DE, then 60% active CMA-ES |
| `benchmark-gtop` | Repeated coordinated-retry experiments | GTOP tutorial-style statistics and raw results |
| `benchmark-biteopt-gtop` | Repeated basic-retry experiments | BiteOpt or DE→CMA statistics and raw results |
| `mazda-mo` | MODE | Constrained mass/common-parts Pareto search |
| `mazda-qd` | CVT-MAP-Elites, optional Diversifier | Mazda behavior archive |
| `trading` | MODE and CVT-MAP-Elites | Four-stock EMA/SMA strategy search |
| `material-flow-planning` | BiteOpt | Native 24-hour factory simulation and throughput measurement |
| `jobshop` | BiteOpt | Flexible job-shop objective; optional Brandimarte `.fjs` input |
| `harvesting` | BiteOpt | Job-shop with bounded machine deployment windows |
| `t-design` | BiteOpt | Weighted spherical t-design using native harmonics |
| `scheduling` | BiteOpt | Dyson-ring transfer scheduler; optional text/XZ input |
| `damp` | BiteOpt | Controlled spring with exact segment propagation |
| `f8` | BiteOpt | F-8 bang-bang aircraft control with native DOPRI5 |
| `lotka` | BiteOpt | Lotka-Volterra fox-control problem with native DOPRI5 |

Cargo uses these names because `examples` is also a reserved Cargo target
category.

## Native objective ports

Seven application families have native Rust objective implementations.
Every binary accepts `--evals`, `--batch`, and `--seed`; the defaults are
20,000, 16, and 42. These small commands are useful smoke tests:

```bash
cargo run --release -p fcmaes-examples --bin jobshop -- --evals 2000
cargo run --release -p fcmaes-examples --bin harvesting -- --evals 2000 2
cargo run --release -p fcmaes-examples --bin t-design -- --evals 2000 10 4
cargo run --release -p fcmaes-examples --bin scheduling -- --evals 2000
cargo run --release -p fcmaes-examples --bin damp -- --evals 2000 12
cargo run --release -p fcmaes-examples --bin f8 -- --evals 2000 6
cargo run --release -p fcmaes-examples --bin lotka -- --evals 2000
```

`jobshop` and `harvesting` use a small embedded flexible-shop instance when no
dataset is given. Supply any Brandimarte-format instance with `--data`; the
harvesting positional argument is the maximum number of active machines:

```bash
cargo run --release -p fcmaes-examples --bin jobshop -- \
  --data /path/to/BrandimarteMk1.fjs
cargo run --release -p fcmaes-examples --bin harvesting -- \
  --data /path/to/BrandimarteMk1.fjs 4
```

`scheduling` similarly embeds a deterministic transfer fixture. Its loader
accepts the original whitespace format in either plain text or XZ form:

```bash
cargo run --release -p fcmaes-examples --bin scheduling -- \
  --data /path/to/tsin3000.60.xz
```

The scheduling module exposes the shaped scalar objective, true score,
two-objective vector, and MAP-Elites descriptor ingredients. Job-shop exposes
all three objectives, while harvesting adds the failure constraint. Other Rust
programs can therefore use MODE or MAP-Elites directly even though these
compact CLI drivers use scalar BiteOpt.

The numerical choices are intentional. `t-design` evaluates associated
Legendre functions with a stable recurrence including SciPy's
Condon-Shortley phase. F-8 and Lotka-Volterra use an allocation-free adaptive
Dormand-Prince 5(4) integrator. The controlled spring has an exact solution on
each constant-control segment, which is faster and more accurate than
numerically integrating the same linear equation.

Unit tests compare fixed objective vectors against the original NumPy/SciPy
implementations. Loader tests are hermetic; release smoke tests also cover the
published Brandimarte Mk1 and both `tsin3000.10.xz` and `tsin3000.60.xz`
datasets when those optional files are present.

## Mazda factory design

The Mazda benchmark has 222 discrete variables, five raw objectives, and 54
constraints. The Rust adapter loads the published `fitness_MazdaMop_C` ABI
from `libmazda.so`; decision decoding, constraint sign conversion, MODE, QD
scalarization, archive management, and Pareto selection are Rust. The
generated response-surface library and its decision table are external data
and are not included in this repository. Supply both paths explicitly.

Run the constrained MO example:

```bash
cargo run --release -p fcmaes-examples --bin mazda-mo -- \
  --library /path/to/libmazda.so --decisions /path/to/mazda.py \
  --evaluations 500000 --popsize 768 --seed 42
```

Add `--de-update` to use MODE's DE population update; the default is NSGA-II.
Progress reports the feasible offspring count, and final output lists up to 30
feasible Pareto points as mass and common-parts count.

Run CVT-MAP-Elites with the published descriptors and penalty:

```bash
cargo run --release -p fcmaes-examples --bin mazda-qd -- \
  --library /path/to/libmazda.so --decisions /path/to/mazda.py \
  --capacity 10000 --samples-per-niche 0 \
  --generations 10000 --chunk-size 100 --seed 42
```

Use `--iso-line` for the Iso+LineDD emitter or append
`--diversify-evaluations 1000000` for a CMA-based Diversifier phase. The
two-dimensional `samples-per-niche=0` path uses a constant-time rectangular
niche index and avoids CVT setup. Set it to 20 to reproduce the Python sample's
k-means CVT; setup then grows quadratically with archive capacity. Use a
smaller archive for a CVT smoke test:

```bash
cargo run --release -p fcmaes-examples --bin mazda-qd -- \
  --library /path/to/libmazda.so --decisions /path/to/mazda.py \
  --capacity 64 --samples-per-niche 4 --generations 10 --chunk-size 16
```

## Trading strategy

The `trading` binary optimizes an EMA/SMA crossing strategy over NVDA, GOOGL,
AAPL, and MSFT. Defaults use adjusted daily closes from 2020 through 2025. A
matching snapshot is included under `ticker_cache/`, allowing a deterministic
run without network access:

```bash
cargo run --release -p fcmaes-examples --bin trading -- --offline
```

The command performs two searches:

- MODE minimizes one negative return factor per stock, subject to at most 12
  signal-triggered trades per stock. `MO quality` is the best geometric mean
  of the four relative returns among feasible Pareto points.
- MAP-Elites minimizes the negative geometric mean return while using the four
  per-stock return factors as behavior descriptors. `QD quality` is the sum of
  archive returns divided by total archive capacity, so empty niches contribute
  zero and both quality and coverage matter.

Returns are relative to buying and holding the same stock over the period. The
program prints machine-readable `CONFIG`, `MO`, `QD`, and `RESULT` lines so
runs can be compared across compiler versions and optimizer settings.

Useful controls include `--mo-evaluations`, `--qd-evaluations`, `--popsize`,
`--capacity`, `--chunk-size`, `--samples-per-niche`, `--seed`, and `--tickers`.
Pass `--offline` to prohibit downloads when a cache is missing.

## Material-flow-planning speed example

The material-flow objective simulates two machines, setup delays, and an
eight-part FIFO for every second of a 24-hour production day. Its deliberately
fine-grained 86,400-tick implementation is entirely native Rust. Run the
objective workload and BiteOpt search with:

```bash
cargo run --release -p fcmaes-examples --bin material-flow-planning
```

The binary first evaluates a deterministic 256-point input sequence, then uses
the native BiteOpt implementation with a fixed seed, ask/tell batch size, and
256-evaluation budget. The fixed-workload checksum and optimizer result make
semantic drift immediately visible. `OBJECTIVE` and `OPTIMIZE` lines report
separate wall times and throughput. Override the workloads with
`--benchmark-evaluations` and `--optimize-evaluations`.

## Problem catalog

| CLI name | Display name | Dimension |
|---|---|---:|
| `cassini1` | Cassini1 | 6 |
| `cassini2` | Cassini2 | 22 |
| `rosetta` | Rosetta | 22 |
| `tandem` | Tandem 6 | 18 |
| `messenger` | Messenger reduced | 18 |
| `gtoc1` | GTOC1 | 8 |
| `messenger-full` | Messenger full | 26 |
| `sagas` | Sagas | 12 |
| `cassini1-minlp` | Cassini1 MINLP with fixed sequence | 6 |

Problem matching ignores case and separators. Messenger Full also accepts
`messenger_full`, `Messenger Full`, and `messfull`. Omit `--problem`, or pass
`--problem all`, to run the complete catalog.

## CLI options

Both binaries share the same parser:

| Option | Default | Meaning |
|---|---:|---|
| `--problem NAME` | all | Select one problem |
| `--retries N` | 32 | Retry count per selected problem |
| `--evaluations N` | 50,000 | Initial per-retry DE→CMA budget |
| `--workers N` | 0 | Retry workers; zero uses available parallelism |
| `--seed N` | 0 | Root retry seed |
| `--value-limit N` | infinity | Retain only results below this value |
| `--stop-fitness N` | negative infinity | Stop after reaching this objective value |
| `--progress-interval N` | 0 | Live status period in seconds; zero disables |
| `--max-eval-fac N` | 50 | Advanced final budget factor |
| `--check-interval N` | 100 | Advanced diversity checkpoint interval |

Show the installed parser help with:

```bash
cargo run --release -p fcmaes-examples --bin gtop-advexamples -- --help
```

## Basic examples

Run all problems with 16 workers:

```bash
cargo run --release -p fcmaes-examples --bin gtop-examples -- \
  --retries 32 --evaluations 50000 --workers 16 --seed 1
```

Run only Rosetta:

```bash
cargo run --release -p fcmaes-examples --bin gtop-examples -- \
  --problem rosetta --retries 32 --evaluations 50000 --workers 16 --seed 1
```

## Messenger Full

Run the long coordinated-retry Messenger Full preset with:

```bash
cargo run --release -p fcmaes-examples --bin gtop-advexamples -- \
  --problem messenger-full \
  --retries 50000 \
  --evaluations 1500 \
  --workers 16 \
  --seed 1 \
  --value-limit 12 \
  --max-eval-fac 50 \
  --check-interval 100 \
  --progress-interval 10
```

This is a large run. `max_eval_fac=50` linearly raises the per-retry budget
from 1,500 to 75,000 evaluations across the claimed retry IDs.

## Live progress

`--progress-interval` enables an atomic objective counter and a separate
reporting thread. It writes status to stderr and leaves the final result on
stdout:

```text
progress problem="Messenger full" final=false elapsed=10.0s \
evaluations=... evals_per_second=... retries=.../50000 best=...
```

The fields are elapsed wall time, objective evaluations observed by the runner,
evaluation throughput, completed retry count, and best objective value seen.
When all workers are inside their first long retry, evaluations and best value
can advance while completed retries remains zero.

Save both progress and the result:

```bash
cargo run --release -p fcmaes-examples --bin gtop-advexamples -- \
  --problem messenger-full --retries 50000 --evaluations 1500 \
  --workers 16 --value-limit 12 --progress-interval 10 \
  2>&1 | tee messenger-full.log
```

The monitor performs one relaxed atomic increment per evaluation and a
compare/exchange only when a candidate may improve the global best. It does
not place a mutex around objective evaluation.

## Output

The final line for each problem contains:

```text
Messenger full: value=... evaluations=... runs=... x=[...]
```

With a restrictive value limit, `value=inf` means no completed retry below the
limit was retained. The progress monitor can still show the best objective
observed so far.

## Benchmarks

`benchmark-gtop` runs independent coordinated-retry experiments and emits both
raw samples and an AsciiDoc summary table. Its default problem set excludes
the long-running Tandem and Messenger Full cases:

```bash
cargo run --release -p fcmaes-examples --bin benchmark-gtop -- \
  --runs 100 --workers 32 --seed 1 \
  --raw-output benchmarks/benchmark_gtop_100_raw.tsv \
  --table-output benchmarks/benchmark_gtop_100.adoc
```

Pass `--include-slow` to include both slow cases, or `--problem NAME` to run a
single case, including a slow one. The recorded 100-run measurement and its
methodology are in [the benchmark report](../benchmarks/benchmark_gtop.md).

`benchmark-biteopt-gtop` uses basic retry, includes Tandem, and excludes only
Messenger Full. Its defaults are 100 experiments, 24 workers, 24 retries, and
10,000 evaluations per retry. Run BiteOpt with:

```bash
cargo run --release -p fcmaes-examples --bin benchmark-biteopt-gtop -- \
  --algo biteopt --runs 100 --workers 24 --retries 24 \
  --evaluations 10000 --seed 1 \
  --raw-output benchmarks/benchmark_biteopt_gtop_rust_100_raw.tsv \
  --table-output benchmarks/benchmark_biteopt_gtop_rust_100.adoc
```

Use `--algo de_cma` for the two-stage optimizer with a fixed 4,000/6,000
DE/CMA evaluation split:

```bash
cargo run --release -p fcmaes-examples --bin benchmark-biteopt-gtop -- \
  --algo de_cma --runs 100 --workers 24 --retries 24 \
  --evaluations 10000 --seed 1 \
  --raw-output benchmarks/benchmark_de_cma_gtop_rust_100_raw.tsv \
  --table-output benchmarks/benchmark_de_cma_gtop_rust_100.adoc
```

Both tables report success rate, wall-time statistics, and the mean and
population standard deviation of the final optimum across all runs. See the
[native benchmark index](../benchmarks/README.md) for recorded output and
reproduction notes.
