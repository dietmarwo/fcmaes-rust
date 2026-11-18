# Candidate optimizers: benchmark harness

This workspace runs the experiment declared in [PROTOCOL.md](PROTOCOL.md):
which gradient-free algorithms missing from `fcmaes-core` should be added to
it. `PROTOCOL.md` is binding; this file describes how the harness implements
it.

| Crate or file | Content |
|---|---|
| `solver-api` | `Solver` trait, `Outcome`, `Guard` (budget and box enforcement) |
| `cand-de` | L-SHADE, jSO, L-SRTDE candidates |
| `cand-eda` | AMaLGaM-IDEA candidate |
| `harness` (`candidate-harness`) | incumbent adapters, BIPOP-aCMA-ES, problems, campaign runner |
| `analyze.py` | statistics, gates and decisions (protocol sections 3.5 to 3.7) |

## Reproduce

From this directory, with Rust 1.88 or newer and Python 3 with NumPy and
SciPy (`pip install scipy numpy`):

```sh
cargo build --release -p candidate-harness --features candidates
BIN=target/release/candidate-harness   # or $CARGO_TARGET_DIR/release/...

$BIN list
$BIN measure --out results/budgets.tsv
$BIN run --budgets results/budgets.tsv --out results/raw.tsv --reps 20 --threads 2
python3 analyze.py --raw results/raw.tsv --out results/report.md \
    --reps 20 --budgets results/budgets.tsv
```

Without `--features candidates` the registry holds the six incumbents and
`bipop` only, and the harness builds without the `cand-de` and `cand-eda`
crates.

`run` options: `--problems a,b`, `--solvers x,y`, `--protocols S,R`,
`--reps N` (default 20), `--threads N` (default 2) and `--budget-scale F`
(default 1; values below 1 are for smoke tests only and are recorded in every
row). `run` is resumable: rows already in the output file (key: solver,
problem, protocol, repetition) are skipped, every finished row is appended
and flushed at once, and an incomplete last line left by an interrupted
process is dropped and recomputed. An output file never mixes budget scales.

Smoke test, as used to check the harness:

```sh
$BIN run --budgets results/budgets.tsv --out results/smoke/raw.tsv \
    --budget-scale 0.02 --reps 2
python3 analyze.py --raw results/smoke/raw.tsv --out results/smoke/report.md \
    --reps 2 --solvers bite,de,cma,crfmnes,da,de_cma,bipop --budgets results/budgets.tsv
```

Tests and checks:

```sh
cargo test -p candidate-harness
cargo clippy -p candidate-harness --all-targets -- -D warnings
cargo fmt -p candidate-harness --check
```

## Budgets

`measure` evaluates every problem at 2,000 uniform random points (seeded),
single-threaded, and maps the mean time to B with the class table of
protocol section 3.3. `run` reads B from `results/budgets.tsv`, so budgets
stay fixed across resumptions. Timings are sensitive to other load on the
host: `measure` prints the load average, and it should run on an otherwise
idle machine. Several GTOP problems lie close to the 10 µs class boundary.

Protocol S is one run with budget `B' = ⌊B · budget_scale⌋`; protocol R is
eight runs with `⌊B'/8⌋` each. Every run has its own fresh `Guard`; the
recorded value is the guard's best value, never the solver's own claim.

## Seeds

Root seed 1. The seed of restart `k` of repetition `r` of protocol `P` on
problem `p` is

```text
h = splitmix64(1)
h = splitmix64(h ^ fnv1a64(p))
h = splitmix64(h ^ P)            // ASCII 'S' = 0x53 or 'R' = 0x52
h = splitmix64(h ^ r)
seed = splitmix64(h ^ k)         // k = 0 for protocol S
```

with the SplitMix64 finalizer (`x + 0x9E3779B97F4A7C15`, then xor-shift-
multiply by `0xBF58476D1CE4E5B9` and `0x94D049BB133111EB`) and 64-bit FNV-1a
of the UTF-8 problem name. All solvers get the same seeds and derive every
random number from them. The `measure` sample uses
`splitmix64(splitmix64(1) ^ fnv1a64(p) ^ fnv1a64("measure"))`.

## Incumbent configuration

All incumbents run with `workers = 1` and `fcmaes-core` library defaults, as
configured by the repository's GTOP benchmark binaries
(`examples/src/runner.rs`) and job worker (`crates/fcmaes-job-worker`).

| Name | Configuration |
|---|---|
| `bite` | `optimize_bite` / `DeepBiteOpt` with depth 1, `BiteParams` defaults (population `9 + 3D`, no stall termination); initial guess uniform in the box from `Rng::new(seed)`, BiteOpt seed `⌊U · (2³² − 1)⌋` from the same stream (as `bite_run` of the GTOP benchmark) |
| `de` | `De` with `DeParams` defaults: population 31, F 0.5, CR 0.9, keep 200, mutation limits 0.1 and 0.5; uniform initial population; seed = run seed |
| `cma` | active CMA-ES `Cmaes` with `CmaesParams` defaults (population 31, `mu` = 15, accuracy 1, default termination criteria) in the normalized box, initial mean uniform in the box, sigma 0.3 in normalized coordinates (the job worker's `CMAES_SIGMA`) |
| `crfmnes` | `Crfmnes` with `CrfmnesParams` defaults (population 32, bound penalty 1e5) in the normalized box, initial mean uniform in the box, sigma 0.3 in normalized coordinates (`CRFMNES_SIGMA`) |
| `da` | `optimize_da` with `DaParams` defaults (local search on), initial guess uniform in the box |
| `de_cma` | the DE→CMA sequence `de_cma_run` of the GTOP benchmark: step size `0.05 + 0.05 U` (drawn first, as the retry coordinator does), DE with `max(2B/5, 31)` evaluations and uniform initial population, then CMA-ES (population 31, normalized box) from the DE optimum with that step size, seed `run_seed ^ 0xA0761D6478BD642F`, and `max(B − DE share, 31)` evaluations |

Every objective budget B is passed to the core as `max_evaluations`. No
incumbent ever exceeds it, so the guard never refuses a call:

- `bite` and `de` use the library `optimize_until`, whose stop is checked
  before every evaluation (DE: also before its temporal-locality trial). The
  run is the exact prefix of the library run and uses B exactly.
- `cma` and `crfmnes` use `optimize_until` / `optimize_batch_until`, checked
  before every generation; the stop fires when the remaining budget is
  smaller than one population. The run is the library run cut after the last
  whole generation that fits; up to `popsize − 1` evaluations stay unused.
- `de_cma` stops its DE phase exactly at the DE share (or at B if B is
  smaller) and its CMA phase like `cma`.
- `da`: core `optimize_da` can exceed its documented limit of
  `max_evaluations`. In its local search, the finite-difference gradient
  (`2D + 1` calls) and the first line-search step evaluate without a budget
  check. `optimize_da` has no cooperative stop, so the adapter truncates the
  run: every call after the B-th evaluation is answered with `+inf` without
  evaluating anything, and the guard never sees it. Every decision before
  the B-th evaluation depends only on earlier values, so the evaluated
  sequence is exactly the first B evaluations of the library run with
  `max_evaluations = B`, and DA's result is the best of those B evaluations.

Every point is projected onto the box before evaluation. The core optimizers
clamp already; the projection only removes last-ulp rounding of the
normalized-to-real decoding (`0.5 · v · scale + typx` can exceed the upper
bound by one ulp).

## BIPOP-aCMA-ES

`bipop` (module `harness/src/bipop.rs`) is the restart schedule of N. Hansen,
"Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed"
(GECCO 2009 companion), over core `Cmaes` through ask/tell, with the
parameter values published for it. The primary paper could not be retrieved;
the schedule follows the description in Loshchilov, Schoenauer and Sebag,
"Alternative restart strategies for CMA-ES" (PPSN 2012) and the BIPOP example
of the DEAP library (`examples/es/cma_bipop.py`), and was compared with
Hansen's reference implementation pycma:

- default population `λ_def = 4 + ⌊3 ln D⌋`; initial step size σ₀ = 0.2 of the
  box width (Hansen's BBOB setting σ₀ = 2 on [−5, 5]), which is `0.4` in the
  normalized `[-1, 1]` coordinates of `Cmaes`;
- the first run is the large regime with multiplier 1 (`λ_def`, σ₀); its
  evaluations are booked to neither regime (DEAP);
- the `k`-th large restart uses `2^k λ_def` and σ₀;
- a small restart uses `λ_s = ⌊λ_def (½ λ_next/λ_def)^(U₁²)⌋` with
  `λ_next = 2^(k+1) λ_def` after `k` large restarts (so `λ_s` lies between
  `λ_def` and the latest large population, as in DEAP and pycma) and
  `σ_s = σ₀ · 10^(−2 U₂)`;
- a small run stops after at most `max(1, ⌈½ L / λ_s⌉)` generations, where `L`
  is the number of evaluations of the most recent large restart (DEAP);
- the next restart is small if the small regime has used strictly fewer
  evaluations than the large regime, otherwise large (DEAP and pycma use the
  same strict comparison), so ties go to the large regime and the first
  restart is large;
- every run starts at a uniform random point of the box;
- a run also ends when `Cmaes` reports a stop condition (`CmaesParams`
  defaults, `max_evaluations` = remaining budget) or when the remaining
  budget is smaller than one generation; the schedule ends when the next
  population does not fit into the remaining budget.

Differences from the sources: pycma books the first run to the small budget
and caps small runs at half of the cumulated large-regime evaluations (the
repository's job-worker BIPOP also uses the cumulated value); this harness
follows DEAP. DEAP and pycma also cap the generations of large runs with
Hansen's default `MaxIter`; core `Cmaes` has no generation limit, so large
runs end through the fcmaes termination criteria or the budget. DEAP runs a
fixed number of restarts; here the budget ends the schedule.

On a shifted 10-variable sphere `bipop` reaches below 1e-8 within 50,000
evaluations for five of five seeds, and on 10-variable Rosenbrock below 1e-6
within 200,000 evaluations for five of five seeds (`cargo test`).

## Problems

| Name | Domain | D | Source and objective |
|---|---|---:|---|
| `cassini1` | space | 6 | `fcmaes_gtop::cassini1` |
| `gtoc1` | space | 8 | `fcmaes_gtop::gtoc1(x) − 2,000,000` (shift of the examples registry) |
| `sagas` | space | 12 | `fcmaes_gtop::sagas` |
| `messenger` | space | 18 | `fcmaes_gtop::messenger` (reduced) |
| `tandem` | space | 18 | `fcmaes_gtop::tandem(x, [3, 2, 3, 3, 6])`, the constrained instance "Tandem 6" |
| `cassini2` | space | 22 | `fcmaes_gtop::cassini2` |
| `rosetta` | space | 22 | `fcmaes_gtop::rosetta` |
| `messenger_full` | space | 26 | `fcmaes_gtop::messenger_full` |
| `optical_lens` | optics | 11 | `optical_lens_design::evaluate(x, 8).scalar_score()`, the cost of the tutorial's single-objective arms, publication pupil grid radius 8; 1e12 (`INVALID_COST`) when the design does not trace |
| `bandpass` | circuits | 5 | `sindr_circuit_design::so::evaluate_bandpass(u, 201).objective` on the unit box, publication sweep of 201 points; 1e12 on failure |
| `phased_array` | antennas | 96 | `phased_array_codebook::geometry::evaluate_geometry(c, 4001).objective`, the cost `optimize_geometry` minimizes, publication cut of 4,001 points, unit box; 1e12 on failure |
| `lj13`, `lj38` | chemistry | 39, 114 | `foundations` Lennard-Jones suite, free Cartesian parameterization, box `±1.5 N^(1/3)`; 1e12 on an evaluation error (cannot occur inside the box) |

GTOP bounds are restated from `examples/src/problems.rs`, because the examples
crate depends on tokio and a finance client. No problem was dropped.

The circuit tutorial pins `fcmaes-core = "=0.1.3"` from crates.io and brings
its own copy. Its simulator `sindr 0.1.0-alpha.6` accepts `sindr-devices`
0.1.0-alpha.7 but does not compile against it, so the harness pins
`sindr-devices = "=0.1.0-alpha.6"`, the version of the tutorial's lockfile.

## File formats

`results/budgets.tsv`: header and one line per problem with `problem`,
`domain`, `dim`, `mean_eval_us`, `budget`.

`results/raw.tsv`: header and one line per repetition with `solver`,
`problem`, `domain`, `dim`, `protocol` (`S` or `R`), `rep` (from 0), `seed`
(seed of restart 0), `budget` (evaluations allowed for the row: `B'` for S,
`8 ⌊B'/8⌋` for R), `budget_scale`, `evaluations`, `best_value` (`inf` if no
evaluation was made), `over_budget`, `out_of_bounds`, `wall_ms`. For protocol
R, evaluations, violation counts and wall time are summed over the eight runs
and `best_value` is the best of them. Floats use the shortest round-trip
exponent form.

`results/report.md`: written by `analyze.py`. A cell (problem and protocol)
is complete when every expected solver has the requested number of
repetitions; incomplete cells are listed and excluded from best-incumbent
selection, tests, ranks and gates. Holm-adjusted p-values at most 0.05 are
significant. The report states the Python, NumPy and SciPy versions and the
limits of the experiment.
