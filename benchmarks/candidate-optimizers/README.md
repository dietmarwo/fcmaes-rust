# Candidate optimizers: benchmark harness

This workspace runs the experiment declared in [PROTOCOL.md](PROTOCOL.md):
which gradient-free algorithms missing from `fcmaes-core` should be added to
it. `PROTOCOL.md` is binding; this file describes how the harness implements
it. [DEVIATIONS.md](DEVIATIONS.md) lists the departures from the protocol and
[DECISION.md](DECISION.md) the outcome of the recorded campaign, which used
12 repetitions.

| Crate or file | Content |
|---|---|
| `solver-api` | `Solver` trait, `Outcome`, `Guard` (budget and box enforcement) |
| `cand-de` | L-SHADE, jSO, L-SRTDE candidates (L-SRTDE runs in two forms, `lsrtde` and `lsrtde_b`; see [DEVIATIONS.md](DEVIATIONS.md)) |
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
- `da`: at the time of the campaign, core `optimize_da` could exceed its
  documented limit of `max_evaluations`: in its local search, the
  finite-difference gradient (`2D + 1` calls) and the first line-search step
  evaluated without a budget check. `optimize_da` has no cooperative stop, so
  the adapter truncates the run: every call after the B-th evaluation is
  answered with `+inf` without evaluating anything, and the guard never sees
  it. Every decision before the B-th evaluation depends only on earlier
  values, so the evaluated sequence is exactly the first B evaluations of the
  library run with `max_evaluations = B`, and DA's result is the best of
  those B evaluations. The current `optimize_da` stops at the budget after
  the same first B evaluations, so with it the truncation does not fire and
  the recorded `da` rows reproduce.

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

## Round 2

The outcome is in [DECISION.md](DECISION.md). `python3 posthoc_round2.py`
prints two robustness checks that are outside the declared rules.

[ROUND2.md](ROUND2.md) declares a second round: one restart protocol (T)
with run budgets of the size fcmaes is normally used with, two DE→CMA-style
sequence arms, and the ten round-1 problems with a mean evaluation time of
at most 25 µs. Everything not stated here works as for round 1; protocols S
and R, their file format and the round-1 analysis are as described above.

### Reproduce

```sh
cargo build --release -p candidate-harness --features candidates
BIN=target/release/candidate-harness   # or $CARGO_TARGET_DIR/release/...

$BIN list
$BIN measure --out results/round2/budgets.tsv \
    --problems round2,truss_sizing,dispersion
$BIN run --protocols T --root-seed 2 --reps 20 --threads 2 \
    --out results/round2/raw.tsv
python3 analyze.py --round 2 --raw results/round2/raw.tsv \
    --out results/round2/report.md --reps 20
```

`run --protocols T` defaults to `--problems round2 --solvers round2`, which
`list` prints. Its other options are `--reps`, `--threads`, `--root-seed`
(default 1; the campaign uses 2), `--total` (default 400000), `--run-cap`
(default 100000) and `--min-run` (default 1000). `--budgets` and
`--budget-scale` do not apply, and T cannot be combined with S or R in one
call. The runner is resumable in the same way as for round 1; a file whose
rows carry other protocol-T settings is refused.

Smoke run and full-scale timing, as used before the campaign:

```sh
$BIN run --protocols T --root-seed 2 --total 8000 --run-cap 2000 --min-run 20 \
    --reps 2 --out results/round2/smoke/raw.tsv
python3 analyze.py --round 2 --raw results/round2/smoke/raw.tsv \
    --out results/round2/smoke/report.md --reps 2
$BIN run --protocols T --root-seed 999 --reps 1 --out results/round2/timing/raw.tsv
```

The smoke report states that its settings are not the declared ones and that
its numbers must not be used for any decision. The timing rows (root seed
999) are not campaign rows; only their wall times, evaluations and run
counts are used, for the compute rule of ROUND2.md section 4.

Tests:

```sh
cargo test --workspace --features candidate-harness/candidates
cargo clippy --workspace --all-targets --features candidate-harness/candidates -- -D warnings
```

`harness/tests/round2.rs` covers the sequence arms and the protocol-T driver,
`harness/tests/de_cma_frozen.rs` holds evaluation sequences and campaign rows
of `de_cma` recorded from the round-1 adapter, which the shared sequence code
must reproduce bit for bit.

### Protocol T

A repetition has a total of 400,000 evaluations. Run `k = 0, 1, 2, ...` gets
`min(100000, remaining)` evaluations behind its own fresh `Guard`, with seed

```text
h = splitmix64(2)
h = splitmix64(h ^ fnv1a64(p))
h = splitmix64(h ^ 0x54)        // ASCII 'T'
h = splitmix64(h ^ r)
seed = splitmix64(h ^ k)
```

(the round-1 derivation with root seed 2). `remaining` is the total minus
the evaluations the guards counted, never the solvers' claims. A run ends by
its own stop criterion or by its budget; restarting ends when fewer than
1,000 evaluations remain. If a run makes no evaluation at all, restarting
stops as well (a solver that returns at once would otherwise restart
forever); the runner prints a warning, and the row shows it as
`total − evaluations ≥ min_run`. The recorded value is the best value any of
the guards saw.

### Sequence arms

`jso_cma` and `lsrtde_cma` share one function with `de_cma`
(`cma_sequence` in `harness/src/solvers.rs`); only the first stage differs.
For a run with budget B and seed s:

| Step | Configuration |
|---|---|
| Step size and run seed | `Rng::new(s)`: σ = `0.05 + 0.05 U` drawn first, then `run_seed = next_u64()` |
| First stage share | `max(⌊2B/5⌋, 31)`; the stage stops at `min(max(⌊2B/5⌋, 31), B)` evaluations |
| First stage, `de_cma` | `De`, `DeParams` defaults, `max_evaluations` = the share, uniform initial population, seed `run_seed`, cooperative stop at the stop count |
| First stage, `jso_cma` | `cand_de::Jso::default()` as a whole run with budget = the stop count and seed `run_seed`, so its population schedule is driven by the first-stage budget |
| First stage, `lsrtde_cma` | `cand_de::LSrtde { pbest_difference_from_base: true, ..Default::default() }` (the `lsrtde_b` candidate), configured like jSO |
| Second stage | core `Cmaes`, `CmaesParams` defaults (population 31), normalized box, start at the first stage's best point, σ in every coordinate, seed `run_seed ^ 0xA0761D6478BD642F`, `max_evaluations = max(B − share, 31)`, stopped before the first generation that does not fit into the remaining budget |
| Budget spent in the first stage | the run ends without the second stage |

The stop count equals the share whenever B ≥ 31; below that, the candidate
gets the whole budget B, exactly as `de_cma` stops DE at B. Under protocol T
with 100,000 evaluations per run the first stage gets 40,000.

### Problems

Round-2 problem set (`--problems round2`), with the mean evaluation times of
`results/round2/budgets.tsv` (2,000 uniform random points, one thread, load
average 0.23 before measuring):

| Problem | Domain | D | Mean evaluation time (µs) |
|---|---|---:|---:|
| `cassini1` | space | 6 | 7.2 |
| `gtoc1` | space | 8 | 10.3 |
| `sagas` | space | 12 | 4.2 |
| `messenger` | space | 18 | 10.0 |
| `tandem` | space | 18 | 9.3 |
| `cassini2` | space | 22 | 12.9 |
| `rosetta` | space | 22 | 11.0 |
| `messenger_full` | space | 26 | 13.7 |
| `lj13` | chemistry | 39 | 0.44 |
| `lj38` | chemistry | 114 | 3.0 |

Problems examined under the entry rule of ROUND2.md section 3 and left out:

| Problem | Domain | D | Objective | Mean evaluation time (µs) | Result |
|---|---|---:|---|---:|---|
| `truss_sizing` | structures | 171 | `tutorials/truss-sizing`, the cost of its scalar arms (`so::optimize_arm`): `evaluate(controls, &GroundStructure::reference(), Scenario::TRAINING, false, &counter).objective` (penalized mass), `INVALID_COST` (1e12) when the controls do not decode; unit box | 36.7 | builds, deterministic, too slow (limit 25 µs) |
| `dispersion` | environment | 12 | `tutorials/dispersion-source-localization`, the cost of its scalar arm (`optimize_scalar`): the tutorial's public `scalar_objective(values, &Dataset::synthetic())`, which returns 1e99 for an invalid design; the tutorial's bounds | 43.3 | builds, deterministic, too slow (limit 25 µs) |

Both objectives call the tutorials' own public functions; the harness adds
no scalarization of its own. Determinism was checked by evaluating 200
points of each measurement sample twice (`measure` prints the result) and
in `cargo test`. The decision used the timing in `budgets.tsv` only; a second
measurement gave 47.7 and 46.7 µs. The tutorials' scalar arms start from a
seeded baseline design and replay it at the end; here, as for every other
problem, solvers start from their own uniform initial points. Both problems
stay registered (`list`, `measure`) but are not in the round-2 set. The
optical lens, the band-pass filter and the phased array are left out by
ROUND2.md itself.

### Compute projection

One full-scale repetition (all 12 solvers, all 10 problems, root seed 999,
two threads) took 381.5 s of summed run wall time, which projects to
381.5 s × 20 / 2 ≈ 64 minutes for 20 repetitions on two threads (48 minutes
for 15). That is below the 180 minutes of the compute rule, so the campaign
keeps 20 repetitions.

### File formats

`results/round2/budgets.tsv`: the round-1 format. The `budget` column is the
round-1 class value and is not used by protocol T.

`results/round2/raw.tsv`: the round-1 columns followed by `runs` (number of
runs made, including a final run without evaluations if there was one),
`root_seed`, `total`, `run_cap` and `min_run`. `protocol` is `T`, `budget`
equals `total`, `budget_scale` is 1, `seed` is the seed of run 0, and
`evaluations`, `over_budget`, `out_of_bounds` and `wall_ms` are summed over
the runs.

`results/round2/report.md`: written by `analyze.py --round 2`. Cells are
problems; incumbents `bite`, `de`, `cma`, `crfmnes`, `da`, `de_cma`;
candidates `lshade`, `jso`, `lsrtde_b`, `bipop`, `jso_cma`, `lsrtde_cma`.
The best incumbent per cell has the lowest median (ties: lower mean). Each
candidate is compared with it by a two-sided Mann–Whitney U test (SciPy
default method; a NaN p-value, which occurs when all values are identical,
counts as 1), Holm-corrected over the candidate's complete cells, α = 0.05.
G1 needs at least 3 winning cells, G2 a mean rank among the twelve solvers
strictly better than the best incumbent's. The report lists the decision,
mean ranks, win/loss/tie per cell with adjusted p-values, per-cell medians,
runs per repetition and evaluations used per solver, violations, incomplete
cells (excluded) and limits. A file whose settings differ from the declared
ones (root seed 2, total 400,000, run cap 100,000, minimum run 1,000) is
reported as a smoke test.

`results/round2/smoke/` and `results/round2/timing/` hold the smoke run (raw
rows, report, runner log) and the timing repetition (raw rows, runner log).

## Round 3: problem survey

[ROUND3.md](ROUND3.md) declares a mixed-variable screening of CatCMA with
Margin. Its problems enter under a rule fixed in section 3 (deterministic,
at least one discrete coordinate with a clean per-coordinate decoding, mean
evaluation time at most 2 ms), and the round needs at least five entered
problems, at least three of them with a categorical coordinate.

```sh
$BIN survey --out results/round3
cargo test -p candidate-harness --features candidates --test round3
```

`survey` builds every examined problem, times it over the 2,000-point
measurement sample of `measure` (single thread; it stops early once the
elapsed time exceeds 2,000 × 2 ms, which already decides the rule), checks
determinism by evaluating 200 points twice (10 points for problems that the
timing excludes), and writes `results/round3/problems.tsv` and one type table
per problem, `results/round3/types/<problem>.json`. `survey.log` holds the
console output of the recorded run (load average 0.76 before measuring).

### Outcome

Eight problems meet the entry rule, but only two of them have a categorical
coordinate. The requirement of ROUND3.md section 3 is not met, so the round
stops here: no round-3 solver configuration, candidate driver or analysis
was built.

| Problem | Domain | D | Continuous | Integer | Categorical | Mean evaluation time (µs) | Entered |
|---|---|---:|---:|---:|---:|---:|---|
| `truss_sizing` | structures | 171 | 95 | 76 | 0 | 37.1 | yes |
| `phased_array` | antennas | 96 | 48 | 48 | 0 | 985 | yes |
| `field_service` | logistics | 104 | 52 | 0 | 52 | 25.8 | yes |
| `coverage_cardinality` | networks | 4,000 | 0 | 4,000 | 0 | 47.8 | yes |
| `coverage_weighted` | networks | 4,000 | 0 | 4,000 | 0 | 64.0 | yes |
| `jobshop` | scheduling | 8 | 5 | 0 | 3 | 0.21 | yes |
| `uav_assignment` | logistics | 34 | 30 | 4 | 0 | 0.53 | yes |
| `material_flow` | manufacturing | 2 | 0 | 2 | 0 | 174 | yes |
| `energy_hub` | energy | 10 | 7 | 3 | 0 | > 2,000 (27,100 over 148 points) | no |
| `energy_hub_annual` | energy | 10 | 6 | 4 | 0 | > 2,000 (313,000 over 13 points) | no |
| `phased_array_beam` | antennas | 32 | 0 | 32 | 0 | > 2,000 (6,160 over 650 points) | no |
| `water_network` | water | 28 | 3 | 24 | 1 | > 2,000 (4,270 over 939 points) | no |
| `ml_tuning` | machine learning | 8 | 3 | 4 | 1 | > 2,000 (10.1 s for 1 point) | no |
| `bandpass` | circuits | 5 | 5 | 0 | 0 | 416 | no: no discrete coordinate |
| `cassini1_minlp` | space | 6 | 6 | 0 | 0 | 5.9 | no: no discrete coordinate |

Objectives and sources:

| Problem | Scalar cost (invalid value) | Box |
|---|---|---|
| `truss_sizing` | as in round 2 | unit box |
| `phased_array` | as in round 1 | unit box |
| `field_service` | `scenarios::evaluate_training(x, &instance::load_primary())`, the cost of `so::optimize` (`INVALID_OBJECTIVE` = 1e99) | unit box |
| `coverage_cardinality`, `coverage_weighted` | `so.rs` `objective_value` (private there, restated exactly) of `decode::decode(x)` on the publication instance `reference-4k` (`+inf` for a non-finite input) | `[0, UPPER]`, `UPPER` = 1.999999999999 |
| `jobshop` | `JobShop::mini().fitness(x)`, the default instance of `examples/src/bin/jobshop.rs` | `[1e-7, 0.9999999]` |
| `uav_assignment` | `UavProblem::generate(5, 30, 5000, 65).fitness(x)`, the defaults of `examples/src/bin/uav_task_assignment.rs` | unit box |
| `material_flow` | `Plant::original().fitness(x)` | `[1, 50]` |
| `energy_hub` | `evaluate::evaluate_training(x, Preset::Publication)` (`INVALID_OBJECTIVE` = 1e6) | unit box |
| `energy_hub_annual` | `annual::evaluate_annual(x, 6)` (1e6, a private constant there, restated) | unit box |
| `phased_array_beam` | `so::evaluate_beam(x, &BeamContext::stage_a(4001), 20.0)` (`INVALID_COST` = 1e12) | unit box |
| `water_network` | `evaluate::evaluate_training(x, &network::load())` (`INVALID_OBJECTIVE` = 1e12) | unit box |
| `ml_tuning` | `Evaluator::evaluate(x).scalar_fitness`, publication dataset, minimum recall 0.25, model seed `stream_seed(42, 50)` | unit box |
| `bandpass`, `cassini1_minlp` | as in round 1; the examples registry fixes the Cassini 1 MINLP flyby sequence to `[2, 2, 3, 5]` | as there |

The examples crate is not a dependency (it pulls in tokio and a finance
client); `harness/src/examples.rs` compiles its self-contained modules
`jobshop.rs`, `uav.rs` and `material_flow_planning.rs` from their source
files, unchanged.

Examined and not built:

| Problem | Reason |
|---|---|
| `cassini2_minlp` | decodes `x[22..26]` to flyby planets, but no single-objective code in the repository minimizes it and the repository defines no search box for the planet coordinates |
| Tandem with free sequence | `fcmaes_gtop::tandem` takes the sequence as a separate argument; every use fixes it |
| Dyson-ring scheduling (`examples/src/scheduling.rs`) | trajectory selection repairs collisions across coordinates; order and timing are sorted keys: no clean typed form |
| E12 band-pass (`sindr-circuit-design`) | used only by the MAP-Elites arm, no single-objective code |
| Mazda, voltage control, gate driver, production line | multi-objective or quality-diversity only |

Integer masks of the problems' own DE calls: only the network-coverage
tutorial passes one (`Some(vec![true; dimension])`); every other DE call of
an examined problem passes `None`.

### Type tables

Each JSON file holds `problem`, `dimension`, `source` (file and function the
classification came from), `counts`, `notes` and one entry per coordinate:
`index`, `kind`, `lower`, `upper`, `meaning` and, for a discrete coordinate,
`levels` (the real-vector interval centers a typed point maps to), `values`
(integer coordinates: the integer the decoder computes), `labels` and
`intervals` (per level, the half-open intervals `[lo, hi)` that decode to it;
an interval ending at the upper bound includes it). Classification remarks:

- Random keys (truss member ranking, phased-array slot activation, route and
  visit-order priorities) act only jointly and stay continuous; the
  coordinates of those problems with a per-coordinate discrete decoding are
  typed.
- Truss sections and energy-hub grid tiers are catalogs sorted by size:
  integer. Phase codes are quantized values (integer), although the phase is
  cyclic.
- A field-service assignment coordinate is binned twice when vehicle 7 can
  serve the task (nominal and vehicle-7-unavailable scenarios); its levels are
  the cells of both partitions, each a pair of vehicles. The water-network
  pump priority (a choice between two pumps) and the ML split criterion are
  two-level categorical coordinates.

`tests/round3.rs` checks every entered problem against its own decoder:
every level decodes to itself at its center, points just below and above
every interval boundary decode to the neighboring levels, and moving every
discrete coordinate of 50 random points to its level center leaves the
objective bit-identical. For `material_flow` the decoder is inside `fitness`;
the test identifies the level by replaying the simulation.
