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
per problem, `results/round3/types/<problem>.json`. The tables of the two
coverage problems (4,000 coordinates each, about 1 MB) are not stored;
`survey` regenerates them. `survey.log` holds the console output of the
recorded run (load average 0.76 before measuring).

### Outcome

Eight problems meet the entry rule, but only two of them have a categorical
coordinate, so the requirement of ROUND3.md section 3 is not met.
[DEVIATIONS.md](DEVIATIONS.md) (round 3) records how the round continues: as
a narrower screening on six problems (the two network-coverage problems are
left out by a dimension cap of 200), with binary switches typed integer and
the candidate seeded with the low 32 bits of the shared seed. The campaign
machinery is described in the next section.

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

## Round 3: campaign

This section describes how the harness and the candidate driver implement
[ROUND3.md](ROUND3.md) under the round-3 entries of
[DEVIATIONS.md](DEVIATIONS.md).

### Reproduce

From this directory, with Python 3, NumPy, SciPy and the pinned `cmaes`
package. The steps must run in this order, and the Rust runner and the
driver must never write the raw file at the same time (see "Writer lock"):

```sh
python3 -m pip install --break-system-packages cmaes==0.13.1
cargo build --release -p candidate-harness --features candidates
BIN=target/release/candidate-harness   # or $CARGO_TARGET_DIR/release/...

python3 run_catcmawm.py verify           # bridge check, must print "bridge verified"
python3 run_catcmawm.py spaces           # variable spaces and population sizes

# 1. the seven incumbents (Rust runner)
$BIN run --protocols T --problems round3 --solvers round3 \
    --totals 2000,20000 --root-seed 3 --min-run 100 --reps 20 --threads 2 \
    --out results/round3/raw.tsv
# 2. the candidate (Python driver), after step 1 has finished
python3 run_catcmawm.py run --out results/round3/raw.tsv \
    --totals 2000,20000 --root-seed 3 --min-run 100 --reps 20 --workers 2
# 3. the report
python3 analyze.py --round 3 --raw results/round3/raw.tsv \
    --out results/round3/report.md --reps 20
```

Both writers are resumable: an interrupted step is restarted with the same
command. If the compute rule of ROUND3.md section 5 lowers the repetitions,
`--reps` changes in all three commands. The driver finds the harness binary
through `$CARGO_TARGET_DIR` (else `target/`), or through `--harness PATH`.

Smoke run and timing repetition, as used before the campaign:

```sh
$BIN run --protocols T --problems round3 --solvers round3 --totals 200,2000 \
    --root-seed 3 --min-run 20 --reps 2 --out results/round3/smoke/raw.tsv
python3 run_catcmawm.py run --out results/round3/smoke/raw.tsv \
    --totals 200,2000 --root-seed 3 --min-run 20 --reps 2 --workers 2
python3 analyze.py --round 3 --raw results/round3/smoke/raw.tsv \
    --out results/round3/smoke/report.md --reps 2

$BIN run --protocols T --problems round3 --solvers round3 --totals 2000,20000 \
    --root-seed 999 --min-run 100 --reps 1 --out results/round3/timing/raw.tsv
python3 run_catcmawm.py run --out results/round3/timing/raw.tsv \
    --totals 2000,20000 --root-seed 999 --min-run 100 --reps 1 --workers 2
```

Tests:

```sh
cargo test --workspace --features candidate-harness/candidates
cargo clippy --workspace --all-targets --features candidate-harness/candidates -- -D warnings
cargo fmt --all --check
python3 -m unittest -v test_catcmawm     # needs the release binary
```

`harness/tests/round3_runner.rs` covers the round-3 sets, the two-total
campaign (cells, resume key, refused settings), the writer lock, the seed rule
and the `serve` protocol (bit-identical values on all six problems, decoder
levels, refusals). `test_catcmawm.py` covers the Python seeds against the
`seeds` subcommand, the wire format and the float format against Rust, the
guard, the restart protocol on a toy problem, the type mapping, the bridge
check, resumption and the lock shared with the Rust runner, and the
resampling of G2.

### Solvers

`--solvers round3` selects the seven incumbents `bite`, `de`, `cma`,
`crfmnes`, `da`, `de_cma` and `jso_cma`, configured as in round 2 (sections
"Incumbent configuration" and "Sequence arms"; `jso_cma` is the harness arm
benchmarked in round 2 and needs `--features candidates`). The candidate
`catcmawm` runs through `run_catcmawm.py`.

**Integer masks (ROUND3.md section 2).** None of the six problems' own
optimization code passes an integer mask to DE, so `de` and the DE stage of
`de_cma` run without one, as in round 2. The runner refuses a problem whose
own DE call passes a mask (the network-coverage problems, which are outside
the round-3 set).

| Problem | Own optimization code | DE mask |
|---|---|---|
| `truss_sizing` | `tutorials/truss-sizing/src/so.rs`: `De::new(.., None, ..)` | none |
| `phased_array` | `tutorials/phased-array-codebook/src/geometry.rs`: `optimize_geometry`, BiteOpt only | none (no DE call) |
| `field_service` | `tutorials/field-service-routing/src/so.rs`: `De::new(.., None, ..)` | none |
| `jobshop` | `examples/src/bin/jobshop.rs` through `examples/src/bin/common/mod.rs`: BiteOpt only | none (no DE call) |
| `uav_assignment` | `examples/src/bin/uav_task_assignment.rs`: BiteOpt retry, MODE with `None` | none (no DE call) |
| `material_flow` | `examples/src/bin/material_flow_planning.rs`: BiteOpt ask/tell only | none (no DE call) |

**Candidate.** `cmaes.CatCMAwM` of the Python package `cmaes` 0.13.1 (MIT
license, CyberAgent), the authors' reference implementation. The driver
refuses to run with another installed version or a non-MIT license. It
constructs `CatCMAwM(x_space=..., z_space=..., c_space=..., seed=...)` with
every other argument at its default:

- population `4 + ⌊3 ln N⌋`, N the number of all variables; initial mean at
  the center of the box (integer coordinates: the middle of the value
  range), initial step size 1 and a diagonal covariance whose standard
  deviations are half the range divided by 6 (continuous) or 5 (integer);
  uniform categorical distributions;
- continuous variables are bounded by `x_space`: the Gaussian sample can
  leave the box, but `ask()` clips the solution's `x` to `x_space` and `tell`
  adds a quadratic box penalty for the raw sample, so every evaluated point
  lies in the box and the driver does not clip;
- `should_stop()` checks the CMA-ES criteria of the Gaussian part only
  (function-value range below 1e-12 over a history, step size below
  `tolx`, divergence, no-effect coordinate or axis, condition number above
  1e14); there is no criterion for the categorical part;
- `seed` initializes `numpy.random.RandomState`, which accepts 32-bit seeds;
  the driver passes the low 32 bits of the run seed (DEVIATIONS.md, round 3,
  entry 4);
- `tell` accepts only whole populations. If the remaining allowance is
  smaller than one population, the driver evaluates only the first samples
  that fit and ends the run without a `tell`.

### Problems

`--problems round3`, with the type counts of `results/round3/types/` and the
mean evaluation times of `results/round3/problems.tsv` (2,000 uniform random
points, one thread):

| Problem | Domain | D | Continuous | Integer | Categorical | Mean evaluation time (µs) | CatCMAwM population |
|---|---|---:|---:|---:|---:|---:|---:|
| `truss_sizing` | structures | 171 | 95 | 76 | 0 | 37.1 | 19 |
| `phased_array` | antennas | 96 | 48 | 48 | 0 | 985 | 17 |
| `field_service` | logistics | 104 | 52 | 0 | 52 | 25.8 | 17 |
| `jobshop` | scheduling | 8 | 5 | 0 | 3 | 0.21 | 10 |
| `uav_assignment` | logistics | 34 | 30 | 4 | 0 | 0.53 | 14 |
| `material_flow` | manufacturing | 2 | 0 | 2 | 0 | 174 | 6 |

The two network-coverage problems stay registered (`survey`, `measure`,
tests) but are outside the set.

**Type mapping.** The driver builds the variable spaces from the type
tables: a continuous coordinate gives an `x_space` row `[lower, upper]`, an
integer coordinate a `z_space` row with the integers the decoder computes
(`values`; for example batch sizes 1 to 50, catalog indices 0 to 11, phase
codes 0 to 63), a categorical coordinate a `c_space` entry with its number
of categories. A typed sample maps to the real vector by placing every
discrete coordinate at the center of the interval that decodes to the chosen
level (`levels` of the table) and taking continuous values as sampled. The
driver checks that the table bounds equal the harness box bit for bit.

### Protocol and seeds

Protocol T (section "Protocol T" of round 2) with two totals per problem,
2,000 and 20,000 evaluations, each with run cap equal to the total, minimum
run 100 and root seed 3. `--totals 2000,20000` runs both into one file; each
(problem, total) is a cell, and the resume key is (solver, problem,
protocol, repetition, total). A file accepts only rows whose settings are
among the current ones, so a file never mixes root seeds or minimum runs, and
resumption passes the same `--totals`.

The seed of run `k` of repetition `r` on problem `p` is
`run_seed_from(root, p, 'T', r, k)` (the round-2 derivation). The total does
not enter it, so the cells at 2,000 and at 20,000 evaluations use the same run
seeds. The candidate gets the low 32 bits of the same value. `candidate-harness
seeds` prints the seeds of a grid; `run_catcmawm.py` restates the derivation,
and the tests compare both for six problems, two totals, three repetitions,
four restarts and root seeds 3 and 999.

The driver follows the Rust runner: run `k` gets `min(total, remaining)`
evaluations behind its own guard (the semantics of `solver_api::Guard`:
points outside the box or beyond the allowance are refused, counted and
answered with `+inf`; the guard records the best value). A run ends when
`should_stop()` fires after a generation or when the allowance is spent;
restarting ends when fewer than 100 evaluations remain or after a run
without evaluations. Evaluations, violations and wall time are summed over
the runs; the best value over all runs counts.

### Objective bridge (`serve`)

`candidate-harness serve` reads requests on standard input and answers on
standard output; the driver starts it once per process. A request is a
header line followed by the data lines it announces; every response starts
with `ok` (or is a single `err <message>` line, after which the process
exits) and is flushed as a whole. Every `f64` travels as its IEEE-754 bit
pattern in 16 lowercase hex digits, so values round-trip exactly.

| Request | Data lines | Response |
|---|---|---|
| `info P` | none | `ok D domain`, then the lower and the upper bounds |
| `eval P N` | N vectors | `ok N NS`, then N values; NS is the objective time of the batch in nanoseconds |
| `check P N` | N vectors, each followed by an expected value | `ok M`, M mismatches against the in-process evaluation of a separately built instance |
| `levels P N` | N vectors | `ok N`, then per vector the level index of every discrete coordinate under the problem's own decoder (`-` for continuous) |
| `echo N`, `fmt N` | N lines | the values re-encoded, or in the raw-file float format |
| `quit` | none | the process exits |

`eval` refuses vectors of the wrong length or outside the box. The driver
sends one generation per `eval` request.

`python3 run_catcmawm.py verify` checks the bridge: for 1,000 uniform random
vectors per problem (plus the two box corners), the values returned by
`eval` equal bit for bit the in-process evaluation of a separately built
instance and the vectors survive the wire unchanged; for 200 uniform random
typed points and 50 samples of a fresh `CatCMAwM` per problem, the mapped real
vector decodes with the problem's own decoder to the chosen level of every
discrete coordinate. The decoder probes are those of `tests/round3.rs`
(`round3::level_probe`).

### Writer lock

The Rust runner (every `run`) and the driver take the lock file
`<out>.lock`, holding the process id, before they read the raw file, and
remove it when they finish. A second writer is refused while the lock is
held. A lock whose process no longer exists is replaced with a warning. The
driver's `--workers 2` runs repetitions in two worker processes, each with its
own `serve`; only the parent process holds the lock and writes the file.

### Smoke run and compute projection

The smoke run (totals 200 and 2,000, minimum run 20, two repetitions, all
eight solvers) recorded no contract violation; its report marks itself as a
smoke test.

One full-scale repetition (totals 2,000 and 20,000, root seed 999, two
threads for the Rust runner, two worker processes for the driver; not a
campaign row) gave these summed run wall times:

| Solver | Wall time (s) | Runs per repetition, total 2,000 | Runs per repetition, total 20,000 |
|---|---:|---|---|
| `bite` | 32.1 | 1 on every problem | 1 on every problem |
| `de` | 31.4 | 1 | 1 |
| `cma` | 37.9 | 1 to 4 | 1 to 33 |
| `crfmnes` | 31.4 | 1 | 1 |
| `da` | 31.4 | 1 | 1 to 5 |
| `de_cma` | 33.9 | 1 to 2 | 1 to 6 |
| `jso_cma` | 35.1 | 1 to 2 | 1 to 6 |
| `catcmawm` | 74.4 | 1 to 2 | 1 to 20 |

`phased_array` takes 187 s of the 233 s of the seven Rust solvers. Of the
candidate's 74.4 s, 29.2 s are objective time inside `serve` and 2.4 s the
bridge (encoding, pipe, decoding), so 61% of its wall time is spent outside
objective evaluation, almost all of it in the Python implementation.
`catcmawm` ended runs by `should_stop()` on `jobshop` at 20,000 (one run) and
on `material_flow` (one run at 2,000, 19 at 20,000); every other run ended
at its allowance. Evaluations left unused are the rest of a last generation
that did not fit or a remainder below the minimum run (at most 76 per
repetition, `jso_cma` on `material_flow` at 2,000).

Projection (summed run wall time × repetitions / 2): for 20 repetitions,
233.3 s × 20 / 2 ≈ 39 minutes for the Rust runner and 74.4 s × 20 / 2 ≈ 12
minutes for the driver with two workers (25 minutes serially), about 51
minutes in all; for 15 repetitions about 29 + 9 = 38 minutes. That is below
the 180 minutes of the compute rule, so the campaign keeps 20 repetitions.
The load average before the runs was 1.5 and 1.8 (a build had just finished).

### File formats

`results/round3/raw.tsv`: the protocol-T format of round 2 (`RAW_HEADER_T`);
`budget`, `total` and `run_cap` equal the cell's total, `root_seed` is 3 and
`min_run` 100. The Rust runner and the driver write the same columns; floats
use Rust's shortest round-trip exponent form (`{:e}`), which the driver
reproduces (tested against `serve fmt`), and `wall_ms` has three decimals.
Every row is appended with a single write and flushed; an incomplete last
line left by an interrupted writer is dropped and recomputed by either
writer.

`results/round3/raw.catcmawm.tsv` (next to any raw file the driver writes):
one informational line per candidate row with `problem`, `total`, `rep`,
`popsize`, `runs`, `stopped_by_criterion`, `ended_by_allowance`,
`evaluations`, `objective_ms` (inside `serve`), `bridge_ms` (request round
trips including objective time) and `wall_ms`. The analysis does not read it.

`results/round3/report.md`: written by `analyze.py --round 3`. Cells are
(problem, total); incumbents `bite`, `de`, `cma`, `crfmnes`, `da`, `de_cma`,
`jso_cma`; candidate `catcmawm`. The best incumbent per cell has the lowest
median (ties: lower mean). The candidate is compared with it by a two-sided
Mann–Whitney U test (SciPy default method; a NaN p-value counts as 1),
Holm-corrected over the candidate's complete cells, α = 0.05. Ranks are by
median among the eight solvers per cell (average ranks for ties). G1 needs at
least 3 winning cells on at least 2 problems. G2 needs a mean rank strictly
better than the best incumbent's and, in at least 95% of 10,000 resamples
(NumPy `default_rng(3)`; repetitions drawn with replacement within each cell
and solver, all mean ranks recomputed), a mean rank strictly below every
incumbent's; the report gives the share. Include, Watch and Reject follow
ROUND3.md section 6, with Include requiring no contract violation. The
report lists the decision, mean ranks, win/loss/tie per cell with adjusted p,
per-cell medians, the descriptive split (cells of `field_service` and
`jobshop` against the other four problems), runs and evaluations per solver
and total, violations, incomplete cells (excluded) and the limits of
DEVIATIONS.md round 3. A file whose settings differ from the declared ones
(root seed 3, totals 2,000 and 20,000, minimum run 100) is reported as a smoke
test.

`results/round3/smoke/` and `results/round3/timing/` hold the smoke run (raw
rows, driver details, report, logs) and the timing repetition (raw rows,
driver details, logs).

## Round 3, second batch

[ROUND3B.md](ROUND3B.md) adds three published problems with unordered
categorical variables to round 3. The decision on CatCMAwM is taken on the
union of the round-3 cells and the cells of this batch (ROUND3B.md section
4). Solvers, protocol, seeds, the bridge, the driver and the file formats
are those of "Round 3: campaign"; this section describes what differs. All
three problems entered; none was dropped.

### Reproduce

From this directory, after the round-3 campaign (the union report needs
`results/round3/raw.tsv`). The steps run in this order, and the Rust runner
and the driver never write the raw file at the same time:

```sh
python3 -m pip install --break-system-packages cmaes==0.13.1
cargo build --release -p candidate-harness --features candidates
BIN=target/release/candidate-harness   # or $CARGO_TARGET_DIR/release/...

$BIN survey3b --out results/round3b                    # type tables, timing, determinism
python3 round3b_reference.py verify                     # must print "round-3b references verified"
python3 run_catcmawm.py verify --problems round3b       # must print "bridge verified"
python3 run_catcmawm.py spaces --problems round3b

# 1. the seven incumbents (Rust runner)
$BIN run --protocols T --problems round3b --solvers round3 \
    --totals 2000,20000 --root-seed 3 --min-run 100 --reps 20 --threads 2 \
    --out results/round3b/raw.tsv
# 2. the candidate (Python driver), after step 1 has finished
python3 run_catcmawm.py run --problems round3b --out results/round3b/raw.tsv \
    --totals 2000,20000 --root-seed 3 --min-run 100 --reps 20 --workers 2
# 3. the batch alone, then the union with round 3 (the decision)
python3 analyze.py --round 3b --raw results/round3b/raw.tsv \
    --out results/round3b/report.md --reps 20
python3 analyze.py --round 3u --raw results/round3/raw.tsv,results/round3b/raw.tsv \
    --out results/round3b/report-union.md --reps 20
```

If the compute rule of ROUND3.md section 5 lowers the repetitions, `--reps`
changes in all four commands. `round3b_reference.py verify --smt DIR
--bounce FILE` also compares its copies of the reference code with an
unpacked SMT 2.15.1 package and Bounce's `pest_control.py`.

Tests:

```sh
cargo test --workspace --features candidate-harness/candidates   # includes tests/round3b.rs
python3 -m unittest -v test_catcmawm     # classes Round3b and UnionAnalysis; needs the release binary
```

### Problems

`--problems round3b`, with the type counts of `results/round3b/types/` and
the mean evaluation times of `results/round3b/problems.tsv` (2,000 uniform
random points of the measurement sample, one thread; 200 points evaluated
twice gave identical values for every problem):

| Problem | Domain | D | Continuous | Integer | Categorical | Mean evaluation time (µs) | CatCMAwM population |
|---|---|---:|---:|---:|---:|---:|---:|
| `welded_beam_mixed` | structures | 6 | 1 | 4 | 1 | 0.055 | 9 |
| `cantilever_mixed` | structures | 3 | 2 | 0 | 1 | 0.021 | 7 |
| `pest_control` | agriculture | 25 | 0 | 0 | 25 | 1.38 | 13 |

The problems come from outside this repository and have no optimization
code of their own, so no DE integer mask applies; `de` and `de_cma` run
without one.

**Encoding (ROUND3B.md section 2).** Every discrete coordinate is one real
coordinate in `[0, 1]` split into equal intervals, one per level, in the
order of the source: level `k` of `n` decodes from `[k/n, (k+1)/n)`, the
last level also from 1, by `min(floor(u n), n - 1)`, the rule of
`categorical_index` in `examples/src/encoding.rs`
([docs/combinatorial-encodings.md](../../docs/combinatorial-encodings.md)).
Continuous coordinates keep the bounds of the source. The type tables have
the format of round 3; the candidate's typed point goes to the interval
centers (`levels`).

**Constraints (ROUND3B.md section 2).** Only the welded beam is constrained.
Its cost is `f + 1e6 * sum(max(0, c_i))` over the four constraints of the
source, each written as `g(x) <= 0` and divided by its limit:

| Source constraint | Normalized form used |
|---|---|
| `g1 = S - sigma >= 0` (bending) | `(sigma - S) / S <= 0` |
| `g2 = Pc - F >= 0` (buckling) | `(F - Pc) / F <= 0` |
| `g3 = delta_max - delta >= 0` (deflection) | `(delta - 0.25) / 0.25 <= 0` |
| `g4 = 0.577 S - tau >= 0` (weld shear) | `(tau - 0.577 S) / (0.577 S) <= 0` |

### Welded beam with material and weld-type choice (`welded_beam_mixed`)

**Source.** K. Deb and M. Goyal, "Optimizing engineering designs using a
combined genetic search", Proceedings of the 7th International Conference
on Genetic Algorithms (1997), section 4.4, tables 4 and 5: the mixed welded
beam that the same authors introduced with GeneAS (Computer Science and
Informatics 26(4), 1996). The paper takes the shear, bending, deflection and
buckling formulas from G. V. Reklaitis, A. Ravindran and K. M. Ragsdell,
"Engineering Optimization: Methods and Applications" (1983), the classic
welded beam. Read as a paper; no code exists.

**Definition** (`harness/src/round3b.rs`, module `welded_beam`):

| Coordinate | Variable | Type | Levels or bounds |
|---|---|---|---|
| 0 | `x1` weld type | integer (binary switch) | 0 two-sided, 1 four-sided |
| 1 | `x2` bar material | categorical | steel, cast iron, aluminum, brass (codes 00, 01, 10, 11) |
| 2 | `x3 = h` weld thickness | integer | 0.0625 to 2 in step 0.0625 (32 levels, value `16 h`) |
| 3 | `x4 = t` beam width | integer | 2 to 20 in step 0.0625 (289 levels, value `16 t`) |
| 4 | `x5 = b` beam thickness | integer | 0.0625 to 2 in step 0.0625 (32 levels, value `16 b`) |
| 5 | `x6 = l` weld length | continuous | `[0.1, 10]` |

- Cost `f = (1 + c1) h^2 (l + x1 t) + c2 t b (L + l)`.
- `sigma = 6 F L / (t^2 b)`, `delta = 4 F L^3 / (E t^3 b)`,
  `Pc = 4.013 sqrt(E G t^2 b^6 / 36) / L^2 (1 - t / (2 L) sqrt(E / (4 G)))`
  (Reklaitis et al.).
- Weld: two-sided `A = 1.414 h l`, `J = 1.414 h l ((h + t)^2 / 4 + l^2 / 12)`;
  four-sided `A = 1.414 h (t + l)`, `J = 1.414 h (h + t + l)^3 / 12`
  (Deb and Goyal, as printed, with 1.414).
- `tau' = F / A`, `M = F (L + l / 2)`, `R = sqrt(l^2 / 4 + ((h + t) / 2)^2)`,
  `tau'' = M R / J`, `tau = sqrt(tau'^2 + 2 tau' tau'' l / (2 R) + tau''^2)`
  (Reklaitis et al.).

| Constant | Value | Origin |
|---|---|---|
| `F`, `L`, `delta_max` | 6,000 lb, 14 in, 0.25 in | Deb and Goyal, section 4.4 |
| shear limit | `0.577 S` | Deb and Goyal, constraint `g4` |
| `S`, `E`, `G` (psi), `c1`, `c2` | steel 30e3, 30e6, 12e6, 0.1047, 0.0481; cast iron 8e3, 14e6, 6e6, 0.0489, 0.0224; aluminum 5e3, 10e6, 4e6, 0.5235, 0.2405; brass 8e3, 16e6, 6e6, 0.5584, 0.2566 | Deb and Goyal, table 4 (kpsi and Mpsi there); the same table in Tran, Tran and Wang (2019), table 1 |
| 1.414, 4.013, step 0.0625 | | Deb and Goyal (1.414, step); Reklaitis et al. (4.013) |
| ranges of `h`, `t`, `b` | 0.0625..2, 2..20, 0.0625..2 | Deb and Goyal (ranges of the initial population, with the step) |
| bounds of `l` | 0.1..10 | A. Tran, M. Tran and Y. Wang, "Constrained mixed-integer Gaussian mixture Bayesian optimization and its applications in designing fractal and auxetic metamaterials", Structural and Multidisciplinary Optimization (2019); also the classic welded beam's bounds of `l` |

**Verification** (`tests/round3b.rs`). Deb and Goyal's GeneAS solution
(table 5: four-sided weld, steel, `h = 0.1875`, `t = 8.25`, `b = 0.25`,
`l = 1.6849`; printed `g1 = 380.1660`, `g2 = 402.0473`, `g3 = 0.2346`,
`g4 = 0.1621`, `f = 1.9422`):

| Quantity | Printed | Harness |
|---|---:|---:|
| `g1` | 380.1660 | 380.1653 (380.1660 in single precision) |
| `g2` | 402.0473 | 402.0472 |
| `g3` | 0.2346 | 0.2344 (not reproduced) |
| `g4` | 0.1621 | 0.1339 at `l = 1.6849`; 0.1621 at `l = 1.684907`, inside the rounding interval of the printed `l` |
| `f` | 1.9422 | 1.94188 with table 4's coefficients; 1.94221 with the classic steel coefficients 0.10471 and 0.04811 |

The statements of the text (bending stress 1.3% below its limit, buckling
load 6.7% above the load, shear stress 9.4e-4 % below its limit) are
reproduced. The `g4` check discriminates: with `sqrt(2)` instead of 1.414,
or with `J = 1.414 h (t + l)^3 / 12`, `g4` falls outside the rounding
interval (2.64 and -957).

**Where the sources disagree.** Tran, Tran and Wang (2019) restate the
problem "with some slight modifications", and their code in
`facebookresearch/bo_pr` (MIT, `discrete_mixed_bo/problems/welded_beam.py`)
differs further. Compared with Deb and Goyal:

| Item | Deb and Goyal (used) | Tran et al. / bo_pr |
|---|---|---|
| four-sided `J` | `1.414 h (h + t + l)^3 / 12` | sum of the two-sided `J` of both weld pairs |
| four-sided `R` | `sqrt(l^2/4 + ((h + t)/2)^2)` | maximum over both weld pairs |
| constraints | four | five (adds `b - h >= 0`) |
| `h`, `t`, `b`, `l` | `h`, `t`, `b` multiples of 0.0625, `l` continuous | continuous (paper); bo_pr also quantizes `l` and drops the last level of each |
| `tau'` | `F / A` | bo_pr: `sqrt(F / A)` |
| brass `G` | 6e6 | bo_pr: 1e7 (both papers: 6e6) |
| `sqrt 2` | 1.414 | `sqrt(2)` |

The harness follows Deb and Goyal, whose published solution it reproduces
(with the exceptions above); only the bounds of `l` come from Tran et al.,
because the original gives none (its initial range `2 <= l <= 20` excludes
its own optimum).

### Mixed cantilever beam (`cantilever_mixed`)

**Source.** `MixedCantileverBeam` of the Surrogate Modeling Toolbox, SMT
2.15.1 (`smt/problems/mixed_cantilever_beam.py`, BSD-3-Clause, Copyright (c)
2017 SMT developers; author P. Saves), the test case of P. Saves, Y.
Diouane, N. Bartoli, T. Lefebvre and J. Morlier, "A mixed-categorical
correlation kernel for Gaussian process", Neurocomputing (2023),
arXiv:2211.08262.

**Definition** (module `cantilever`): tip deflection
`y = (P L^3) / (3 E S^2 I)` with `P = 50e3` and `E = 200e9` (SMT option
defaults), `L` in `[10, 20]` and `S` in `[1, 2]` (SMT `FloatVariable`s),
and `I` the normalized moment of inertia of one of twelve cross-sections,
SMT's `listI` = 0.0833, 0.139, 0.380, 0.0796, 0.133, 0.363, 0.0859, 0.136,
0.360, 0.0922, 0.138, 0.369. Coordinate 0 is categorical (12 levels),
coordinates 1 and 2 are `L` and `S`. Unconstrained.

SMT evaluates its categorical index `k` (0..11, labels "1".."12") as
`listI[int(k) - 1]`; for `k = 0` Python's index -1 selects the last entry.
SMT's labels are therefore shifted by one against the paper, whose levels
{1, 4, 7, 10} form one thickness group (entries 0, 3, 6, 9 of `listI`). The
harness follows SMT's design-space order and values; since the categories
are unordered, the shift only renames them.

**Verification.** `round3b_reference.py verify` (224 points: 200 random and
the 24 corners of the continuous box per category): the harness equals SMT's
formula bit for bit when `L**3` is computed with the C library's `pow`, as
the Rust port does. NumPy computes `L**3` with a SIMD power routine on hosts
with AVX-512, which differs from `pow` by one unit in the last place at a
few points; against SMT itself 220 of 224 values are identical and the rest
differ by one unit in the last place (relative difference at most 2.1e-16).
The operation order is SMT's: `(P * pow(L, 3)) / (((3 E) * (S * S)) * I)`.

### Pest control (`pest_control`)

**Source.** The pest-control benchmark of C. Oh, J. Tomczak, E. Gavves and
M. Welling, "Combinatorial Bayesian Optimization using the Graph Cartesian
Product" (NeurIPS 2019; QUVA-Lab/COMBO, BSD-2-Clause,
`COMBO/experiments/test_functions/multiple_categorical.py`), in the
fixed-seed form used by Casmopolitan (X. Wan et al., ICML 2021;
`test_funcs/pest.py`, MIT) and Bounce (L. Papenmeier, L. Nardi and M.
Poloczek, NeurIPS 2023; `bounce/util/pest_control.py`, MIT, which credits
BODi).

**Determinism.** In COMBO every beta draw comes from NumPy's global
generator, which is never seeded for the evaluation: the published form is
noisy, and the number of draws depends on the choices (a control-rate draw
only at stations with a pesticide). Casmopolitan and Bounce pass a seed, and
then every draw is `np.random.RandomState(seed).beta(...)`, a fresh
generator per call: each draw array restarts the same uniform stream, the
initial and spread fractions are the same at every evaluation, and the
control rates depend on the choices only through the shape parameter
(`beta_k` after the earlier uses of pesticide `k`). The function is then a
deterministic function of the 25 choices. Bounce's `PestControl` uses seed
0 (its default; `configs/default.gin` sets none), Casmopolitan seed 8194
(the 20th of its sorted seeds). The two implementations give the same
values (500 random points, seeds 0 and 8194). The harness uses seed 0.

**Definition** (module `pest`): 25 categorical coordinates with 5 levels in
the source's order (0 no pesticide, 1 to 4 pesticides); 100 simulations,
threshold 0.1, initial fraction `Beta(1, 30)`, spread `Beta(1, 17/3)`,
control `Beta(1, beta_k)` with `beta_k` starting at 2/7, 3/7, 3/7, 5/7 and
growing by `rate_k / 25` per use (`rate_k` = 1/7, 2.5/7, 2/7, 0.5/7), prices
1.0, 0.8, 0.7, 0.5 and maximal discounts 0.2, 0.3, 0.3, 0; the value is the
paid price plus the summed share of simulations above the threshold, all as
in the reference code. Unconstrained.

**Evaluation path.** Ported to Rust with identical values; no Python worker
is needed. The port includes NumPy's legacy `RandomState` (module
`legacy_random`: MT19937 with `mt19937_seed`, the polar Gaussian with its
cached second value, `legacy_standard_gamma` and `legacy_beta` with Johnk's
algorithm), ported from NumPy 2.5.3 (BSD-3-Clause). The draws are computed
once per problem instance. `round3b_reference.py` holds a faithful,
attributed copy of Bounce's function; `verify` compares the harness with it
at 503 points (500 random, all-none, all-4, a cyclic pattern): bit-identical.
`tests/round3b.rs` checks NumPy's first draws and reference values for seeds
0, 8194 and 1.

### Analysis

`analyze.py --round 3b` writes the tables of `--round 3` for this batch
alone (`results/round3b/report.md`); its decision table notes that the
decision is taken on the union. `analyze.py --round 3u --raw
<round 3>,<round 3b>` implements ROUND3B.md section 4: cells of both files,
best incumbent, Mann–Whitney U tests with Holm correction over all union
cells of the candidate, ranks and mean ranks over all union cells, G1 (at
least 3 winning cells on at least 2 problems), G2 (mean rank strictly better
than the best incumbent's, and below every incumbent's in at least 95% of
10,000 resamples, NumPy `default_rng(3)`), Include, Watch and Reject, per-cell
medians, and the descriptive split into cells of problems with a categorical
coordinate (`field_service`, `jobshop` and the three round-3b problems) and
the rest (`results/round3b/report-union.md`). The two files must share root
seed and minimum run and must not hold the same problem. `--round 3` is
unchanged.

## Round 3, convergence test

[ROUND3C.md](ROUND3C.md) runs CatCMAwM and the core solvers on three round-3
problems with larger total budgets and asks whether the candidate is
superior on any of them at the end of the budget. Solvers, protocol, the
bridge and the file formats are those of "Round 3: campaign"; the root seed
is 5, there are 10 repetitions, and each problem has its own raw file.

### Reproduce

From this directory, one problem after the other; the Rust runner and the
driver never write a raw file at the same time:

```sh
BIN=target/release/candidate-harness   # or $CARGO_TARGET_DIR/release/...
for spec in phased_array:100000 field_service:200000 water_network:20000; do
    p=${spec%%:*}; t=${spec##*:}
    $BIN run --protocols T --problems $p --solvers round3 --totals $t \
        --root-seed 5 --min-run 100 --reps 10 --threads 2 \
        --out results/round3c/$p.tsv
    types=results/round3/types
    [ $p = water_network ] && types=results/round3c/types
    python3 run_catcmawm.py run --problems $p --types $types \
        --out results/round3c/$p.tsv --totals $t --root-seed 5 \
        --min-run 100 --reps 10 --workers 2
done
python3 convergence_round3c.py          # writes results/round3c/report.md
```

`results/round3c/types/water_network.json` types the pump-priority switch as
an integer for the candidate; [DEVIATIONS.md](DEVIATIONS.md) says why.
