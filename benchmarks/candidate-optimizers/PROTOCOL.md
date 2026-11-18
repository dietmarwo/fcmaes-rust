# Candidate optimizers: survey and benchmark protocol

Declared on 18 November 2026, before any campaign result exists. Gates and
thresholds in this file are fixed; a deviation is recorded in the report and
never applied silently.

**Question.** Which gradient-free algorithms that `fcmaes-core` does not
contain would add something on real-world problems, and should therefore be
implemented in core?

**Scope.** Single-objective, box-bounded, continuous, deterministic
objectives. Multi-objective, mixed-integer and noisy optimization are out of
scope for this round and listed at the end.

## 1. What fcmaes already has

| Family | fcmaes-rust core | Python fcmaes only |
|---|---|---|
| Differential evolution | DE (fcmaes variant: age-based reinitialization, mixed-integer mutation) | GCL-DE, LDE wrappers |
| Covariance adaptation | active CMA-ES, CR-FM-NES | sep-CMA, CMA with margin, lightweight CMA (wrappers around `cmaes`) |
| Gradient estimation | PGPE | |
| Self-adaptive ensembles | BiteOpt, DeepBiteOpt | CSMA-ES through BiteOpt |
| Annealing | Dual Annealing | |
| Meta level | retry, coordinated (advanced) retry, DE→CMA sequences | wrappers for SciPy, NLopt, pygmo |
| Multi-objective / QD | MODE, MAP-Elites, Diversifier | |
| Surrogates | separate `fcmaes-surrogate` crate | |

The [optimizer boundary](../../docs/optimizer-boundary.md) already keeps local
derivative-free refiners, Bayesian optimization and gradient methods outside
core. Earlier comparisons in this repository measured third-party
implementations of L-SHADE, BIPOP-CMA-ES, PSO and DE/best/1/bin on GTOP
([optimizer comparison](../optimizer-comparison/comparison.md)). Those rows
used external crates with their own defaults and parallel models, so they do
not settle whether a faithful native implementation belongs in core.

## 2. Survey of missing algorithms

| Algorithm | Family | Why it is interesting | Decision for this round |
|---|---|---|---|
| **L-SHADE** (Tanabe and Fukunaga, 2014) | adaptive DE | Success-history adaptation of F and CR, external archive, linear population reduction; root of most CEC winners since 2014 | **Benchmark** |
| **jSO** (Brest, Maučec and Bošković, 2017) | adaptive DE | L-SHADE with weighted current-to-pbest mutation and staged parameter limits; a standard strong baseline | **Benchmark** |
| **L-SRTDE** (Stanovov and Semenkin, 2024) | adaptive DE | Success-rate-based F and greediness, two populations (newest and top); CEC 2024 competition entry | **Benchmark**, implemented from the published description only (the reference code is GPL-3.0 and is not consulted) |
| **BIPOP-aCMA-ES** (Hansen, 2009) | CMA-ES restarts | Interleaves large-population and small-population restarts; had the best equal-budget Tandem mean in the earlier external comparison | **Benchmark**, as a restart schedule over core `Cmaes` |
| **AMaLGaM-IDEA** (Bosman, Grahl and Thierens, 2013) | estimation of distribution | Maximum-likelihood Gaussian with anticipated mean shift and adaptive variance scaling; a different model-building principle from CMA-ES | **Benchmark** |
| NL-SHADE-RSP, NL-SHADE-LBC, RDE, RDEx | adaptive DE | CEC 2021 to 2025 entries; incremental over L-SHADE/jSO | Deferred: add only if the DE family passes a gate |
| LRA-CMA-ES (Nomura, Akimoto and Ono, 2023) | CMA-ES | Learning-rate adaptation for multimodal and noisy problems without large populations | Deferred: needs changes inside core `Cmaes` |
| sep-CMA-ES, VD-CMA, LM-CMA, LM-MA-ES | large-scale ES | Linear or limited-memory covariance models | Not benchmarked: CR-FM-NES covers the large-scale niche, and the problems here have at most 114 variables |
| xNES, SNES | natural evolution strategies | Predecessors of CR-FM-NES | Not benchmarked: superseded in core |
| CMA-ES with margin, CatCMA | mixed-integer ES | Integer and categorical handling | Out of scope (mixed-integer round) |
| SPSO-2011, CLPSO | particle swarm | Popular, simple | Not benchmarked: PSO was last on all seven GTOP problems in the earlier comparison |
| BOBYQA, NEWUOA, COBYLA, Nelder–Mead | local model-based and simplex | Very efficient on smooth problems with tiny budgets | Outside the core boundary; the DE→Nelder–Mead experiment found no general advantage |
| MADS (NOMAD), DIRECT, MCS | direct search, partitioning | Deterministic, constraint-aware | Not benchmarked: local or low-dimensional by design; run them through the retry adapter if needed |
| HillVallEA, RS-CMSA | niching | Find several optima | Different task (multimodal enumeration); MAP-Elites and Diversifier cover diversity |
| Bayesian optimization, TPE, SMAC | surrogate | Tiny budgets | Outside the core boundary (BO-craft, `fcmaes-surrogate`) |
| GWO, WOA, ABC and other metaphor heuristics | swarm | Widely cited | Not benchmarked: no consistent evidence of an advantage over DE or CMA-ES baselines |

## 3. Benchmark design

### 3.1 Solvers

Six incumbents from `fcmaes-core`, with library defaults:
BiteOpt, DE, active CMA-ES, CR-FM-NES, Dual Annealing, and the DE→CMA
sequence used by the GTOP benchmarks. PGPE is left out because it targets
high-dimensional noisy policy search.

Five candidates, with the parameter values published by their authors:

| Candidate | Fixed parameters |
|---|---|
| L-SHADE | initial population 18·D, final 4, memory H = 6, p = 0.11, archive rate 2.6 |
| jSO | initial population 25·ln(D)·√D, final 4, H = 5, p from 0.25 down to 0.125, archive rate 1.0, published F and CR stage limits |
| L-SRTDE | initial population 20·D, final 4, H = 5, mean F = 0.4 + 0.25·tanh(5·SR), pb = 0.7·exp(−7·SR), rank-selection pressure 3 |
| BIPOP-aCMA-ES | default λ = 4 + ⌊3 ln D⌋, large regime doubles λ per restart, small regime as published, initial σ = 0.3 of the box width |
| AMaLGaM-IDEA | full covariance, τ = 0.35, base population 17 + 3·D^1.5, published variance-scaling and mean-shift constants, population doubling on restart |

No candidate parameter is tuned on the benchmark problems. Every solver runs
through one interface (`solver-api`), on one thread, behind a guard that
refuses evaluations beyond the budget or outside the box and records the best
value itself.

### 3.2 Problems

Thirteen real-world problems from five domains, all taken from code in this
repository:

| Problem | Domain | D | Source |
|---|---|---:|---|
| Cassini1 | space trajectories | 6 | `fcmaes-gtop` |
| GTOC1 | space trajectories | 8 | `fcmaes-gtop` |
| Sagas | space trajectories | 12 | `fcmaes-gtop` |
| Messenger reduced | space trajectories | 18 | `fcmaes-gtop` |
| Tandem (constrained instance of the examples registry) | space trajectories | 18 | `fcmaes-gtop` |
| Cassini2 | space trajectories | 22 | `fcmaes-gtop` |
| Rosetta | space trajectories | 22 | `fcmaes-gtop` |
| Messenger full | space trajectories | 26 | `fcmaes-gtop` |
| Optical lens design (scalar cost of the tutorial) | optics | 11 | `tutorials/optical-lens-design` |
| Band-pass filter design | analog circuits | 5 | `tutorials/sindr-circuit-design` |
| Phased-array geometry | antennas | 96 | `tutorials/phased-array-codebook` |
| Lennard-Jones cluster, 13 atoms | chemistry | 39 | `foundations` suite |
| Lennard-Jones cluster, 38 atoms | chemistry | 114 | `foundations` suite |

Eight of the thirteen problems are GTOP problems, on which the fcmaes
defaults were developed. This favors the incumbents and makes the gates
conservative.

If a problem cannot be built in the isolated workspace, it is dropped and the
reason is recorded. The campaign needs at least eleven problems and at least
three outside space trajectories; otherwise it stops.

### 3.3 Budgets and protocols

The evaluation budget B depends only on the measured cost of one evaluation
(mean over 2,000 uniform random points, one thread):

| Mean evaluation time | B |
|---|---:|
| up to 10 µs | 200,000 |
| up to 50 µs | 100,000 |
| up to 250 µs | 40,000 |
| above 250 µs | 16,000 |

Each solver is measured under two protocols, because fcmaes is normally used
with restarts and several candidates tie their schedules to the budget:

- **S (single run):** one run with budget B.
- **R (restarts):** eight independent runs with budget B/8 each; the best
  value counts.

A *cell* is one problem under one protocol (26 cells). Each solver gets 20
independent repetitions per cell. Seeds derive from root seed 1, the problem,
the protocol, the repetition and the restart index, and are the same for all
solvers.

**Compute rule.** The host has two cores. If the smoke campaign projects more
than 150 minutes for the full campaign, repetitions drop uniformly from 20 to
12. This is decided from timings alone, before any result is examined.

### 3.4 Validation before the campaign

Every candidate must pass, in its own crate:

- contract tests: budgets of 1, 7 and 1,000 evaluations are never exceeded,
  no point leaves the box, and a seed reproduces its run;
- sphere, 10 variables: below 1e-8 within 50,000 evaluations;
- Rosenbrock, 10 variables: below 1e-6 within 200,000 evaluations in at
  least three of five seeds.

A candidate that fails validation is reported as failed and does not enter
the campaign. An independent review compares each implementation with its
published description before the campaign starts.

### 3.5 Statistics

- The result of a repetition is the best objective value recorded by the
  guard.
- In each cell, the *best incumbent* is the incumbent with the lowest median
  (ties: lower mean).
- A candidate is compared with the best incumbent by a two-sided
  Mann–Whitney U test. P-values are Holm-corrected over the candidate's
  cells; α = 0.05. The outcome is a **win** (significantly better), a
  **loss** (significantly worse) or a **tie**.
- In each cell, all eleven solvers are ranked by median (average ranks for
  ties). A solver's *mean rank* is its average over all cells.

### 3.6 Inclusion gates

| Gate | Condition |
|---|---|
| G1, complementarity | at least 3 winning cells, on at least 2 different problems |
| G2, generalist | mean rank strictly better than the best incumbent's mean rank |

| Decision | Rule |
|---|---|
| **Include** in `fcmaes-core` | G1 or G2 holds, and the guard recorded no contract violation |
| **Watch** | at least one winning cell, but neither gate |
| **Reject** for now | no winning cell |

An *include* decision means: implement in core with the usual ask/tell
interface, documentation and tests. The benchmark crates stay outside the
root workspace and outside the dependency graph of `fcmaes-core`.

### 3.7 Reporting

- Raw rows (one per repetition) are the authoritative record:
  solver, problem, protocol, repetition, seed, budget, evaluations, best
  value, violations and wall time.
- Failed, dropped and violating rows are kept.
- The report gives per-cell medians, the win/loss/tie table, mean ranks and
  the decision per candidate, with limits: 20 repetitions, two budget
  shapes, one host, published defaults only.

## 4. Out of scope for this round

- **Multi-objective:** NSGA-III, MOEA/D, SMS-EMOA against MODE.
- **Mixed-integer:** CMA-ES with margin, CatCMA against DE's integer
  handling.
- **Noisy objectives:** LRA-CMA-ES, uncertainty handling.
- **Large scale** (thousands of variables): LM-MA-ES, VD-CMA against
  CR-FM-NES and PGPE.

## 5. Sources

- R. Tanabe, A. Fukunaga, "Improving the search performance of SHADE using
  linear population size reduction", CEC 2014.
- J. Brest, M. S. Maučec, B. Bošković, "Single objective real-parameter
  optimization: algorithm jSO", CEC 2017.
- V. Stanovov, E. Semenkin, "Success rate-based adaptive differential
  evolution L-SRTDE for CEC 2024 competition", CEC 2024; and "Adaptation of
  the scaling factor based on the success rate in differential evolution",
  Mathematics 12(4):516, 2024.
- J. Brest, M. S. Maučec, "Comparative study of modern differential evolution
  algorithms: perspectives on mechanisms and performance", Mathematics
  13(10):1556, 2025.
- N. Hansen, "Benchmarking a BI-population CMA-ES on the BBOB-2009 function
  testbed", GECCO 2009 companion.
- P. A. N. Bosman, J. Grahl, D. Thierens, "Benchmarking parameter-free
  AMaLGaM on functions with and without noise", Evolutionary Computation
  21(3), 2013.
- D. Brockhoff, T. Villain, "Benchmarking Powell's legacy", GECCO 2025
  companion.
