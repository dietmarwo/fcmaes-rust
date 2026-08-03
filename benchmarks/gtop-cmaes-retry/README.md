# Single-threaded CMA-ES through `fcmaes` retry on GTOP

This dependency-isolated benchmark answers a practical question that a
same-implementation microbenchmark cannot:

> Can an existing serial optimizer use all physical CPU cores through the
> generic `fcmaes_core::retry` adapter, and how much does this improve the
> returned solution distribution within the same user-visible wall time?

The optimizer is the external [`cmaes`](https://crates.io/crates/cmaes) 0.2.2
crate. That release offers `run_parallel()` and `next_parallel()`, which
evaluate its objective internally with Rayon, but it does **not** expose a
public ask/tell boundary for externally evaluating and later returning a
population. This benchmark deliberately calls each instance's serial `run()`
path. Parallelism exists only between independent starts scheduled by
`fcmaes_core::retry`; there is no nested population parallelism.

fcmaes-core's own population optimizers do expose ask/tell or equivalent batch
evaluation. That is a complementary fixed-wall-time strategy for expensive
objectives; this experiment isolates retry instead of mixing the two sources
of parallelism.

The external crate is deliberately confined to this standalone Cargo
workspace. It is not a dependency of `fcmaes-core`, the Python package, or the
root workspace.

## Primary equal-wall experiment

| Arm | Restart lanes | Workers | Question answered |
|---|---:|---:|---|
| External CMA-ES, serial | 1 | 1 | What solution quality can one CPU core return within four seconds? |
| External CMA-ES through `fcmaes` retry | 16 | 16 | What quality can the same four-second wait return when independent serial searches use every physical core? |

An individual CMA-ES call can terminate protectively before four seconds. Each
lane therefore restarts it immediately with the next deterministic seed and
retains the lane's best result until the deadline. This keeps the serial lane
at approximately one active core and lets the retry arm keep the physical cores
busy. The outer retry call aggregates the best lane result.

The publication preset resolves the retry arm to the number of physical cores.
On the reference Ryzen 9 9950X this is 16, even though the CPU exposes 32
logical processors. Every top-level pair alternates arm order to reduce drift.
Across 100 paired seeds, the primary statistics are:

- mean and population standard deviation of the best objective, where smaller
  is better;
- the paired fraction in which retry returns a strictly better result; and
- measured wall-time mean/sdev as the fairness audit.

Optimizer-start counts, evaluations, CPU time, and active cores document how
much additional compute the retry arm spends. This is intentionally an
equal-wall, fixed-machine comparison, not an equal-CPU efficiency claim.

The two arms share all search inputs:

- schedule-independent retry IDs and seeds;
- the same uniformly sampled initial means in normalized `[-1, 1]` space;
- active negative CMA weights, population, initial sigma, and box clamping;
- the same four-second lane deadline and a high evaluation safety ceiling; and
- serial objective evaluation inside every CMA-ES instance.

Lane zero in the retry arm uses the serial arm's seed stream; the remaining
lanes add independent streams. Contention can change how many generations each
lane completes, so paired equality is neither expected nor the goal. The
quality distribution at matched elapsed time is the goal.

## Secondary diagnostic phases

The harness retains two bounded diagnostics, but neither supplies the primary
practical conclusion.

`fixed-work` disables target stopping and executes the complete retry list. It
reports paired wall-time speedup, parallel efficiency, average active cores,
and an explicit same-work audit. Numerical protective stops inside CMA-ES stay
enabled where they cannot safely be disabled, so actual calls may be slightly
below the nominal per-start cap; the paired external arms must still report
identical actual work.

`target` uses the relaxed targets from the native
[GTOP report](../benchmark_gtop.md). It stops scheduling new starts once a
completed retry reaches the target; starts already running are allowed to
finish. The primary outputs are target success and user-visible call wall
time, including that worker drain for a success or exhaustion for a failure.
Evaluation counts are retained as resource accounting, but they are not a
substitute for the time experienced by the user.

The existing coordinated DE→CMA results remain a separate system-level
reference. They change the optimizer sequence, budgets, bounds, and crossover
policy and therefore do not belong in the controlled same-work speedup table.

## Publication result

The [100-pair publication bundle](results/equal-wall-100-v2/comparison.md) was
measured on a Ryzen 9 9950X with 16 physical cores. Both arms received 4,000 ms
per pair. Values are best-objective mean and population sdev; smaller is
better. `W/T/L` counts paired retry wins, ties, and losses.

| Problem | Serial mean ± sdev | Retry mean ± sdev | Serial / retry success | Retry W/T/L |
|---|---:|---:|---:|---:|
| Cassini1 | 5.166778 ± 0.191308 | 4.930709 ± 0.000002 | 38/100 / 100/100 | 86/12/2 |
| Cassini2 | 20.395034 ± 2.585887 | 15.140516 ± 1.979760 | 0/100 / 0/100 | 96/3/1 |
| GTOC1 | −1,034,840.69 ± 140,299.40 | −1,295,551.11 ± 104,974.27 | 0/100 / 0/100 | 94/6/0 |
| Messenger | 13.950756 ± 1.492585 | 11.300458 ± 0.694519 | 0/100 / 0/100 | 95/5/0 |
| Rosetta | 6.779610 ± 2.263446 | 2.816424 ± 0.800562 | 0/100 / 0/100 | 93/4/3 |
| SAGAS | 171.742982 ± 35.723425 | 120.593375 ± 48.152676 | 0/100 / 0/100 | 92/7/1 |
| Tandem | −201.495632 ± 206.305874 | −598.689263 ± 216.831269 | 0/100 / 0/100 | 93/7/0 |

Retry improves the mean on all seven problems and wins 86–96 pairs. It reduces
sdev on five problems, but not on SAGAS or Tandem: more independent starts can
sample both a much better basin and the usual basin, increasing dispersion even
while improving the mean. Only Cassini1 reaches its target in this short
protocol. The result therefore demonstrates better equal-wall opportunity, not
that four seconds of CMA-ES solves the GTOP suite.

### Equal-wall fairness audit

The wall-time distributions overlap to within measurement and scheduling
noise. Retry spends the same user-visible wait while keeping the machine's
physical cores active.

| Problem | Serial wall mean ± sdev (s) | Retry wall mean ± sdev (s) | Serial active cores | Retry active cores |
|---|---:|---:|---:|---:|
| Cassini1 | 4.000109 ± 0.000112 | 4.000491 ± 0.000468 | 1.00 | 16.00 |
| Cassini2 | 4.000150 ± 0.000039 | 4.000503 ± 0.000139 | 1.00 | 16.00 |
| GTOC1 | 4.000147 ± 0.000140 | 4.000440 ± 0.000125 | 1.00 | 15.99 |
| Messenger | 4.000135 ± 0.000091 | 4.000469 ± 0.000257 | 1.00 | 16.00 |
| Rosetta | 4.000140 ± 0.000071 | 4.000523 ± 0.000195 | 1.00 | 16.00 |
| SAGAS | 4.000114 ± 0.000090 | 4.000479 ± 0.000269 | 1.00 | 16.00 |
| Tandem | 4.000146 ± 0.000083 | 4.000510 ± 0.000423 | 1.00 | 16.00 |

### Work behind the equal wait

The extra work is the mechanism, not a hidden efficiency claim. Counts below
are per-pair means; every CMA-ES instance itself remains single-threaded.

| Problem | Serial starts | Retry starts | Serial evaluations | Retry evaluations |
|---|---:|---:|---:|---:|
| Cassini1 | 141.9 | 2,114.4 | 1,115,347 | 16,598,162 |
| Cassini2 | 10.1 | 151.2 | 819,929 | 12,177,224 |
| GTOC1 | 74.7 | 1,106.4 | 789,225 | 11,711,201 |
| Messenger | 16.9 | 252.0 | 974,569 | 14,512,197 |
| Rosetta | 9.0 | 133.1 | 800,001 | 11,902,792 |
| SAGAS | 67.4 | 1,005.6 | 1,702,941 | 25,383,663 |
| Tandem | 14.8 | 221.6 | 850,532 | 12,774,524 |

The final 1,400 rows represent 5,600.436 s (93.341 min) of sequential arm
wall time and 13.221 process-CPU hours.

## Retry is an external-optimizer adapter

`fcmaes_core::retry` is not restricted to optimizers implemented by fcmaes.
Its closure receives the objective, bounds, budget, initial-guess metadata,
and a schedule-independent run seed, then returns the candidate, objective,
and actual evaluation count. This benchmark supplies that closure for the
external `cmaes` crate without adding `cmaes` to `fcmaes-core`.

The same pattern can wrap another single-threaded Rust optimizer, or an FFI
adapter to an external solver, when each instance:

- owns independent mutable state and can run concurrently with other instances;
- uses `context.run_seed` when deterministic replay across worker counts matters;
- respects `context.bounds` and the requested evaluation or deadline policy;
- reports its actual objective-call count; and
- returns a finite, dimensionally valid `RetryRunResult`.

Serial Nelder–Mead, Bayesian optimization, gradient/local solvers, or
domain-specific search procedures can therefore use fcmaes retry without
becoming fcmaes-core algorithms. Avoid also enabling an external optimizer's
own worker pool unless nested parallelism and oversubscription are deliberate.
The canonical [adapter contract](../../docs/retry.md#external-optimizer-and-adapter-contract)
lists the exact obligations.

For an optimizer that exposes ask/tell, the adapter can instead parallelize
the candidates within one run, or combine a small number of retry lanes with a
small evaluation pool per lane. Keep the product of outer lanes and inner
workers within the available CPU allocation. The
[optimizer parallelism guide](../../docs/optimizers.md#parallel-evaluation-asktell-and-retry)
describes this choice.

The first SAGAS pass exposed and incorrectly accepted a negative
time-to-50-AU result. The port now penalizes every non-finite or negative travel
time, and SAGAS was rerun with the same seeds. The final bundle's
[provenance note](results/equal-wall-100-v2/PROVENANCE.md) documents the guard,
replacement, and row validation. The closest valid retry result was 18.912168,
above the 18.279 target. Discovering the anomaly and repeating SAGAS added
800.059 s (13.334 min) and 1.889 process-CPU hours to the research run; those
superseded rows are not part of the final statistics.

## Reproduce

Verify the adapter contract and run the bounded smoke campaign:

```bash
cd benchmarks/gtop-cmaes-retry
cargo run --release --locked -- --mode verify
cargo run --release --locked -- \
  --mode campaign --preset smoke \
  --output results/my-smoke
```

Run the publication protocol:

```bash
cargo run --release --locked -- \
  --mode campaign --preset publication \
  --output results/my-equal-wall-100
```

The publication preset selects all seven GTOP problems in the parent report
except the very slow Messenger Full: Cassini1, Cassini2, GTOC1, Messenger,
Rosetta, Sagas, and Tandem. It executes 100 pairs with 4,000 ms per arm, one
serial lane versus one lane per physical core, and a one-billion-evaluation
per-start safety ceiling. Preserve a partial campaign with `--resume`.
Individual fields can be overridden, for example:

```bash
cargo run --release --locked -- \
  --mode campaign --preset publication \
  --phases fixed-work --problems cassini1 \
  --runs 5 --retries 40 --evaluations 2000 \
  --workers 1,2,4,8,16 \
  --output results/pilot-v1
```

Render an existing result bundle without executing optimization:

```bash
cargo run --release --locked -- \
  --mode report --output results/equal-wall-100-v2
```

Each bundle contains:

- `results.csv`: one authoritative row per phase, arm, problem, seed, and
  worker count;
- `run.json`: machine topology and exact campaign protocol; and
- `comparison.md`: generated equal-wall quality and work-audit tables plus any
  requested diagnostic phases.

The checked-in
[equal-wall smoke report](results/equal-wall-smoke-v2/comparison.md) validates
deadline consumption, reporting, and the 16-lane topology only. Its three
seeds and 250 ms deadline are intentionally too small for a publication claim.
It measures 0.250 s for both arms, about 1.00 versus 15.97 active cores, and
therefore catches the early-termination bias that a naive one-call protocol
would miss.

The checked-in [five-pair scaling pilot](results/scaling-pilot-v1/comparison.md)
uses all 16 physical cores on the reference machine. It is bounded evidence
for retry scheduling, not target-attainment evidence: 40 starts with 2,000
evaluations each are deliberately below the GTOP quality regime. Run it with:

```bash
cargo run --release --locked -- \
  --mode campaign --preset pilot \
  --output results/scaling-pilot-v1
```

All 60 paired worker-count comparisons reproduce identical work. At 16
workers, mean wall-time speedup is 12.23× for Cassini1, 11.80× for Rosetta, and
11.18× for Tandem. The corresponding mean active-core measurements are 13.61,
13.27, and 12.58; diminishing efficiency at the last scaling point is visible
rather than hidden.

## Interpretation

The defensible practical claim comes from the paired equal-wall distribution:
it measures what a user receives after the same wait on the same machine. More
cores necessarily mean more aggregate CPU work; that is the mechanism being
evaluated, not a confound to hide. The fixed-work pilot remains useful only for
checking retry scheduling. The native CMA-ES arm changes implementation, and
its outcome should be read beside the separate
[controlled implementation diagnostic](../cmaes-implementation/README.md).

Wall times depend on the CPU, compiler, operating system, thermals, and
background load. Report the machine and repeat paired runs before treating a
speedup as publication evidence.
