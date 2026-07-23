# Native Rust GTOP benchmark results

This report collects the recorded GTOP tables in one Markdown document. All
three tests use relaxed stopping values of approximately
`1.005 * absolute_best` for positive objectives and the corresponding 0.5%
relaxation toward zero for negative objectives. The raw TSV files preserve
every individual experiment.

The problem definitions and putative best solutions originate from ESA's
[Global Trajectory Optimisation Problems database](https://www.esa.int/gsp/ACT/projects/gtop/).
The unusually difficult Tandem case is documented separately on ESA's
[TandEM problem page](https://www.esa.int/gsp/ACT/projects/gtop/tandem/),
including its 18-variable box bounds and selectable fly-by sequences.

## Coordinated retry

Messenger Full remains excluded because its full 100-run measurement is too
expensive for this test. Tandem was measured separately with the same
configuration and is included here.

| Problem | Runs | Absolute best | Stop value | Success rate | Mean wall time | Wall-time sdev |
|---|---:|---:|---:|---:|---:|---:|
| Cassini1 | 100 | 4.9307 | 4.95535 | 100% | 0.25 s | 0.12 s |
| Cassini2 | 100 | 8.383 | 8.42491 | 100% | 4.16 s | 2.26 s |
| Gtoc1 | 100 | -1581950 | -1574080 | 100% | 3.33 s | 2.87 s |
| Messenger | 100 | 8.6299 | 8.673 | 100% | 3.36 s | 1.83 s |
| Rosetta | 100 | 1.3433 | 1.35 | 100% | 4.58 s | 1.73 s |
| Tandem | 100 | -1500.46 | -1493 | 85% | 40.21 s | 39.11 s |
| Sagas | 100 | 18.188 | 18.279 | 100% | 0.90 s | 0.95 s |

Raw data: [`benchmark_gtop_100_raw.tsv`](benchmark_gtop_100_raw.tsv) and
[`benchmark_gtop_tandem_100_raw.tsv`](benchmark_gtop_tandem_100_raw.tsv).
The separate
[`benchmark_gtop_tandem_100_metadata.json`](benchmark_gtop_tandem_100_metadata.json)
records the slow-run configuration and total invocation time.

Configuration:

- 32 native retry threads and 100 independent experiments per problem
- 1,500 initial evaluations per retry
- advanced maximum evaluation factor 50
- diversity checkpoint interval 100 retries
- optimizer sequence: 40% Differential Evolution, then 60% active CMA-ES
- per-problem retry caps and value limits from
  `examples/src/benchmark_gtop.rs`

The coordinated test has an adaptive budget, not the 240,000-evaluation
allowance used by the basic-retry and external-library comparison. A retry's
limit grows linearly from 1,500 to 75,000 evaluations as the run ID approaches
the per-problem retry cap. The exact theoretical ceilings and actual recorded
means are:

| Problem | Retry cap | Exact configured ceiling | Mean actual evaluations | Actual range |
|---|---:|---:|---:|---:|
| Cassini1 | 4,000 | 153,000,000 | 1,525,802 | 369,151–4,664,857 |
| Cassini2 | 6,000 | 229,500,000 | 19,890,045 | 5,219,854–54,945,340 |
| Gtoc1 | 10,000 | 382,500,000 | 14,729,200 | 1,339,960–71,104,757 |
| Messenger | 8,000 | 306,000,000 | 19,603,567 | 3,592,572–66,816,828 |
| Rosetta | 4,000 | 153,000,000 | 21,406,666 | 6,733,813–43,991,489 |
| Tandem | 20,000 | 765,000,000 | 230,727,025 | 6,962,364–707,799,769 |
| Sagas | 4,000 | 153,000,000 | 8,978,824 | 1,190,960–56,583,789 |

Every run consumed less than its theoretical ceiling. The 15 Tandem failures
are included in its mean and sdev. Across all 100 Tandem runs, the mean final
optimum was -1488.503885 (population sdev 21.850556); the best was
-1500.468716.

The original Python/C++ fcmaes
[Performance report](https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Performance.adoc)
records 81% Tandem success, 166.92 s mean wall time, and 147.87 s wall-time
sdev on the same AMD 9950X class of CPU with 32 parallel Python processes. The
native Rust run records 85%, 40.21 s, and 39.11 s with 32 native retry threads.
The close success rates are consistent with the implementations using the same
adaptive search policy; the wall times also reflect their different process
and thread execution models.

The substantially larger actual budgets explain why coordinated retry has much
better success rates than the 240,000-evaluation comparison. This table is a
quality reference, not an equal-budget speed comparison.

## BiteOpt basic retry

This benchmark includes Tandem, excludes Messenger Full, and uses 24
independent BiteOpt retries of at most 10,000 evaluations on 24 native worker
threads for every experiment.

| Problem | Runs | Absolute best | Stop value | Success rate | Mean optimum | Sdev optimum | Mean wall time | Wall-time sdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cassini1 | 100 | 4.9307 | 4.95535 | 57% | 5.094738 | 0.180157 | 0.05 s | 0.00 s |
| Cassini2 | 100 | 8.383 | 8.42491 | 0% | 13.778437 | 2.559722 | 0.07 s | 0.00 s |
| Gtoc1 | 100 | -1581950 | -1574080 | 0% | -1103593.281024 | 142070.428562 | 0.08 s | 0.00 s |
| Messenger | 100 | 8.6299 | 8.673 | 0% | 11.896085 | 0.930384 | 0.06 s | 0.00 s |
| Rosetta | 100 | 1.3433 | 1.35 | 0% | 4.358705 | 1.120562 | 0.07 s | 0.00 s |
| Tandem | 100 | -1500.46 | -1493 | 0% | -419.987695 | 168.414195 | 0.07 s | 0.00 s |
| Sagas | 100 | 18.188 | 18.279 | 2% | 83.081492 | 52.445301 | 0.03 s | 0.00 s |

Raw data:
[`benchmark_biteopt_gtop_rust_100_raw.tsv`](benchmark_biteopt_gtop_rust_100_raw.tsv).

## DE→CMA basic retry

This test has the same 100 experiments, 24 workers, 24 retries, and 10,000
evaluations per retry as the BiteOpt test. Each retry assigns 4,000 evaluations
to Differential Evolution and 6,000 to active CMA-ES.

| Problem | Runs | Absolute best | Stop value | Success rate | Mean optimum | Sdev optimum | Mean wall time | Wall-time sdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cassini1 | 100 | 4.9307 | 4.95535 | 22% | 5.343353 | 0.889715 | 0.05 s | 0.00 s |
| Cassini2 | 100 | 8.383 | 8.42491 | 0% | 13.324033 | 2.034216 | 0.07 s | 0.00 s |
| Gtoc1 | 100 | -1581950 | -1574080 | 0% | -1102350.655440 | 143746.821645 | 0.07 s | 0.00 s |
| Messenger | 100 | 8.6299 | 8.673 | 0% | 11.042875 | 0.750887 | 0.05 s | 0.00 s |
| Rosetta | 100 | 1.3433 | 1.35 | 0% | 2.705204 | 0.846640 | 0.07 s | 0.00 s |
| Tandem | 100 | -1500.46 | -1493 | 0% | -490.089599 | 181.284594 | 0.06 s | 0.00 s |
| Sagas | 100 | 18.188 | 18.279 | 0% | 111.796178 | 68.411041 | 0.03 s | 0.00 s |

Raw data:
[`benchmark_de_cma_gtop_rust_100_raw.tsv`](benchmark_de_cma_gtop_rust_100_raw.tsv).

## Statistics and environment

Means and standard deviations include every experiment, including failures.
Standard deviations use the population definition (`ddof=0`), matching NumPy's
default.

- CPU: AMD Ryzen 9 9950X, 16 physical cores / 32 logical CPUs
- Compiler: rustc 1.93.0 for the original six-problem run and rustc 1.97.1 for
  the later Tandem run; Cargo release profile
- OS: Linux 6.8.0-136-generic x86_64
- Date: 2026-07-22 for the original run; 2026-07-23 for Tandem

Rust workers are native threads. Each top-level experiment is run sequentially
and has exclusive use of its configured retry workers.

## Reproduce

From the repository root:

```bash
cargo run --release -p fcmaes-examples --bin benchmark-gtop -- \
  --runs 100 --workers 32 --seed 1 \
  --raw-output benchmarks/benchmark_gtop_100_raw.tsv

python3 benchmarks/run_coordinated_tandem.py

cargo run --release -p fcmaes-examples --bin benchmark-biteopt-gtop -- \
  --algo biteopt --runs 100 --workers 24 --retries 24 \
  --evaluations 10000 --seed 1 \
  --raw-output benchmarks/benchmark_biteopt_gtop_rust_100_raw.tsv

cargo run --release -p fcmaes-examples --bin benchmark-biteopt-gtop -- \
  --algo de_cma --runs 100 --workers 24 --retries 24 \
  --evaluations 10000 --seed 1 \
  --raw-output benchmarks/benchmark_de_cma_gtop_rust_100_raw.tsv
```

Each binary prints the corresponding Markdown table. `--table-output PATH` can
also write that table to a chosen `.md` file. The Tandem driver records each
experiment in a resumable shard before combining the final raw file. For
coordinated retry, `--include-slow` adds Tandem and Messenger Full;
`--problem NAME` runs one case explicitly.
