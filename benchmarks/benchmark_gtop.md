# Rust GTOP coordinated retry benchmark

This native Rust run uses relaxed stopping values of approximately
`1.005 * absolute_best` for positive objectives and the corresponding 0.5%
relaxation toward zero for the negative GTOC1 objective.

Tandem and Messenger Full were intentionally excluded because their full
100-run measurements are too expensive for this quick test.

## Results

| Problem | Runs | Absolute best | Stop value | Success rate | Mean wall time | Wall-time sdev |
|---|---:|---:|---:|---:|---:|---:|
| Cassini1 | 100 | 4.9307 | 4.95535 | 100% | 0.25 s | 0.12 s |
| Cassini2 | 100 | 8.383 | 8.42491 | 100% | 4.16 s | 2.26 s |
| Gtoc1 | 100 | -1581950 | -1574080 | 100% | 3.33 s | 2.87 s |
| Messenger | 100 | 8.6299 | 8.673 | 100% | 3.36 s | 1.83 s |
| Rosetta | 100 | 1.3433 | 1.35 | 100% | 4.58 s | 1.73 s |
| Sagas | 100 | 18.188 | 18.279 | 100% | 0.90 s | 0.95 s |

Mean and standard deviation include every experiment, including failures. The
standard deviation is the population value (`ddof=0`), matching NumPy's
default. All six problems happened to succeed in all 100 experiments.

The copy-paste-ready AsciiDoc version is in
[`benchmark_gtop_100.adoc`](benchmark_gtop_100.adoc). All 600 individual
measurements are in
[`benchmark_gtop_100_raw.tsv`](benchmark_gtop_100_raw.tsv).

## Configuration

- CPU: AMD Ryzen 9 9950X, 16 physical cores / 32 logical CPUs
- Retry workers: 32 native threads
- Independent experiments: 100 per problem, with stable distinct seeds
- Initial evaluations per retry: 1,500
- Advanced maximum evaluation factor: 50
- Advanced diversity checkpoint interval: 100 retries
- Optimizer sequence: 40% differential evolution, then 60% active CMA-ES
- Per-problem retry caps and result value limits: defined by the native
  benchmark cases in `examples/src/benchmark_gtop.rs`
- Compiler: rustc 1.93.0; Cargo release profile
- OS: Linux 6.8.0-136-generic x86_64
- Date: 2026-07-22

Rust workers are native threads. Each top-level experiment is run sequentially
and has exclusive use of all 32 configured retry workers.

## Reproduce

From the repository root:

```bash
cargo build --release -p fcmaes-examples --bin benchmark-gtop
target/release/benchmark-gtop \
  --runs 100 \
  --workers 32 \
  --seed 1 \
  --raw-output benchmarks/benchmark_gtop_100_raw.tsv \
  --table-output benchmarks/benchmark_gtop_100.adoc
```

The default invocation excludes Tandem and Messenger Full. Use
`--include-slow` to add both or `--problem tandem` / `--problem messenger-full`
to benchmark either one explicitly.
