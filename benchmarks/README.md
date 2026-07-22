# Native benchmark results

This directory contains recorded output from the Rust benchmark binaries.
The raw TSV files preserve every experiment; the AsciiDoc files contain the
generated summary tables.

| Workload | Summary | Raw samples |
|---|---|---|
| Coordinated-retry GTOP | [`benchmark_gtop_100.adoc`](benchmark_gtop_100.adoc) | [`benchmark_gtop_100_raw.tsv`](benchmark_gtop_100_raw.tsv) |
| BiteOpt basic-retry GTOP | [`benchmark_biteopt_gtop_rust_100.adoc`](benchmark_biteopt_gtop_rust_100.adoc) | [`benchmark_biteopt_gtop_rust_100_raw.tsv`](benchmark_biteopt_gtop_rust_100_raw.tsv) |
| DE→CMA basic-retry GTOP | [`benchmark_de_cma_gtop_rust_100.adoc`](benchmark_de_cma_gtop_rust_100.adoc) | [`benchmark_de_cma_gtop_rust_100_raw.tsv`](benchmark_de_cma_gtop_rust_100_raw.tsv) |

The coordinated-retry methodology and environment are documented in
[`benchmark_gtop.md`](benchmark_gtop.md).

Recreate its default workload from the repository root:

```bash
cargo run --release -p fcmaes-examples --bin benchmark-gtop -- \
  --runs 100 --workers 32 --seed 1 \
  --raw-output benchmarks/benchmark_gtop_100_raw.tsv \
  --table-output benchmarks/benchmark_gtop_100.adoc
```

Run the short basic-retry benchmark with either optimizer sequence:

```bash
cargo run --release -p fcmaes-examples --bin benchmark-biteopt-gtop -- \
  --algo biteopt --runs 100 --workers 24 --retries 24 \
  --evaluations 10000 --seed 1

cargo run --release -p fcmaes-examples --bin benchmark-biteopt-gtop -- \
  --algo de_cma --runs 100 --workers 24 --retries 24 \
  --evaluations 10000 --seed 1
```

Wall times depend strongly on CPU, operating system, compiler version, and
background load. Treat recorded timings as reproducibility data for the stated
machine, not as universal performance guarantees.
