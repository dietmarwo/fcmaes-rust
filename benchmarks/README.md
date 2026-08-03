# Native benchmark results

Human-facing benchmark reports in this directory use Markdown. Raw TSV files
preserve every experiment and are the authoritative inputs for statistics.

| Workload | Report | Raw samples |
|---|---|---|
| Coordinated retry, BiteOpt retry, and DE→CMA retry on GTOP | [`benchmark_gtop.md`](benchmark_gtop.md) | [`benchmark_gtop_100_raw.tsv`](benchmark_gtop_100_raw.tsv), [`benchmark_gtop_tandem_100_raw.tsv`](benchmark_gtop_tandem_100_raw.tsv), [`benchmark_biteopt_gtop_rust_100_raw.tsv`](benchmark_biteopt_gtop_rust_100_raw.tsv), [`benchmark_de_cma_gtop_rust_100_raw.tsv`](benchmark_de_cma_gtop_rust_100_raw.tsv) |
| Serial external CMA-ES versus parallel `fcmaes` retry at equal wall time on GTOP | [`gtop-cmaes-retry/README.md`](gtop-cmaes-retry/README.md) | [`gtop-cmaes-retry/results/equal-wall-100-v2/results.csv`](gtop-cmaes-retry/results/equal-wall-100-v2/results.csv) (100-pair publication campaign), [`gtop-cmaes-retry/results/scaling-pilot-v1/results.csv`](gtop-cmaes-retry/results/scaling-pilot-v1/results.csv) (secondary fixed-work pilot) |
| fcmaes versus independent Rust optimizer crates | [`optimizer-comparison/comparison.md`](optimizer-comparison/comparison.md) | [raw artifacts](https://github.com/dietmarwo/fcmaes-rust/tree/main/benchmarks/optimizer-comparison/raw) |
| Controlled active CMA-ES implementation diagnostic | [`cmaes-implementation/README.md`](cmaes-implementation/README.md) | [`cmaes-implementation/results/implementation-v1/paired.csv`](cmaes-implementation/results/implementation-v1/paired.csv) (20-pair publication campaign) |
| Core boundary: DE versus Nelder–Mead refinement and EGO at equal wall deadlines | [`optimizer-boundary/results/decision-v2/comparison.md`](optimizer-boundary/results/decision-v2/comparison.md) | [`refiner-raw.tsv`](optimizer-boundary/results/decision-v2/refiner-raw.tsv), [`bo-trace.tsv`](optimizer-boundary/results/decision-v2/bo-trace.tsv) |

Recreate the recorded native fcmaes workloads from the repository root:

```bash
cargo run --release -p fcmaes-examples --bin benchmark-gtop -- \
  --runs 100 --workers 32 --seed 1 \
  --raw-output benchmarks/benchmark_gtop_100_raw.tsv

python3 benchmarks/run_coordinated_tandem.py

cargo run --release -p fcmaes-examples --bin benchmark-biteopt-gtop -- \
  --algo biteopt --runs 100 --workers 24 --retries 24 \
  --evaluations 10000 --seed 1

cargo run --release -p fcmaes-examples --bin benchmark-biteopt-gtop -- \
  --algo de_cma --runs 100 --workers 24 --retries 24 \
  --evaluations 10000 --seed 1
```

The binaries print Markdown tables and accept `--table-output PATH` when a
separate generated `.md` file is useful.
The recorded slow Tandem run also includes
[`benchmark_gtop_tandem_100_metadata.json`](benchmark_gtop_tandem_100_metadata.json)
with its exact configuration and total invocation time.

Run the dependency-isolated optimizer comparison with:

```bash
benchmarks/optimizer-comparison/run_all_external.sh
```

Run the controlled same-family CMA-ES diagnostic from its standalone workspace:

```bash
cd benchmarks/cmaes-implementation
cargo run --release --locked -- --mode verify
cargo run --release --locked -- \
  --mode campaign --preset pilot --output results/pilot
```

The checked-in three-seed smoke bundle validates the harness, reporting, and
parallel topology. The separate 7,920-row publication bundle contains the
complete 20-pair diagnostic protocol and its scoped conclusions. The easy
analytic functions are implementation controls, not recommended CMA-ES
applications.

Run an external single-threaded CMA-ES implementation through the generic
retry adapter and compare one versus physical-core restart lanes under the
same four-second wall allowance with:

```bash
cd benchmarks/gtop-cmaes-retry
cargo run --release --locked -- \
  --mode campaign --preset publication \
  --output results/my-equal-wall-100
```

This experiment reports paired best-objective mean/sdev as its primary result.
Measured wall mean/sdev verifies comparable user wait, while starts,
evaluations, CPU time, and active cores document the extra parallel work.
In its completed 100-pair campaign, retry improves mean objective on every one
of the seven GTOP problems at the matched four-second wall allowance and wins
86–96 pairs per problem. With about 15 times as many independent starts, iid
restart order statistics predict a win rate near 94%; the observed wins mainly
confirm the scheduler. The mean and sdev columns carry the distributional
result. The linked benchmark README contains the complete quality, equal-wall
fairness, and compute-work tables. It also demonstrates that generic fcmaes
retry can coordinate a compatible single-threaded external optimizer; the
optimizer need not be implemented by fcmaes-core.

Reproduce the dependency-isolated optimizer-boundary experiment using the
commands in its [protocol](optimizer-boundary/README.md). It keeps the
experiment-only `egobox-ego` and Nelder–Mead implementations outside the root
workspace and `fcmaes-core` dependency graph.

Wall times depend strongly on CPU, operating system, compiler version, and
background load. Treat recorded timings as reproducibility data for the stated
machine, not as universal performance guarantees.
