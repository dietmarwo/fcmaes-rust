# Native benchmark results

Human-facing benchmark reports in this directory use Markdown. Raw TSV files
preserve every experiment and are the authoritative inputs for statistics.

| Workload | Report | Raw samples |
|---|---|---|
| Coordinated retry, BiteOpt retry, and DE→CMA retry on GTOP | [`benchmark_gtop.md`](benchmark_gtop.md) | [`benchmark_gtop_100_raw.tsv`](benchmark_gtop_100_raw.tsv), [`benchmark_gtop_tandem_100_raw.tsv`](benchmark_gtop_tandem_100_raw.tsv), [`benchmark_biteopt_gtop_rust_100_raw.tsv`](benchmark_biteopt_gtop_rust_100_raw.tsv), [`benchmark_de_cma_gtop_rust_100_raw.tsv`](benchmark_de_cma_gtop_rust_100_raw.tsv) |
| fcmaes versus independent Rust optimizer crates | [`optimizer-comparison/comparison.md`](optimizer-comparison/comparison.md) | [`optimizer-comparison/raw/`](optimizer-comparison/raw/) |

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

Wall times depend strongly on CPU, operating system, compiler version, and
background load. Treat recorded timings as reproducibility data for the stated
machine, not as universal performance guarantees.
