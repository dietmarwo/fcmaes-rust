# GTOP optimizer comparison

This directory compares fcmaes BiteOpt with independent Rust optimizer
libraries on the seven problems and stop targets from
the [native GTOP benchmark report](../benchmark_gtop.md).
The original definitions are in ESA's
[GTOP database](https://www.esa.int/gsp/ACT/projects/gtop/), with
[TandEM documented here](https://www.esa.int/gsp/ACT/projects/gtop/tandem/).

The alternative crates are never added to the public fcmaes Cargo workspace.
[`prepare_external.sh`](prepare_external.sh) copies the published adapter
sources into a sibling workspace, copies the exact native GTOP objective
implementation from `examples/src/gtop.rs`, and builds the comparison there.
Consequently, `cargo build --workspace` and `cargo test --workspace` from the
repository root do not resolve or compile any comparison library.

Run the complete experiment from the public repository root:

```bash
benchmarks/optimizer-comparison/run_all_external.sh
```

The run uses 100 experiments, 24 workers, 24 BiteOpt retries, 10,000
evaluations per retry, and a common maximum of 240,000 evaluations per
experiment. It can be interrupted and restarted; complete raw rows are
retained and skipped.

The equal-budget fcmaes coordinated row uses 48 DE→CMA retries whose limits
grow from 2,500 to 7,500 evaluations and sum to the same 240,000 total.
BIPOP-CMA-ES is treated as an adaptive restart strategy and compared
conceptually with this row, not only with straight independent BiteOpt retry.

After the complete matrix exists, run the pre-registered Tandem stress test:

```bash
python3 benchmarks/optimizer-comparison/run_tandem_stress_external.py
```

It selects the alternative with the lowest mean Tandem optimum in the 100-run
table, then runs 1,000 independent retries with 10,000,000 evaluations each.
The retries use 24 concurrent single-threaded optimizer processes, so a
library without internal population parallelism can still use all 24 cores
without nested oversubscription. Shards are kept in the external workspace and
make the run resumable.

Published artifacts:

- `comparison.md`: generated result tables and interpretation constraints.
- `raw/*.tsv`: every individual measurement plus a combined file.
- `sources/`: the exact common harness and optimizer adapters.
- `Cargo.lock.reference`: exact external dependency resolution.
- `environment.txt`: recorded host, toolchain, and parallel configuration.
- `SHA256SUMS`: checksums for sources, raw results, and report artifacts.
- `logs/`: progress output from the recorded run.
- `tandem_stress_metadata.json`: selection rule, selected optimizer, stress
  configuration, and total wall time.

`genetic_algorithms` L-SHADE is a serial baseline because version 3.0.0 does
not expose parallel evaluation in its DE engine. `math-optimisation` supplies
an explicitly 24-thread DE baseline, but is GPL-3.0-or-later and remains
strictly isolated in the external benchmark workspace.
