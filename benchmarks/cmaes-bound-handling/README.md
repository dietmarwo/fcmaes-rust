# CMA-ES bound handling: clip, repair, transform

A declared comparison of three ways active CMA-ES in `fcmaes-core` handles
samples that leave the box: clip (`repair_z: false`, the default), repair
(`repair_z: true`) and the experimental transform
(`CmaesParams::bound_transform`, the `BoundTransform` of pycma).

- [PROTOCOL.md](PROTOCOL.md): question, comparisons and decision rule,
  declared before the implementation and the runs.
- [DEVIATIONS.md](DEVIATIONS.md): details the protocol leaves open, fixed
  before the affected runs.
- [results/report.md](results/report.md): tables and decision, written by
  [decide.py](decide.py) from the raw files in [results/](results/).

## Reproduce

All runs are single-threaded. Every step skips the rows its output file
already holds, so an interrupted step resumes when it is run again.

```sh
# Binaries (from the repository root)
cargo build --release --locked -p fcmaes-examples --bin gtop-advexamples
(cd benchmarks/candidate-optimizers && cargo build --release --locked --features candidates -p candidate-harness)
(cd benchmarks/cmaes-bound-handling/runner && cargo build --release --locked)

cd benchmarks/cmaes-bound-handling
T=../../target/release   # or $CARGO_TARGET_DIR/release

# A: neural controller, 110,000 evaluations, seeds 1 to 10
$T/controller --seeds 1-10 --budget 110000 --out results/controller.tsv
# (--verify checks clip and repair runs against the job worker's es_retry)

# B: truss_sizing, jSO->CMA, protocol T
(cd ../candidate-optimizers && $T/candidate-harness run --protocols T \
    --problems truss_sizing --solvers jso_cma,jso_cma_rz,jso_cma_tr \
    --root-seed 7 --total 1000000 --run-cap 1000000 --reps 10 --threads 1 \
    --out ../cmaes-bound-handling/results/truss.tsv)

# C and D: coordinated DE->CMA retry, seeds 1 to 12
python3 run_retry.py --binary $T/gtop-advexamples --problem messenger
python3 run_retry.py --binary $T/gtop-advexamples --problem cassini2

# Informational: A at 900,000 evaluations, and the boundary-start probes
$T/controller --seeds 1-5 --budget 900000 --out results/controller-900k.tsv
$T/probes --out results/probes.tsv

# Tables and decision
python3 decide.py
```

`gtop-advexamples` takes `--bound-transform` (and `--repair-z`) for every
CMA-ES it builds; the candidate harness has the arms `jso_cma_rz` and
`jso_cma_tr`, identical to `jso_cma` except for the bound handling of the
CMA-ES stage.
