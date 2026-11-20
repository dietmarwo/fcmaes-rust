# Coordinated retry: jSO→CMA against DE→CMA

Protocol: [PROTOCOL.md](PROTOCOL.md), declared before any run. This page
records how stage 1 was produced. Stage 2 has not been run.

## Revision

All 320 stage-1 rows come from one binary, `gtop-advexamples` built in
release mode from the commit that adds `--jso-cma` (the first commit after
the protocol commit). The switch only selects the sequence of coordinated
retry; it is off by default.

## Bit identity of the control arm

Before any benchmark run, the binary without `--jso-cma` was run for
`cassini2` and `cassini1`, seed 1, with the stage settings. Its complete
output was byte-identical to the output of the revision before the change:

| problem, seed 1 | value | evaluations | identical |
|---|---:|---:|---|
| cassini2 | 8.942502857947 | 10586304 | yes |
| cassini1 | 4.930708172623 | 9459102 | yes |

## Reproduce

```sh
cargo build --release --locked -p fcmaes-examples --bin gtop-advexamples
export CARGO_TARGET_DIR=...   # or pass --binary PATH
python3 benchmarks/coordinated-jso-cma/run_stage.py --stage 1   # resumable, 2 runs in parallel
python3 benchmarks/coordinated-jso-cma/collect.py               # results/stage1.tsv
python3 benchmarks/coordinated-jso-cma/decide.py                # results/report.md
python3 benchmarks/coordinated-jso-cma/decide.py --self-test
```

`run_stage.py` writes one file per run to `results/runs/` (temporary name,
renamed at the end), skips finished runs, holds a lock file, and stops
starting runs after `--budget-s` seconds so that it can be called in chunks.
Each run file holds the program output, the command and the wall time.
The command line in the run files shows the binary by name only.

## Files

- `results/runs/`: one raw file per run (320).
- `results/stage1.tsv`: problem, arm, seed, value, evaluations, retries,
  wall_s, retained. A run without a retained result would show `inf` in
  `value` and `no` in `retained` and counts as worse than every result;
  there is none in stage 1. Here `retries` is the `runs=` field of the program
  output.
- `results/report.md`: stage-1 table and the protocol's selection.
- `decide.py` uses `scipy.stats.mannwhitneyu` (two-sided, default method).
- Wall times depend on the load of the host; the host had other jobs during
  most of stage 1, and two runs were in parallel on two cores.
