# Coordinated retry: jSO→CMA against DE→CMA

Protocol: [PROTOCOL.md](PROTOCOL.md), declared before any run. This page
records the outcome and how stage 1 was produced.

## Outcome

**No port** by the protocol's rule: in stage 1 `jso_cma` is significantly
better on no problem, so no problem qualifies for stage 2 and the test ends
there. Stage 2 was not run.

| Problem | Median `de_cma` | Median `jso_cma` | p | Holm p |
|---|---:|---:|---:|---:|
| cassini1 | 4.93073 | 4.93071 | 0.114 | 0.430 |
| cassini2 | 8.63725 | 8.76629 | 0.041 | 0.247 |
| rosetta | 1.35244 | 1.35195 | 0.839 | 0.839 |
| tandem | -1292.35 | -1339.90 | 0.365 | 0.730 |
| messenger | 8.70163 | 8.70163 | 0.057 | 0.283 |
| gtoc1 | -1447330 | -1490810 | 0.108 | 0.430 |
| sagas | 18.5009 | 82.138 | 5.2e-07 | 4.2e-06 |
| messenger-full | 6.09938 | 7.01867 | 8.3e-05 | 5.8e-04 |

- `jso_cma` is significantly worse on `sagas` and `messenger-full`.
- The median evaluations of the two arms differ by at most 4% on every
  problem, inside the protocol's 10% margin.
- All 320 runs finished with a retained result.
- The medians and tests were recounted from the 320 raw run files with a
  second script; they agree with [results/report.md](results/report.md).

The protocol's question was whether jSO is worth a port to the C++ backend
of fast-cma-es. That port was made on the maintainer's decision while stage 1
was running. The result stands as guidance for its use: `jso_cma` is not a
replacement for `de_cma` in coordinated retry.

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
