# M1 solver reconnaissance

Date: 2026-07-30
Solver: `microlp = "=0.6.0"`
Command: `cargo run --release --locked --bin reconnaissance`

The measurement ran while another long-running optimization campaign shared
the machine. The values establish horizon order of magnitude and API
suitability; they are not isolated cross-solver benchmarks.

| Hours | Variables | Constraints | Wall time | Simplex pivots |
|---:|---:|---:|---:|---:|
| 24 | 168 | 48 | 0.000343 s | 110 |
| 288 | 2,016 | 576 | 0.022581 s | 1,519 |
| 2,016 | 14,112 | 4,032 | 0.322141 s | 10,080 |
| 8,760 | 61,320 | 17,520 | 2.602267 s | 41,434 |

## API findings

- Bounded continuous variables, equality/inequality rows, minimization,
  objective values, primal values, termination reasons, and statistics are
  available.
- `Stats::lp_iterations` reports cumulative simplex pivots.
- `Problem` and `SolveOutcome` satisfy `Send`; one rebuilt problem can be
  solved independently per candidate worker.
- `Solution` supports adding a row and fixing/unfixing a variable, but the
  public API cannot replace arbitrary capacity bounds or constraint
  right-hand sides. Candidate dispatch models are rebuilt rather than falsely
  advertised as warm-started.
- Pure LP solves have a wall-clock limit but no public simplex-iteration limit.
  The MIP node limit is irrelevant here. A wall-clock limit would make outer
  fitness host-dependent, so the tutorial uses unlimited solves and accepts
  only proven optima.

## Gate decision

`microlp` remains the only dispatch solver and preserves a pure-Rust hot path.
The main optimizer campaigns use 96-step smoke and 288-step publication
models. Seasonal H₂ sizing is a focused 1,460-period, six-hour chronological
arm; its selected design receives an independent 8,760-hour replay.
