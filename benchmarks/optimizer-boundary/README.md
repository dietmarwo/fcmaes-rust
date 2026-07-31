# Optimizer-boundary experiment

This dependency-isolated benchmark asks two release-scope questions:

1. Does a bounded Nelder–Mead implementation deserve to become a general
   `fcmaes-core` optimizer or refiner?
2. Does Bayesian optimization belong in core, or is explicit interoperation
   with a specialized crate the better boundary?

The experiment is intentionally outside the root Cargo workspace. Its local
`egobox-ego` dependency does not enter `fcmaes-core`, the Python extension, or
normal application builds.

## Corrections relative to the exploratory experiment

The original exploratory campaign established useful hypotheses but did not
support a release decision. This version corrects its protocol:

- every population generation and simplex call is charged to an exact resource
  budget;
- a simplex always receives at least `2 × (dimension + 1)` calls, so it can be
  initialized and perform descent;
- DE and DE→NM share an identical seed and bit-identical DE prefix;
- the DE prefix is reported as its own arm, distinguishing tail improvement
  from a different global-search trajectory;
- 16-worker DE populations and NM multistarts really execute concurrently;
- all root-seed outcomes are retained rather than only aggregate medians;
- fixed common-random-number ReBop and genuinely resampled ReBop are separate
  problem variants;
- both ReBop variants are selected on four training paths and scored on eight
  disjoint, fixed validation paths; and
- Bayesian and DE best-so-far traces are compared at equal reconstructed wall
  deadlines, including measured optimizer overhead.

## Refiner protocols

DE uses a population of 16. A DE generation costs 16 rounds with one worker and
one round with 16 workers. Serial NM consumes one call per round. NM multistart
runs one sequential simplex per worker.

| Protocol | Workers | Available rounds | Full DE calls |
|---|---:|---:|---:|
| `serial-r160-w1` | 1 | 160 | 160 |
| `parallel-r60-w16` | 16 | 60 | 960 |

For DE→NM, 20% of the rounds are requested for the tail, raised to at least two
complete simplex constructions. The remaining head is rounded down to complete
DE generations and the tail receives the remainder. Thus the hybrid and the
full DE control consume the same idealized wall rounds even though the parallel
DE control can make more objective calls.

The four landscapes are:

- `optical-lens`: deterministic ray tracing with hard invalid-design regions;
- `cfd-ventilation`: deterministic lattice-Boltzmann room simulation;
- `rebop-crn`: stochastic reaction paths evaluated on a fixed common seed set;
- `rebop-resampled`: new reaction-path seeds on every objective call.

The experiment-local bounded NM implementation is not production API. Keeping
it here is deliberate: the benchmark is deciding whether promotion is
justified.

## Equal-wall-time BO protocol

Sequential EGO from `egobox-ego` and DE are run to 160 objective calls on the
deterministic CFD and optical-lens landscapes. Every call records best-so-far
quality and cumulative optimizer overhead with simulator time subtracted.

For assumed simulator latency `c`, the reconstructed completion time of call
`i` is:

```text
optimizer_overhead(i) + i × c
```

Both arms are read at deadline `N × c`, for nominal call budgets
`N ∈ {25, 60, 150}` and latencies from 1 ms through 1 s. This prevents BO from
receiving both an equal objective-call budget and uncharged surrogate fitting.
Latency reconstruction is appropriate for a single external or blocking
simulator worker; it is not presented as a batch-qEI result.

## Reproduction

From this directory:

```bash
cargo test --release --locked

cargo run --release --locked -- \
  --mode refiner --seeds 20 --output results/decision-v2

cargo run --release --locked -- \
  --mode bo --seeds 20 --output results/decision-v2

python3 analyze.py \
  --results results/decision-v2 \
  --output results/decision-v2/comparison.md
```

Run the timing-sensitive BO command without an unrelated CPU-intensive
campaign. The refiner program checkpoints complete problem/protocol blocks and
the BO program checkpoints every completed arm.

The checked-in raw TSV files are authoritative. `comparison.md` is a
deterministic rendering of those files.

## Recorded decision

The 20-seed `decision-v2` campaign is complete. Its
[comparison](results/decision-v2/comparison.md),
[raw refiner outcomes](results/decision-v2/refiner-raw.tsv),
[BO traces](results/decision-v2/bo-trace.tsv), and
[environment record](results/decision-v2/environment.txt) are checked in.

The evidence does not justify promoting either experiment-local method into
`fcmaes-core`:

- standalone NM lost every median problem/protocol comparison to full DE;
- DE→NM had no robust one-worker advantage and was materially worse than full
  DE on the 16-worker optical and CFD blocks;
- BO helped at some very small, sequential deadlines, notably CFD from an
  assumed 10 ms evaluation cost, but the advantage did not persist at the
  larger budget and was landscape-dependent; and
- end-of-trace EGO overhead was about 2.96 seconds on CFD and 4.02 seconds on
  optical, while DE overhead was about 0.1 milliseconds.

The resulting public policy is documented in
[The optimizer boundary](../../docs/optimizer-boundary.md): external methods
are supported through retry's optimizer closure, without entering the core
dependency graph.
