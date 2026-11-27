# Optimizer kernel performance: BiteOpt, DE, MODE, MAP-Elites and active CMA-ES

Two implementation campaigns of 27 November 2026 against commit `d7ff66a`.
One screened 80 changes to BiteOpt, DE, MODE and MAP-Elites and kept 25. The
other screened 30 changes to active CMA-ES and kept 6. Both landed in #83.

| Document | Content |
|---|---|
| [Review](REVIEW.md) | What was rerun and found when the two patches were merged |
| [Four-optimizer results](RESULTS-four-optimizers.md) | The delivered log: all 80 decisions and the measurements |
| [CMA-ES results](RESULTS-cmaes.md) | The delivered log: all 30 decisions and the measurements |
| [Four-optimizer plan](../../docs/plans/four-optimizer-performance-plan-2026-11-27.md) | Baseline measurements and the 80 proposals |
| [CMA-ES plan](../../docs/plans/cmaes-performance-plan-2026-11-27.md) | Baseline measurements and the 30 proposals |

A third campaign of the same day, against the prepared 0.5.0 (`ccb9f8b`),
covered the remaining solvers in 19 tracks and 380 trials:

| Document | Content |
|---|---|
| [Review](REVIEW-remaining-optimizers.md) | What was rerun and found when the patch was merged |
| [Results](RESULTS-remaining-optimizers.md) | The delivered log: all 380 decisions and the measurements |
| [Plan](../../docs/plans/remaining-optimizers-performance-plan-2026-11-27.md) | Baseline measurements and the proposals |

## Raw evidence

The raw rows are not committed. They ship in the two delivery archives:

| Archive | Bytes | SHA-256 | Files |
|---|---:|---|---:|
| `fcmaes-rust-four-optimizers-patch.zip` | 360,035 | `cc9d00f462c068d61632ee47af761b0537bb1e7ae0422cd27a366d64bdd4ac6d` | 159 |
| `fcmaes-rust-cmaes-performance-patch.zip` | 479,511 | `6c29807306b6cb80be308a68cfb5e74a57189cb625d3cdf53cfcefd1a16e2b13` | 211 |
| `fcmaes-rust-remaining-optimizers-performance-patch.zip` | 3,645,053 | `f6bee4ad2b0d7b5b6f5a65f59ded085bef35e9e34e38f4da846013278da8cb54` | 881 |

[`evidence-four-optimizers.sha256`](https://github.com/dietmarwo/fcmaes-rust/blob/main/benchmarks/optimizer-kernels/evidence-four-optimizers.sha256)
and
[`evidence-cmaes.sha256`](https://github.com/dietmarwo/fcmaes-rust/blob/main/benchmarks/optimizer-kernels/evidence-cmaes.sha256)
list every file of each archive; check a copy with `sha256sum -c` from the
extracted directory. The lists were made at import, because the archives
carry no list of their own.
[`evidence-remaining-optimizers.sha256`](https://github.com/dietmarwo/fcmaes-rust/blob/main/benchmarks/optimizer-kernels/evidence-remaining-optimizers.sha256)
is the third archive's own `SHA256SUMS` (880 entries).

The four-optimizer archive holds the patch, 2,698 paired timing rows in 15
CSV files, the decision table, the benchmark drivers, 94 variant files and
the application outputs. The CMA-ES archive holds the patch, six result
files with 2,302 rows, 149 files of per-trial diffs and data, the replay
drivers and the build and test logs. Its `manifest.json` records the tree
of the patched checkout, `caa0a9e`, which the review reproduced.

## Reproduce the review

Two example programs of `fcmaes-core` compare two checkouts. Build each in a
checkout of the release to compare and in a current one, then compare the
output files.

```bash
cargo build --release --locked -p fcmaes-core --example replay_digest --example cma_target

# One line per seeded run: a digest of every evaluated point and of the result.
for kind in bite de mode map cma; do
  RAYON_NUM_THREADS=1 target/release/examples/replay_digest $kind > digest-$kind.txt
done

# Evaluations and seconds to 1e-8: problem, dimension, last seed, budget.
target/release/examples/cma_target ellipsoid 100 8 1500000
```

`replay_digest` runs 2,770 cases: direct, deep and batched BiteOpt; serial,
ask/tell and delayed DE with integer masks and normal starts; MODE with one
to five objectives, constraints, odd populations, tied and non-finite
values; MAP-Elites on grids and CVT archives of two to eight descriptor
dimensions; and CMA-ES below and above dimension 64 with `optimize`,
`ask`/`tell` and `tell_x`. Equal lines mean equal runs. The digests depend
on the platform's `libm`, so compare files made on one machine.

`replay_remaining` does the same for the third campaign: 1,893 runs of
jSO, CR-FM-NES, PGPE, Dual Annealing, ArchiveDE, the three retry
coordinators with one worker and the Diversifier.

```bash
cargo build --release --locked -p fcmaes-core --example replay_remaining
target/release/examples/replay_remaining > remaining.txt   # or: scalar archivede retry moretry diversifier
```

[`review/replay-remaining-summary.txt`](https://github.com/dietmarwo/fcmaes-rust/blob/main/benchmarks/optimizer-kernels/review/replay-remaining-summary.txt)
has its counts.

[`review/time-to-target.csv`](https://github.com/dietmarwo/fcmaes-rust/blob/main/benchmarks/optimizer-kernels/review/time-to-target.csv)
holds the review's `cma_target` rows for both builds, and
[`review/replay-summary.txt`](https://github.com/dietmarwo/fcmaes-rust/blob/main/benchmarks/optimizer-kernels/review/replay-summary.txt)
the counts of equal and differing digest lines.
