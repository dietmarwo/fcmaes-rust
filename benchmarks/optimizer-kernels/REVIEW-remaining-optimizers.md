# Review of the remaining-optimizer campaign

27 November 2026. The delivery is
`fcmaes-rust-remaining-optimizers-performance-patch.zip` (3,645,053 bytes,
SHA-256 `f6bee4ad2b0d7b5b6f5a65f59ded085bef35e9e34e38f4da846013278da8cb54`).
Its patch `fcmaes-rust-performance.patch` (SHA-256
`b008f3e661cec2e629d5f9dafa0c626215b688b6fef6a5d752321c8ea7db558a`) applied
unchanged to `ccb9f8b`, the prepared and untagged 0.5.0; that is the first
commit of the pull request.
[RESULTS-remaining-optimizers.md](RESULTS-remaining-optimizers.md) is the
delivered log, left as delivered, and
[the plan](../../docs/plans/remaining-optimizers-performance-plan-2026-11-27.md)
is the one it worked from.

**The patch is sound: default paths replay exactly, and the new capped and
observed entry points hold their contracts on every budget tried.** The
review found three effects the delivered changelog did not state: a panic
it fixes, a change in where nested parallel work runs, and a last-bit
change in weighted retry. It made no change to the solver code.

## What was measured

Both trees were built here (Rust 1.97, two cores).

| Check | Delivered | Here |
|---|---|---|
| Archive inventory | 880 entries | all match |
| Workspace tests | 445 | 445; 450 after this review |
| Formatting, Clippy with warnings denied, `fcmaes-core` documentation | pass | pass |
| Deterministic replays | 86 endpoint comparisons | 1,893 seeded runs, below |
| Paired timings | 9,012 screening and 1,460 confirmation rows | not repeated |

### Replay against the tree before the change

`replay_remaining` (an example program of `fcmaes-core`, added by the
review) hashes every point and value the objective sees and the returned
result. Five objectives, among them one with ties and one with non-finite
values, three dimensions, three seeds and three budgets for the scalar
solvers.

| Solver | Runs | Same calls and result |
|---|---:|---:|
| jSO, three populations | 405 | 405 |
| CR-FM-NES, even populations | 270 | 270 |
| CR-FM-NES, population 31 | 135 | the base panics, see below |
| PGPE | 405 | 405 |
| Dual Annealing, with and without local search | 270 | 270 |
| ArchiveDE, two to five objectives, three policies | 108 | 108 |
| Independent retry of BiteOpt, one worker | 90 | 90 |
| Coordinated retry, one worker | 90 | 90 |
| Weighted retry, exponents 1, 2 and 3 | 48 | 48 |
| Diversifier, grid and CVT archives | 72 | 72 |

### The new entry points

Five tests were added
(`crates/fcmaes-core/tests/remaining_optimizer_review.rs`).

- **Every budget.** For populations of 2, 7, 8, 31 and 32 and every budget
  from 1 to four populations and three, with a finite objective and with
  one that returns NaN on part of the box, capped CR-FM-NES and PGPE make
  exactly the budget of calls, all inside the box, and return the best
  finite value among them at the point that produced it. The delivered
  test checks five budgets at one population.
- **Same search.** On a budget of whole populations the capped loops ask
  for the points of the ordinary loops, in order. The capped Diversifier
  makes the calls of the ordinary one up to its cap, for every budget from
  1 to five populations and three.
- **Observed vectors.** A vector passed to `moretry_observed` that is too
  short, too long or not finite is rejected like a reevaluated one would
  be; a missing one is reevaluated and counted.
- **Threads.** One retry worker runs on the calling thread, two on a
  private pool.

## Findings

1. **The "odd population" fix repairs a panic.** In the base, CR-FM-NES
   with an odd population fails in `ask` with an index out of bounds (all
   135 replayed runs). The changelog said only that odd populations are
   rounded up.
2. **A one-worker retry no longer confines nested parallel work.** The
   base ran a single worker on a private one-thread Rayon pool, so
   `parallel_batch` with `workers <= 0` inside the retry closure ran on
   that one thread. The patch runs the worker on the calling thread, and
   the same call now uses the caller's pool: both cores here. With two or
   more workers nothing changes. This is the documented meaning of
   `workers <= 0`, but a caller who set one retry worker to stay on one
   core gets more. The changelog now says so and a test pins the thread.
3. **The squared scalarization is not bit-identical.** `v * v` and `sqrt`
   replace `powf` for the default exponent 2. Of 20 million scalarized
   pairs 25,362 (one in 790) differ in the last bit. None of the 48
   replayed weighted retries left its trajectory, but one can. The
   changelog now says so.
4. **The delivered changelog entries sat under "Unreleased".** 0.5.0 is
   prepared but not tagged, so they moved into its section.

## Not verified

- Any timing. The largest reported gains are on runs of fractions of a
  millisecond (one-worker retry) and in the scalarization.
- The candidate and boundary benchmark crates (L-SHADE, L-SRTDE, AMaLGaM,
  bounded Nelder–Mead) beyond their tests in CI.
- Surrogate/2 beyond reading the change: the skipped scan filled a list
  that the fitted LogEI branch never reads.
- Retries with more than two workers.

## Open points

- The capped entry points exist in Rust only. The Python binding and the
  job worker still finish whole populations.
- Whether a one-worker retry should keep its nested work on one thread is
  a policy question. The patch chose speed for very short runs.
