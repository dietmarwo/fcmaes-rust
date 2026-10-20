# Surrogate implementation decisions

Recorded 2026-10-20 for the first experimental delivery and revised the same day after review (performance and the simulation-seed identity fix). See [PLAN.md](PLAN.md) for the revised full program.

## Placement and dependencies

The numerical implementation is `crates/fcmaes-surrogate`, version 0.1.0, `publish=false`. The adapter is `crates/fcmaes-job-worker/src/surrogate.rs`. The M0 delivery changed no other repository. Source base is fcmaes-rust `92aaf03f9f03257a284126a322160d045422bc9a`; the M0 delivery kept the shared API at `eb1c9de66e677097a7ea22d9a4b2f6357c18751d`. The review moved it to `6a42472d1904bdcc81f81a29023c8b51ca025ee6` for declared artifacts (see below).

The GP uses a small purpose-built dense kernel module (`linalg.rs`): packed row-major lower-triangular storage, a left-looking Cholesky whose inner products run over contiguous rows, eight-accumulator dot products reduced in a fixed order, row-wise forward/backward substitution, an O(n) row append for fantasies, and `L^-1` / `K^-1` rows for the gradient. Each factorization is serial, so its result does not depend on the pool. Parallelism is across fitting restarts, gradient rows (n >= 96, reduced in row order), acquisition candidates/local starts and independent evaluations. libm supplies the GP special functions. The delivered nalgebra Cholesky (column-major storage, per-element divisions in the kernel fill) dominated proposal time and was replaced; the crate no longer depends on nalgebra. Existing fcmaes-core RNG/baseline algorithms avoid introducing an unrelated optimization stack. serde_json's `float_roundtrip` feature preserves exact serialized inputs for replay. Cargo.lock records exact dependency versions.

This choice keeps the dependency set small for n <= 256. It does not conclude anything about faer or blocked BLAS-3 factorizations at n=500–1000; those need their own prototypes and latency evidence.

## Numerics and search

- Matérn-5/2 and RBF kernels have unit signal variance, ARD lengthscales, a configurable nugget (default 1e-10 in standardized output units; M0 used 1e-8), and jitter starting at 1e-10 with eight bounded attempts. The nugget bounds how precisely the GP interpolates near the optimum: on smooth deterministic problems 1e-10 instead of 1e-8 lowered the final regret on 9–10 of 10 seeds (Branin, Hartmann-3/6), and 1e-12 lowered it further; on objectives with a deterministic high-frequency ripple (a stand-in for solver noise) 1e-10 was neutral and 1e-12 slightly worse. 1e-10 is the default; raise `nugget` for rough or numerically noisy simulations. Posterior variance is latent. Fantasy append updates a factor in O(n²) and recomputes the diagnostic negative log posterior; its agreement with full refactorization is tested.
- Lengthscale priors are LogNormal with log mean sqrt(2)+0.5 log(d), log variance three, as densities on physical lengthscales. The MAP fit minimizes the negative log posterior over log lengthscales in `[log 0.025, log(1000 sqrt d)]` with a bounded L-BFGS (memory 8, projected Armijo backtracking, variables held at a bound by their gradient frozen per iteration) and the analytic gradient `dNLP/du_j = sum_{a>b} W_ab g_ab (z_aj - z_bj)^2 + 1 + (u_j - mu)/3`, `W = K^-1 - alpha alpha^T`, `z = x/l`, `g = -2 dk/dr^2` (`k` for RBF, `5/3 (1+s) e^-s` for Matérn-5/2). The gradient is checked against central differences in unit tests. Restart 0 starts from the carried (warm) hyperparameters or the prior mode, further restarts from seeded draws around the mode. `fit_iterations` (default 50) bounds iterations per restart; values are computed before gradients, so rejected line-search trials cost one factorization only. This is a compact projected quasi-Newton method, not full L-BFGS-B.
- Initial points are the centre plus maximin LHS (best of four seeded candidates); proposal candidates combine global LHS and sparse local perturbations. Four best starts get coordinate refinement. This is not Sobol/RAASP's complete published implementation.
- Trust region (added after comparing with BoTorch, egobox, scikit-optimize and Optuna, see [the comparison](../../benchmarks/surrogate-comparison/README.md)): two of every three proposals maximize the acquisition inside a TuRBO-1-style box around the incumbent, side `L * l_i / geomean(l)` in unit coordinates, with the published TuRBO-1 constants (L starts at 0.8, doubles after three successive improving batches up to 1.6, halves after `ceil(max(4, d) / q)` non-improving batches, resets to 0.8 below 2^-7). Deliberate deviations from TuRBO-1: LogEI with coordinate refinement instead of Thompson sampling, a global GP on all data, an in-place restart at the incumbent, and every third proposal searching the whole box. The global proposals matter: a pure trust region trapped one of ten Hartmann-3 runs in a local minimum (regret 0.74); with one global proposal in three the worst Hartmann-3 run stayed below 0.008 and later, with the smaller nugget, below 3e-7. Against the global search alone (10 seeds, q=1) the hybrid improved Hartmann-6 (10/10 seeds), Styblinski–Tang-10 (10/10, median regret 101 → 57) and Branin (8/10). Every second (more global) or every fifth (more local) proposal global was measured too and was not better overall. The region state lives in `StrategyState.trust`, so replay and determinism are unchanged; `trust_region: false` restores the global search.
- Measured during review and **not adopted**: (1) gradient L-BFGS refinement of the acquisition (analytic posterior-mean/variance gradients and LogEI derivatives) made eight-point batch proposals at n=248 2–4x faster but did worse on Styblinski–Tang-10 (mean final regret 98.9 vs 77.7 with coordinate refinement, 20 seeds; Hartmann-6 indistinguishable), and a coordinate-then-gradient hybrid did not help either; (2) lowering the Kriging-believer incumbent to the fantasy mean inside a batch made Styblinski–Tang-10 worse in two paired 10-seed comparisons (median regret 122.6 vs 96.1 and 126.1 vs 95.7). The coordinate refinement and the delivered incumbent rule are kept. Both are single-suite observations, not general conclusions. Also measured and not adopted in the second pass (10 seeds, q=1): a finer coordinate refinement (40 sweeps down to step 1e-7), refinement steps relative to the trust-region size, the RBF kernel, the bilog output transform, 1024 instead of 256 candidates, and refitting the hyperparameters every round instead of every third; none improved the suite overall.
- LogEI includes a numerically stable negative-tail expansion and exact zero-variance cases. LCB is a minimization alternative. Standardize/bilog/auto preserve raw observations; failed outcomes are excluded from the GP rather than represented as invented precise penalties.
- M0 caps dimension at 32, new+warm history at 256, q at eight and job inner threads at eight. These caps bound work; they are not evidence of good quality at every point inside the cap. The measured quality cases are d=2/3 with n=48 and d=6/10 with n=96/128.

## Reproducibility and cancellation

Synchronous decisions depend on seed, ordered history and explicit state, not completion timing. Independent stream seeds are indexed by semantic item. Parallel vectors collect in input order; reductions remain ordered. Completion logging is immediate and can arrive out of order, while decision commits use proposal order. The complete histories match across one, two and eight installed threads in every paired benchmark run (see the report); unit tests cover 1/2/3/8. Cross-CPU/compiler bit equality is untested.

The task owns one bounded pool. `concurrent * evaluation_threads <= inner_threads`, and inner_threads<=8. `SimContext.threads()` is a callback allowance, not a per-evaluation OS partition. Nested Rayon work shares aggregate capacity. Ordinary callbacks cannot be interrupted mid-call; simulation checkpoints observe cancellation/deadline. Cancelled calls are counted separately, and their elapsed wall time contributes to the callback sum. Final validation cancellation leaves the historical claim unrefuted.

## Evidence and warm starts

Each completed optimization callback is logged with shared checksum framing and fsync, then an atomic snapshot is updated. Log sink errors propagate and prevent not-yet-started work where observed. Surviving snapshot data does not imply complete common runner accounting after a kill. Full trace replay verifies proposals/state/IDs/counts; it cannot verify objective truth.

Compact warm inputs are SHA-256 checked and bind model identity, resolved parameters and model input hashes; simulation warm inputs also bind the run seed. Bound changes drop points rather than clip them. A warm job starts new strategy state. Final exact independent objective/bounds recheck upgrades evidence to rechecked; mismatch refutes the best_found claim and marks the snapshot untrusted. Neither recheck nor replay proves a global optimum.

The M0 delivery added no shared artifacts: file names in result details referred to local attempt files, to avoid inventing common result contract fields before the shared SDK, runner and MCP capability existed. That capability now exists in optimization-jobs ([artifacts](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#artifacts)): each task declares its trace (`records`) and snapshot (`json`) before writing them, agents read them while the job runs and page through the trace, and the sealed result records their SHA-256, which a warm-start input reuses. The declarations live in the shared declaration file, so the result contract and the worker protocol are unchanged.

## Measurements and reproduction

[validation/REPORT.md](validation/REPORT.md) compares the delivered M0 (`4dd693b`) with the reviewed code (both review passes), and [the library comparison](../../benchmarks/surrogate-comparison/README.md) puts it next to BoTorch, egobox, scikit-optimize and Optuna. The report covers proposal overhead over d in {2,5,10,20,32} and n in {16..248}, and equal-budget quality on Branin/Hartmann-3 (48 calls, 5 seeds), Hartmann-6/Styblinski–Tang-10 (96/128 calls, 20 seeds) and the tutorial's RK4 oscillator (32 calls, 20 seeds). Only compact summaries are committed; the raw histories are regenerated by the commands below. The measuring container exposed two CPUs; pools of 1, 2 and 8 threads were installed. Library timings exclude fsync/startup.

From the repository root:

```sh
cargo build --release --locked -p fcmaes-surrogate --examples
B=target/release/examples
OUT=/tmp/surrogate-evidence; mkdir -p $OUT
S20=7,11,19,23,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107
for t in 1 2 8; do $B/benchmark --threads $t --output $OUT/new-small-$t.json; done
for t in 1 2; do
  $B/benchmark --threads $t --problems hartmann6,styblinski10 --seeds $S20 \
    --output $OUT/new-medium-$t.json
  $B/benchmark --threads $t --problems oscillator --seeds $S20 \
    --output $OUT/new-oscillator-$t.json
done
$B/overhead 1 > $OUT/overhead-new-1.jsonl
$B/overhead 2 > $OUT/overhead-new-2.jsonl
python3 scripts/surrogate/analyze.py $OUT
```

The `old-*` inputs of the comparison come from the same examples built at `4dd693b` (copy `examples/benchmark.rs`, `examples/overhead.rs` and `src/problems.rs` into that checkout); `analyze.py` reports whatever pairs are present. The one-thread library run serializes four logical pending evaluations; it is a decision-determinism check, not a legal four-concurrent-evaluation worker reservation. All methods share eight initial evaluations and the per-problem call budget. CMA-ES population is four.

To regenerate independent numeric fixtures use `mpmath==1.4.1` and `scripts/surrogate/generate_fixtures.py`; Rust tests use checked-in JSON and do not need Python numerical packages.

## Deferred decisions

Async queues, TuRBO, pathwise TS, handoff and the measured overhead governor remain P1 product work. Simulation support in other scalar profiles is the first integration extension. Use the new plan's sub-minute simulation workloads to decide the local GP cap, feature count, refit/adoption frequency and handoff policy. Larger budgets and 16/32-thread deployments require new evidence, not a higher constant alone.
