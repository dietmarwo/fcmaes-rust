# Surrogate implementation decisions

Recorded 2026-10-20 for the first experimental delivery. See [PLAN.md](PLAN.md) for the revised full program.

## Placement and dependencies

The numerical implementation is `crates/fcmaes-surrogate`, version 0.1.0, `publish=false`. The adapter is `crates/fcmaes-job-worker/src/surrogate.rs`. No changes to LP-IP-craft, SQP-craft or optimization-jobs are included. Source base is fcmaes-rust `92aaf03f9f03257a284126a322160d045422bc9a`; the existing shared API remains pinned to `eb1c9de66e677097a7ea22d9a4b2f6357c18751d`.

Use nalgebra 0.33, already present in the workspace, with serial Cholesky/ordered solves inside each model. Parallelism is across fitting restarts, acquisition candidates/local starts and independent evaluations. libm supplies the GP special functions. Existing fcmaes-core RNG/baseline algorithms avoid introducing an unrelated optimization stack. serde_json's `float_roundtrip` feature preserves exact serialized inputs for replay. Cargo.lock records exact dependency versions.

This choice prioritizes a small verifiable baseline. It does not conclude that nalgebra is faster than faer at n=500 or that no Rust L-BFGS-B/Sobol implementation exists. Such decisions require future prototypes, dependency review and latency evidence.

## Numerics and search

- Matérn-5/2 and RBF kernels have unit signal variance, ARD lengthscales, nugget 1e-8, and jitter starting at 1e-10 with eight bounded attempts. Posterior variance is latent. Fantasy append updates a factor in O(n²) and recomputes the diagnostic negative log posterior; its agreement with full refactorization is tested.
- Lengthscale priors are LogNormal with log mean sqrt(2)+0.5 log(d), log variance three, as densities on physical lengthscales. MAP uses bounded coordinate-pattern search in log lengthscales, with 0.025 lower bound and deterministic restarts. There are no analytic-gradient or L-BFGS-B claims.
- Initial points are the centre plus maximin LHS (best of four seeded candidates); proposal candidates combine global LHS and sparse local perturbations. Four best starts get coordinate refinement. This is not Sobol/RAASP's complete published implementation.
- LogEI includes a numerically stable negative-tail expansion and exact zero-variance cases. LCB is a minimization alternative. Standardize/bilog/auto preserve raw observations; failed outcomes are excluded from the GP rather than represented as invented precise penalties.
- M0 caps dimension at 32, new+warm history at 256, q at eight and job inner threads at eight. These caps bound work; they are not evidence of good quality at every point inside the cap. The measured quality cases are d=2/3, n=48.

## Reproducibility and cancellation

Synchronous decisions depend on seed, ordered history and explicit state, not completion timing. Independent stream seeds are indexed by semantic item. Parallel vectors collect in input order; reductions remain ordered. Completion logging is immediate and can arrive out of order, while decision commits use proposal order. The complete histories match across one and eight installed threads in all 40 paired benchmark runs; unit tests cover 1/2/3/8. Cross-CPU/compiler bit equality is untested.

The task owns one bounded pool. `concurrent * evaluation_threads <= inner_threads`, and inner_threads<=8. `SimContext.threads()` is a callback allowance, not a per-evaluation OS partition. Nested Rayon work shares aggregate capacity. Ordinary callbacks cannot be interrupted mid-call; simulation checkpoints observe cancellation/deadline. Cancelled calls are counted separately, and their elapsed wall time contributes to the callback sum. Final validation cancellation leaves the historical claim unrefuted.

## Evidence and warm starts

Each completed optimization callback is logged with shared checksum framing and fsync, then an atomic snapshot is updated. Log sink errors propagate and prevent not-yet-started work where observed. Surviving snapshot data does not imply complete common runner accounting after a kill. Full trace replay verifies proposals/state/IDs/counts; it cannot verify objective truth.

Compact warm inputs are SHA-256 checked and bind model identity, resolved parameters and model input hashes; simulation warm inputs also bind the run seed. Bound changes drop points rather than clip them. A warm job starts new strategy state. Final exact independent objective/bounds recheck upgrades evidence to rechecked; mismatch refutes the best_found claim and marks the snapshot untrusted. Neither recheck nor replay proves a global optimum.

Shared remote artifacts/paging are not added. File names in result details refer to local attempt files. This avoids inventing common result contract fields before the shared SDK/runner/MCP capability is designed and tested.

## Measurements and reproduction

[validation/REPORT.md](validation/REPORT.md) summarizes raw JSON and environment metadata in that directory. The host exposes nine logical CPUs and an eight-CPU cgroup quota; pools explicitly use one or eight threads. It is not a dedicated eight-core server. Library timings exclude fsync/startup, while the tutorial exercises durable worker paths. The 10 ms delay campaign is synthetic, not a real 1–60 s simulation study.

From the repository root:

```sh
cargo run --release --locked -p fcmaes-surrogate --example benchmark -- \
  --threads 8 --concurrent 4 --evaluations 48 \
  --output docs/surrogate/validation/smoke-8-threads.json
cargo run --release --locked -p fcmaes-surrogate --example benchmark -- \
  --threads 1 --concurrent 4 --evaluations 48 \
  --output docs/surrogate/validation/smoke-1-thread.json
cargo run --release --locked -p fcmaes-surrogate --example benchmark -- \
  --threads 8 --concurrent 4 --evaluations 48 --sleep-ms 10 --seeds 7,11 \
  --output docs/surrogate/validation/sleepy-8-threads.json
python3 scripts/surrogate/analyze.py
```

The one-thread library run serializes four logical pending evaluations; it is a decision-determinism check, not a legal four-concurrent-evaluation worker reservation. All methods share eight initial evaluations and the 48-total-call budget. CMA-ES population is four. Quality improves strongly on these low-dimensional functions; proposal overhead is higher than the baselines, so no general wall-time win is claimed.

For the optional figure install matplotlib and run `python3 scripts/surrogate/analyze.py --plot`. To regenerate independent numeric fixtures use `mpmath==1.4.1` and `scripts/surrogate/generate_fixtures.py`; Rust tests use checked-in JSON and do not need Python numerical packages.

## Deferred decisions

Async queues, TuRBO, pathwise TS, handoff and the measured overhead governor remain P1 product work. Simulation support in other scalar profiles is the first integration extension. Use the new plan's sub-minute simulation workloads to decide the local GP cap, feature count, refit/adoption frequency and handoff policy. Larger budgets and 16/32-thread deployments require new evidence, not a higher constant alone.
