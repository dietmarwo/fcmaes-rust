# Experimental native surrogate v2

The original `fcmaes.surrogate/1` and `fcmaes_surrogate::run` remain the recommended small deterministic-box implementation. The new opt-in `fcmaes.surrogate/2` and `fcmaes_surrogate::advanced` support a larger archive, bounded GP fitting, asynchronous callbacks, real region restarts, correlated approximate Thompson sampling, an overhead controller and a budgeted BiteOpt handoff.

The new policies did **not** pass the quality gate for default promotion. Read [the raw-data-derived results](advanced-evidence/README.md). The review of the delivery found that the LogEI policy never narrowed its search and that synchronous decisions depended on callback completion order; both are fixed. On 30 fresh seeds after the fixes, `advanced` against `legacy` has a paired geometric loss ratio of 3.5 on sphere (95% interval 2.1–6.0, down from 231 before the fix), 1.06 on Styblinski–Tang (0.87–1.31) and 1.01 on the oscillator (1.00–1.03). The existing method still wins precision on sphere, and the upper bounds on sphere and Styblinski–Tang exceed the gate's 1.05. Ackley in the test panel starts at its optimum and is not useful search evidence.

## Rust API

```rust
use fcmaes_surrogate::{Space, advanced::{self, Options, RunSpec, Schedule}};
let space = Space::new(vec![-2.0; 2], vec![2.0; 2], vec![false; 2])?;
let options = Options { concurrent: 2, schedule: Schedule::Asynchronous,
    training_cap: 64, archive_cap: 1000, ..Options::default() };
let result = advanced::run(RunSpec { space: &space, options: &options,
    evaluations: 64, seed: 41, warm: &[], target: None, inner_threads: 3 },
    &|x| Ok(x.iter().map(|v|v*v).sum()), &mut |_event| Ok(()), &|| None)?;
# Ok::<(), fcmaes_surrogate::Error>(())
```

Supply a durable fallible sink for expensive work. The example closure discards events only for a small library demonstration. The worker supplies checked framing and atomic warm snapshots.

## Options and limits

| Option | Default | Meaning |
|---|---|---|
| base | legacy Config defaults | Kernel, transform, initial points, fitting/candidate budgets and deterministic regularization |
| training_cap | 128 | 8–240 selected observations; initial_points must fit |
| archive_cap | 10000 | Total new plus warm observations, bounded at 10000 |
| schedule | synchronous | `synchronous` or `asynchronous` |
| acquisition | log_ei | `log_ei` or approximate correlated `thompson` |
| design | latin_hypercube | Candidate design; optional `sobol_digital_shift` |
| features | 512 | 32–4096 Fourier features for Thompson paths |
| restart_failures | 32 | Region-attributed completed evaluations without meaningful improvement before a fresh design |
| surrogate_threads | 1 | Bounded numerical pool |
| concurrent | 1 | At most 8 in-flight callbacks |
| evaluation_threads | 1 | Allowance per callback in the bounded aggregate evaluation pool |
| governor | false | Timing-dependent candidate/refit control, recorded for replay |
| handoff_after | null | Optional BO callback count before serial BiteOpt |

In worker payloads, set `concurrent` and `evaluation_threads` at the top level; the resolved payload records them in `parameters` too, and a differing copy there is refused. The legacy base acquisition must be LogEI; the advanced acquisition is selected separately.

The base `trust_region` flag enables the trust region of the original method. Its side `region_length` (unit coordinates, recorded in each proposal's diagnostics) starts at 0.8, halves after max(4, d) completed evaluations without a meaningful improvement, doubles after three improvements up to 1.6, and resets to 0.8 below 1/128. Two of every three LogEI proposals refine inside a lengthscale-shaped box of that side around the incumbent; the third searches the whole box. Sparse local candidates around the incumbent use the same scale. Thompson proposals score the whole candidate pool. A restart after `restart_failures` non-improving evaluations also resets the region. The other native limits still hold: 1–32 continuous coordinates, deterministic scalar outcomes and valid finite box/log bounds. This is not noisy GP inference, constrained BO or multiobjective BO.

Training selection retains the usable incumbent, local neighbours and deterministic maximin global coverage. The durable archive retains every completion, including failures. Training-factor memory is quadratic in the training cap; factorization is cubic in that cap. Duplicate filtering and coordinate refinement still inspect archive points and can dominate at large archives: a cap is a resource bound, not a scalability guarantee. Training caps above 240 and automatic multi-region allocation are not exposed.

## Numerics and sampling

Observations are processed in ID order, so the region counters and the GP's training order do not depend on which callback of a batch completed first. Fixed-hyperparameter factor reuse compares training IDs and normalized coordinates. Re-estimated transformed targets are solved again; covariance is reused only when valid. A real observation can append the Cholesky factor with the new target. Jitter, pivot ratio, selected training IDs, reuse/refit decisions, fallbacks and the region side are recorded. Pivot ratio is not a matrix condition number.

Exact finite-candidate posterior samples are a small correctness oracle. Decoupled paths use an approximate RFF prior and an exact-kernel conditioning correction, including the model's regularization noise. The Matérn-5/2 frequencies use a multivariate t distribution with five degrees of freedom. Samples keep cross-candidate correlation. Approximation quality depends on feature count; no exact-sampling claim is made.

Sobol candidates use the 32-bit Joe–Kuo direction numbers of the first 32 dimensions as SciPy 1.17.0 initializes them (BSD-3-Clause; see the [Joe–Kuo](../../crates/fcmaes-surrogate/third_party/JOE_KUO_LICENSE.txt) and [SciPy](../../crates/fcmaes-surrogate/third_party/SCIPY_LICENSE.txt) notices), and a seeded digital shift. The unscrambled points equal SciPy's for the first 1024 points in every dimension. This is not Owen scrambling. One-dimensional bin occupancy and the applicable two-dimensional 256-point net are checked; no blanket discrepancy guarantee is inferred for every dimension/projection. Startup retains the centre/maximin design so comparisons do not silently change initialization.

## Scheduling and stopping

Synchronous work alternates fitting and callbacks. Asynchronous work runs at most one numerical proposal/refit task while callbacks execute; proposals use a recorded observation/pending snapshot. Late completions enter the next decision. A bounded evaluation pool carries nested Rayon callbacks; there is no full-size private pool per callback.

Reserve `concurrent * evaluation_threads + surrogate_threads <= inner_threads` for async, or `max(concurrent * evaluation_threads, surrogate_threads) <= inner_threads` for sync. The worker currently caps inner_threads at eight. Coordinator/blocked transport threads do not create an additional numerical pool. A callback must obey its thread allowance; it is not a per-callback OS quota.

Stop prevents further dispatch, abandons queued proposals and drains active work. Simulations cooperate through existing SimContext checkpoints; plain callbacks must finish or be stopped by the runtime process deadline. The largest uninterruptible GP operation is a bounded factorization/gradient block at the training cap. Completed, failed, cancelled and validation work remain separate.

BiteOpt handoff uses normalized coordinates, including log variables, and the best observed incumbent. Cached points are told back without a new callback and emit a Cached event. Actual callback budget remains global; final partial work and immediately preceding improvements are retained. Only BiteOpt handoff is exposed in this release.

## Trace, warm inputs and claims

Events use `fcmaes.surrogate.events/2`; warm inputs use `fcmaes.surrogate.observations/2`. V1 decoders remain available. The worker binds model/version, parameters, inputs and simulation run seed, drops out-of-bounds warm points and charges only new calls. A warm start rebuilds the optimizer; it is not exact process resume.

`advanced::replay` regenerates BO proposals from recorded history/pending snapshots and governor actions, and checks IDs, completions, cancellations and counters. Handoff points are checked structurally, not independently re-optimized. Replay never proves objective truth. Seed-only equality is promised only where tested in synchronous mode; async decisions depend on observed event order.

The best result is an actually observed usable point. An independent deterministic exact recheck can upgrade objective/bounds evidence; cancelled validation leaves the historical claim. No result proves a global optimum or separately verifies constraints hidden inside a penalty objective.

## Reproduce

```bash
cargo test --locked -p fcmaes-surrogate -p fcmaes-job-worker
cargo build --release --locked -p fcmaes-surrogate --examples
cargo build --release --locked -p fcmaes-job-worker --example simulation_worker
python3 tutorials/surrogate/advanced.py --root /absolute/new/demo \
  --runner /absolute/optimization-run --worker /absolute/fcmaes-job-worker \
  --simulation-worker /absolute/simulation_worker
python3 scripts/surrogate/run_advanced_panel.py --binary /absolute/advanced_panel \
  --output /absolute/new/panel --confirm --seed-start 301
python3 scripts/surrogate/summarize_advanced_panel.py pack /absolute/new/panel panel.jsonl.gz
python3 scripts/surrogate/summarize_advanced_panel.py summary panel.jsonl.gz
python3 scripts/surrogate/verify_advanced_evidence.py
```

The harness retains failed/censored cells and resumes only matching unfinished cells, with two worker processes and a 90-second per-cell watchdog. Do not replace retained evidence. Hardware deployment, sustained 16-core limits and actual multi-host/reboot acceptance remain January work.
