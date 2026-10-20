# Surrogate first-version implementation log

Date: 2026-10-20. Status: **implemented and tested experimental milestone M0**. This project follows the supplied revised plan's Rust-simulation focus, with an explicit smaller first delivery. It does not claim completion of asynchronous TuRBO, large-budget handoff, or the full multi-week roadmap.

## Review revision (2026-10-20)

The delivered M0 was committed unchanged (`4dd693b`, without the two delivery-only root files and the raw evidence) and then revised:

- **MAP fit.** The coordinate pattern search over log lengthscales (up to 2d factorizations per sweep, 66–1026 per fit) is replaced by a bounded L-BFGS with the analytic gradient of the negative log posterior (`optim.rs`, `gp.rs`). The gradient is checked against central differences for both kernels; the fit reaches an equal or lower NLP in 46 of 50 overhead configurations with 19–104 factorizations. Config `fit_sweeps` (1–24) is replaced by `fit_iterations` (default 50, 1–200).
- **Linear algebra.** nalgebra's Cholesky is replaced by `linalg.rs`: packed row-major lower triangle, contiguous eight-accumulator dot products in fixed order, row-wise solves, O(n) row append for fantasies, `L^-1`/`K^-1` rows for the gradient. The kernel matrix is filled once per lengthscale vector from pre-scaled inputs, symmetric half only. The crate no longer depends on nalgebra. The independent 100-digit fixtures still pass at 1e-10.
- **Simulation identity.** Snapshots recorded the payload seed as `simulation_seed` although simulations ran with `retry_run_seed(seed, task)`, so a task-k snapshot of a multi-run job could warm-start a single-run job that simulates with a different seed. The identity now records the task's run seed and warm starts compare against task 0's (a test covers both directions).
- **Hygiene.** `Scalar::simulation`'s placeholder plain objective panics instead of silently returning `+inf`; the simulation worker example and `analyze.py` were rewritten for readability; the ai-context header was restored with the surrogate section added below it.
- **Measured and not adopted:** a gradient (L-BFGS) acquisition refinement and an incumbent update inside Kriging-believer batches; both lost on Styblinski–Tang-10 (details in [decisions.md](decisions.md)).

First-pass results: a fresh fit plus one proposal at d=32, n=248 went from 4.6 s to 0.49 s on one thread (9.4x), at d=10 from 0.90 s to 0.19 s (4.8x); eight-point batch proposals with carried hyperparameters became about 1.3–1.5x faster at n=248. The faster fit alone did not change quality systematically (per-seed outcomes split about evenly). [validation/REPORT.md](validation/REPORT.md) now compares M0 with the build after the second pass below: same speedups, and better final regret on 4–5 of 5 Branin/Hartmann-3 seeds, 20 of 20 Hartmann-6 seeds, 15 of 20 Styblinski–Tang-10 seeds and 13 of 20 oscillator seeds; histories are identical on 1, 2 and 8 threads.

Gates rerun for the revision: `cargo fmt --all -- --check`, clippy with `-D warnings` on both crates, release tests of both crates, the MSRV 1.88 check, `scripts/surrogate/test_job_api.py`, the tutorial, and the shared runner integration harness (see the pull request for the exact outputs).

## Second review pass: comparison with established libraries (2026-10-20)

[benchmarks/surrogate-comparison](../../benchmarks/surrogate-comparison/README.md) runs BoTorch (default and noiseless), egobox, scikit-optimize, Optuna's GP and TPE samplers and fcmaes-surrogate on Branin, Hartmann-3/6, Styblinski–Tang-10, Rosenbrock-8 and the tutorial's RK4 oscillator at equal call budgets, ten seeds, q=1 and q=4. The comparison exposed three weaknesses, all addressed:

- **Precision near the optimum.** The fixed nugget of 1e-8 limited final precision (Branin 1.9e-5 against egobox's 6.6e-8). The nugget is now configurable (`nugget`, default 1e-10); the GP fixtures were regenerated for the new diagonal.
- **Global-only search at d = 6–10.** A TuRBO-1-style trust region now drives two of every three proposals (`trust_region`, default true); the third searches the whole box, which keeps runs out of local traps. Hartmann-6 and Styblinski–Tang-10 improved on every seed.
- **A wrong Hartmann-3 reference.** The quoted minimum -3.86278214782076 is not attainable with the standard constants; the true minimum is -3.8627797873326624. The benchmark, worker metadata and reports use the corrected value.

With these changes fcmaes-surrogate has the lowest median final regret among the GP methods on every problem except Branin (egobox), at 0.02–1 s of proposal time per run against 1–200 s for the other GP libraries. BiteOpt remains better on Styblinski–Tang-10. Tests: 53 Rust tests in the surrogate crate and worker plus the existing suites; determinism on 1/2/3/8 threads holds with the trust region.

## Source provenance and deliverables

| Item | Exact basis |
| --- | --- |
| Project | https://github.com/dietmarwo/fcmaes-rust |
| Upstream base | `92aaf03f9f03257a284126a322160d045422bc9a` |
| Shared optimization-jobs dependency | `eb1c9de66e677097a7ea22d9a4b2f6357c18751d`, unchanged pin |
| New crate | `fcmaes-surrogate` 0.1.0, source-only, publish=false |
| Existing workspace/core/job-worker | Version remains 0.1.5 |
| Common contracts | `optimization.job/0.1`, `optimization.result/0.2` |
| New worker profile | `fcmaes.surrogate/1` (experimental current compact payload) |
| Supplied material | Original plan, newly supplied second plan, and application-areas document preserved in `docs/surrogate/` |

The ZIP contains the full repository source tree, existing documentation/data, the new library/profile/tests/tutorials, raw benchmark evidence, this log, the reviewed plan and a patch against the exact base. It excludes Git history, Cargo build outputs and local caches. No repository was pushed and no dependency source was patched. Downloading/building requires normal access to Cargo dependencies and the pinned public shared repository. The implementation does not add fcmaes-surrogate to the published Python API.

## Work ledger: plan, implementation and gate

| Step | Plan | Implemented | Verification / outcome |
| --- | --- | --- | --- |
| 1. Reconcile plans | Preserve new sub-minute simulation priorities without calling all P1 features complete. | Revised PLAN with F1–F27 ledger, explicit M0 and M1–M5 gates; preserved both inputs. | Document/source comparison; future API proposals clearly separated from supported payload. |
| 2. Numerical foundation | Establish exact, bounded GP primitives before scheduling complexity. | Normalized/log boxes; standardize/bilog/auto; RBF/Matérn-5/2 ARD GP; lognormal MAP prior; bounded restart coordinate search; Cholesky/jitter and fantasy append; stable LogEI/LCB. | Independent 100-decimal fixtures for posterior/NLP and LogEI tails; duplicate and constant data, kernel/prior checks, fantasy/full-refit equality, extreme transforms. Passed. |
| 3. Deterministic proposals | Make every proposal a function of explicit state/history/seed. | Seeded centre/maximin LHS, global/local candidates, coordinate refinement, pending KB fantasies, separation checks and logged fit fallback. | Same histories/proposals on 1/2/3/8 threads; stop leaves input state unchanged; pending points not repeated. Passed. |
| 4. Driver and durability | Log completions before waiting for stragglers; propagate sink errors. | Started/completed/cancelled events, ordered batch commit, complete-trace semantic replay, failure classification, stop/target budgets. | Fast completion observed before slow callback finishes; logger failure propagation; NaN/panic/Err/penalty paths; mutation, missing finish, repeated start and bad counter rejection. Passed. |
| 5. Job integration | Reuse common SDK/runner contracts and registry. | Profile validation/discovery; task-owned bounded pool; model factory resolution; checksummed fsync trace; atomic compact snapshots; hashed identity-bound warm inputs. | Actual SDK execution, strict options/resources/model kind, unused inputs, input corruption, wrong identity/version, narrowed bounds and new-call-only warm accounting. Passed. |
| 6. Simulation API | Support fallible checkpointed Rust callbacks with explicit capacity. | Scalar::simulation and SimContext checkpoint/remaining_seconds/threads/point seed; resource product validation; surrogate-only support. | Cooperative cancellation, nested pool cap, stable recheck seed, explicit retry rejection; actual CLI cancellation. Passed. Cross-profile extension deferred. |
| 7. Claims and rechecks | Report actual observations and preserve evidence semantics. | Exact best-point recheck, separate validation count; mismatch refutation and untrusted snapshot; cancellation-aware validation. | Changing objective refuted; seeded deterministic model rechecked; cancelled recheck stays claimed and unrefuted. Passed. |
| 8. Runner/MCP/tutorial | Exercise real executables, not only direct library calls. | MCP submit/wait/result test, actual runner cancel test, Branin/warm/RK4 tutorial, new CI commands. | Both new Python tests passed; all three tutorial jobs completed with complete accounting and rechecked evidence. |
| 9. Quality/overhead | Use equal evaluation budgets and disclose overhead. | Four-method benchmark, raw observations/timings, deterministic 1/8-thread comparison, synthetic-latency campaign, report/figure script. | 96 runs total; all 40 paired 1/8-thread histories exactly equal. GP improves small-function final quality but costs more proposal time. No general wall-time win claimed. |
| 10. Reproducible delivery | Keep source, source diff, plans, logs and evidence together. | Full-project ZIP, current API and tutorial docs, ai-context addendum, this log, source manifest. | Archive integrity and source extraction/build verification recorded in delivery-checks below. |

## Numerical and semantic fixes retained during review

- Updated fantasy append's diagnostic negative log posterior, checked against a fresh GP factorization.
- Enabled exact floating-point JSON round trips so replay sees identical observations/state.
- Classified failures before transformation and retained raw failures without invented GP responses.
- Added immediate durable completion events and atomic warm snapshots; failures in either stop unstarted work and propagate.
- Strengthened replay to reject incomplete streams, unmatched/duplicate starts and inconsistent final counts, in addition to regenerated proposal/state mismatches.
- Separated cancelled partial calls from completed evaluations; their elapsed time is retained in evaluation_seconds_sum. The sum is callback wall time, not CPU seconds.
- Bound simulation resource validation by concurrent * evaluation_threads; named warm inputs are explicitly owned by the profile while other unused inputs remain errors.
- Bound warm identity to resolved parameters/model data and simulation seed; rechecked mismatches invalidate warm trust.
- Prevented cancellation during final validation from becoming a false objective_mismatch refutation.
- Allowed the library benchmark to serialize a fixed four-point logical batch on one thread, while retaining the stricter actual job reservation rule.

## Tests run for the M0 delivery

The M0 delivery archive carried the raw logs; they are not committed. These are targeted delivery gates, not a claim that every existing workspace/Python/tutorial/publication campaign was rerun.

| Gate | Command / scope | Result |
| --- | --- | --- |
| Formatting | `cargo fmt --all -- --check` | Passed |
| Clippy | `cargo clippy --locked -p fcmaes-surrogate -p fcmaes-job-worker --all-targets -- -D warnings` | Passed |
| Release tests | `cargo test --release --locked -p fcmaes-surrogate -p fcmaes-job-worker` | 43 non-doc tests plus 1 doc test passed; zero failed |
| MSRV | `cargo +1.88.0 check --locked -p fcmaes-surrogate -p fcmaes-job-worker --all-targets` | Passed |
| Executables | Locked release runner/worker and simulation_worker example builds | Passed |
| New real runner/MCP tests | `python3 scripts/surrogate/test_job_api.py` | 2 passed |
| Common runner integration | Pinned shared `scripts/job-api/test_integration.py`, schema validation required, actual fcmaes binaries | 12 passed, 2 skipped: LP and SQP binaries not supplied |
| Executable tutorial | `python3 tutorials/surrogate/run.py --root FRESH_NODE` | Branin, warm and RK4 jobs passed |
| Python scripts | `python3 -m py_compile scripts/surrogate/*.py tutorials/surrogate/run.py` | Passed |
| Report consistency | `python3 scripts/surrogate/analyze.py --plot` | 40 paired histories/objectives equal; report and scientific SVG generated |

The release suite includes 26 existing job-worker tests, seven new surrogate/simulation integration tests, ten library tests and one existing registry doc test. Tests execute real numerical callbacks; the separate Python integration checks launch real runner/worker/MCP processes. No real systemd service containment, mixed LP/SQP node run, or independent external code review was performed.

## Measured results of the M0 delivery (before review)

These numbers describe the delivered code; the review revision above and [validation/REPORT.md](validation/REPORT.md) supersede them. Hardware/runtime: Linux x86_64, Rust 1.99.1/LLVM 23.1.1; Rust 1.88 checked; Python 3.12.14. Nine logical CPUs are visible, with `cpu.max = 800000 100000` (eight-CPU quota). Installed pools use one or eight threads. Host is shared, so timing numbers are observations, not portable guarantees.

Five fixed seeds (7,11,19,23,41), 48 optimization callbacks, batch four, shared eight-point centre/LHS initialization, valid CMA-ES population four. Both benchmark functions minimize; lower objective/regret is better.

| Problem | Method | Median final objective | Median simple regret | Median wall ms on 8 threads |
| --- | --- | ---: | ---: | ---: |
| Branin | GP | 0.397893097 | 0.0000057393 | 17.639 |
| Branin | Random | 1.041590566 | 0.64370 | 0.718 |
| Branin | CMA-ES | 0.5940467512 | 0.19616 | 0.797 |
| Branin | BiteOpt | 0.508655231 | 0.11077 | 0.949 |
| Hartmann-3 | GP | -3.862772089 | 0.000010058 | 19.344 |
| Hartmann-3 | Random | -3.394573601 | 0.46821 | 0.759 |
| Hartmann-3 | CMA-ES | -3.724709910 | 0.13807 | 0.807 |
| Hartmann-3 | BiteOpt | -3.846104039 | 0.016678 | 0.977 |

All 40 corresponding one/eight-thread runs have exactly equal complete observation histories and final objectives. With one installed thread the logical batch still has four members, evaluated serially. GP median wall time was 29.593 ms (Branin) and 33.218 ms (Hartmann-3) on one thread. Eight-thread observations were about 1.68x/1.72x faster for this small GP workload; this is not eight-core linear speedup or a repeated dedicated performance study.

An additional 16-run campaign added 10 ms artificial delay per call, two seeds, same budgets/batches. GP median wall time was 144.845/155.175 ms versus about 123–135 ms for baselines, while retaining better final objective quality on these cases. This exercises scheduling and illustrates overhead; it does not establish a crossover on real 1–60 second Rust simulations. Raw JSON and a convergence figure were part of the delivery archive and are not committed.

The actual durable tutorial produced Branin 0.39789079962105767, eight new warm evaluations from 48 reused observations, and RK4 calibration loss 2.4197300658564477e-6. The quick oscillator exercises real Rust integration/checkpoints and nested capacity, but it is not the held-out expensive simulation campaign. The tutorial's exact output was part of the delivery archive.

## Build and reproduce

From the extracted fcmaes-rust root:

```sh
cargo build --release --locked -p optimization-runner -p fcmaes-job-worker
cargo build --release --locked -p fcmaes-job-worker --example simulation_worker
python3 tutorials/surrogate/run.py --root /tmp/fcmaes-surrogate-fresh
cargo test --release --locked -p fcmaes-surrogate -p fcmaes-job-worker
python3 scripts/surrogate/test_job_api.py
```

See [decisions.md](decisions.md) for benchmark commands and fixture regeneration. Standard Python suffices for tutorials/tests/report generation; matplotlib is optional for the figure, and mpmath is optional for fixture regeneration. Existing upstream workspace components retain their own dependencies.

## Limitations and next work

This delivery is a synchronous deterministic small-budget baseline. No asynchronous dispatch, background refits, proposal queue, governor, TuRBO, pathwise TS, CMA-ES/BiteOpt handoff, noisy GP, Sobol, L-BFGS-B, rank transform, constraints or stateless proposal service is implemented. The library supports explicit pending fantasies but that alone is not an async scheduler.

Only fcmaes.surrogate/1 accepts Scalar::simulation. Other profiles reject it until adapted. Existing Scalar::new constructors retain behavior; external direct Scalar struct literals need migration due to private simulation storage. Resource allowances require simulation cooperation; callbacks creating their own pools can oversubscribe. Cancellation is checkpoint-based and plain callbacks finish or face the process deadline. No thread-scaling evidence above eight exists.

Local traces/snapshots are not common registered/paged MCP artifacts. Warm start consumes a compact hashed observation snapshot and starts a new optimizer state, not an exact checkpoint or full-log resume. Checksums/replay/recheck are distinct checks and none certifies a global optimum. Model versioning and deterministic factory behavior are application responsibilities.

The next milestone is cross-profile simulation semantics plus real 1–60 s baseline campaigns; then pathwise TS/TuRBO/handoff, async/governor, shared artifacts and held-out product gates. [PLAN.md](PLAN.md) gives concrete acceptance criteria for each. No future performance improvement is asserted before it is measured.

## Delivery checks

- The source archive passed ZIP CRC verification and extracted successfully.
- Every extracted payload file matched the source SHA-256 manifest.
- The extracted source passed locked all-target checks for fcmaes-surrogate and fcmaes-job-worker using the existing dependency cache.
- The source patch passed git apply --check against an archive of the exact upstream base.
- Final packaging changes after that build are documentation, logs and manifest only; no numerical or adapter code changed.
- New documentation relative links and source whitespace checks passed.
