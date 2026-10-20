# fcmaes-surrogate 0.1.0 (experimental, source-only)

Deterministic small-budget minimization of continuous boxes using exact ARD Gaussian processes and synchronous Kriging-believer batches. The library does not depend on the job API. It is a new workspace member, not part of the published `fcmaes-core` or Python distribution.

Install a bounded Rayon pool. All proposal and evaluation work uses the current pool; calling directly without one uses Rayon's global default pool.

```rust
use fcmaes_surrogate::{Config, RunSpec, Space, run};

let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
let space = Space::new(vec![-2.0; 2], vec![2.0; 2], vec![false; 2])?;
let config = Config::default();
let result = pool.install(|| run(
    RunSpec {
        space: &space, config: &config, evaluations: 48,
        concurrent: 4, seed: 41, warm: &[], target: None,
    },
    &|x| Ok(x.iter().map(|v| v * v).sum()),
    &mut |_event| Ok(()), // supply a durable, fallible sink for real simulations
    &|| None,
))?;
assert_eq!(result.evaluations, 48);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Supported: 1–32 coordinates, at most 256 new plus warm observations, batches 1–8, Matérn-5/2/RBF, configurable fixed nugget (default 1e-10), MAP fit by bounded L-BFGS with the analytic gradient (packed row-major Cholesky, deterministic summation order), LogEI/LCB maximized by coordinate refinement of the best candidates, a TuRBO-1-style trust region (lengthscale-shaped box around the incumbent, success/failure resizing; every third proposal searches the whole box), standardize/bilog/auto, seeded centre/maximin LHS, fallible callbacks and semantic event replay. Invalid input errors propagate. GP fit failures use a logged design fallback; exhausted distinct candidates return an error. Cancelled calls are separate from completed callbacks and failed outcomes. Replay checks decisions from recorded observations; it never proves objective correctness.

Absent: Thompson sampling and multiple trust regions (TuRBO-m), async, noisy GP, Thompson sampling, Sobol, full L-BFGS-B (Cauchy point and subspace minimization), constraints, handoff, Python bindings and stateless job sessions. A warm start creates a new state from observations, not an interrupted-state resume. Same-host thread determinism is tested; cross-platform bit equality is not promised.

See [reviewed plan](../../docs/surrogate/PLAN.md), [job profile](../../docs/job-api/surrogate.md), [tutorial](../../tutorials/surrogate/README.md) [measured results](../../docs/surrogate/validation/REPORT.md) and the [comparison with BoTorch, egobox, scikit-optimize and Optuna](../../benchmarks/surrogate-comparison/README.md).
