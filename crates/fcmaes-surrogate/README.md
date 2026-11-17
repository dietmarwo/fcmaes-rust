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

Absent from this method (`run`, `fcmaes.surrogate/1`): Thompson sampling, multiple trust regions (TuRBO-m), async, Sobol designs, handoff, noisy GP, full L-BFGS-B (Cauchy point and subspace minimization), constraints, Python bindings and stateless job sessions. The experimental `advanced` module below adds Thompson sampling, async callbacks, a Sobol design and a BiteOpt handoff. A warm start creates a new state from observations, not an interrupted-state resume. Same-host thread determinism is tested; cross-platform bit equality is not promised.

See [reviewed plan](../../docs/surrogate/PLAN.md), [job profile](../../docs/job-api/surrogate.md), [tutorial](../../tutorials/surrogate/README.md) [measured results](../../docs/surrogate/validation/REPORT.md) and the [comparison with BoTorch, egobox, scikit-optimize and Optuna](../../benchmarks/surrogate-comparison/README.md).

Experimental: `fcmaes_surrogate::advanced` and the `fcmaes.surrogate/2` job profile add an archive of up to 10,000 observations with a bounded local GP training set, asynchronous callbacks, region restarts, approximate correlated Thompson sampling, a Sobol candidate design, an overhead governor and a BiteOpt handoff. They did not pass the default-promotion gate and stay opt-in; see the [advanced guide](../../docs/surrogate/ADVANCED.md).
