# Application workers for prompt tutorials

The [prompt-driven series in optimization-jobs](https://github.com/dietmarwo/optimization-jobs/tree/main/tutorials/agentic) uses four optional workers beside their original numerical models. The generic `fcmaes_job_worker::Adapter` and `Registry` handle discovery, schemas, retries, MODE, MAP-Elites, accounting and result rechecks. Each application still needs a registered model factory: a generic adapter cannot infer the physics from a prompt.

| Standalone workspace under `tutorials/` | Binary | Registered models |
|---|---|---|
| `rapier-trebuchet` | `trebuchet-job-worker` | `app.trebuchet.scalar`, `.pareto`, `.checked`, `.qd` |
| `nexosim-production-line` | `production-job-worker` | `app.production.pareto`, `.qd` |
| `energy-hub-bilevel` | `energy-job-worker` | `app.energy.scalar`, `.pareto`, `.checked`, `.qd` |
| `gtoc1-route-search` | `gtoc1-job-worker` | `app.gtoc1.mga` |

Build from a tutorial directory with `cargo build --release --locked --features job-worker`. The normal default binary remains the original tutorial CLI. Run `cargo test --release --locked --features job-worker` to check native/job parity, discovery and invalid-parameter rejection. The feature is off by default; no simulator is added to the root worker.

Register a binary with `optimization-run workers add --id ID --executable /absolute/path/to/BINARY`, then inspect `optimization-run problems describe ID`. Discovery provides payload schemas, model parameter schemas and runnable examples; do not assume a model name or bounds from another worker. Use the same `--root` for registration and subsequent jobs.

Application parameters are `target_m` for the trebuchet; `replications`, `horizon_minutes` and `simulation_seed` for production; `preset` (`smoke` or `publication`) and, on `.checked`, `scenario_set` (`training` or `holdout`) for energy; and `bodies` for GTOC1. Dimensions and domains are checked, and unused parameters are rejected. Production replications run serially inside each evaluation so the worker owns parallelism. The four workers use the existing pinned job API dependency; no protocol change is needed.

## What equivalence means

The scalar/MODE/QD wrappers call the native tutorial functions. `.checked` adds explicit engineering/physical constraints for re-evaluation; penalties alone are not feasibility evidence. Energy dispatch remains microlp in process, with exactly the original aggregate constraints on separate training and holdout scenarios. The LP-IP-craft one-hour dispatch in the prompt series is a separate teaching example, not a new dispatch backend.

The job searches use the current path-based fcmaes worker/core. Energy and GTOC1 retain the original tutorial's pinned `fcmaes-core = 0.1.3` for their existing algorithms. Objective parity does not imply identical optimizer trajectories. The production decoder still rounds integer variables, while its original MODE driver additionally sets an integer mutation mask. The generic MODE job does not expose that mask.

GTOC1 jobs optimize one grammar-valid route in direct launch/leg-duration coordinates, using the history-blind route bounds and original impulsive MGA objective. `gtoc1-route-card 3,2,3,5,10` emits those bounds and a neutral schedule for a proposed body order. Generic BiteOpt retry is a separate protocol from the original coordinated DE-CMA-ES experiment. A worker scalar `feasible` outcome checks its declared box; it does not establish low-thrust feasibility. An invalid MGA evaluation has the native wrapper penalty `1e99` and must not be treated as a useful route.

## Extend to another tutorial

Register the original scalar, multi-objective or quality-diversity function; declare its dimension/domain and any explicit constraints; validate application parameters; supply a runnable discovery example; and add parity and invalid-input tests. Keep held-out scenarios out of selection. The application worker is reusable from shell or MCP and does not depend on any particular language model.
