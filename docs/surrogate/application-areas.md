# Application areas for a generic optimization job API on parallel optimizers

*October 2026. Scope: the shared optimization job API of fcmaes-rust, LP-IP-craft
and SQP-craft (job envelope `optimization.job/0.1`, result contract
`optimization.result/0.2`).*

A generic job API is most useful where the best available optimizers are
parallel, and their parallelism can be expressed with three decisions:

- how many independent tasks a job has;
- how many threads each task may use;
- how many jobs share the machine.

This document does three things:

- lists application areas and the parallel algorithm families that lead in them;
- shows how each maps onto the job API;
- says whether the mapping works today, needs a registered model, or needs a new
  profile or capability.

**Status labels used throughout:**

| Status | Meaning |
| --- | --- |
| **Ready** | Works today with a file input; no code change. |
| **Model needed** | The profile exists, but the application's model must be compiled into a worker as a registered model. Jobs cannot carry callbacks. |
| **Profile needed** | The algorithm or capability is not exposed by any job profile yet. |

## 1. How parallel algorithms map onto the API

| Level | API construct | Parallel patterns it carries |
| --- | --- | --- |
| Across tasks | One task per run, start, point or scenario; `outer_workers` run in parallel | Independent restarts (IPOP/BIPOP-style CMA-ES restarts, BiteOpt/DE/CMA-ES retries), multistart NLP, MIP start portfolios, independent MODE or MAP-Elites runs, batches of candidate evaluations, scenario subproblems |
| Within a task | `inner_threads`, with `outer_workers × inner_threads ≤ cpu_slots` | Parallel population evaluation, parallel linear algebra in simplex/interior-point/SQP, parallel tree search |
| Across jobs | Several jobs on one node, each with reserved `cpu_slots` | Algorithm portfolios and racing, configuration comparisons, mixed LP/MIP + black-box + NLP workloads on one budget |
| Agent loop | `*.evaluate/1` profiles, typed claims, evidence and refutations | LLM proposes, engine computes and checks (Split-Brain): candidate checking, structural search with parameter tuning inside |

**Current constraints:**

- One Linux node, CPU only.
- Models are registered in the worker or, for LP/MIP, passed as a hashed CSR artifact.
- fcmaes tasks use one thread each; parallelism comes from independent tasks.
- LP/MIP and SQP tasks may use up to eight numerical threads.

## 2. Algorithm families and their job mapping

| Family | Representative parallel methods | Parallel pattern | Job mapping | Status |
| --- | --- | --- | --- | --- |
| Continuous black-box, one objective | CMA-ES with restarts (IPOP, BIPOP), BiteOpt, DE, CR-FM-NES, PGPE; vectorized ES on accelerators (evosax, EvoJAX) | Restarts across tasks; population evaluation within a task | `fcmaes.retry/1` (BiteOpt, DE, CMA-ES) | Model needed. CR-FM-NES, PGPE and BIPOP: profile needed |
| Large-scale ES and policy search | OpenAI-ES, PGPE, CR-FM-NES with thousands of evaluations per generation | Population evaluation across many workers | Would need `inner_threads > 1` for fcmaes | Profile needed |
| Multi-objective | NSGA-II/III, MODE, MOEA/D, SMS-EMOA | Population evaluation; independent runs | `fcmaes.mode/1`, claim `non_dominated_set` with exact hypervolume | Model needed |
| Quality diversity | MAP-Elites, CMA-ME, CMA-MAE, PGA-ME, accelerator QD (QDax) | Batched emitters; independent archives | `fcmaes.map-elites/1` (grid archive, SBX or Iso+LineDD), claim `elite_archive` | Model needed. CMA emitters and CVT archives: profile needed |
| Linear programming | Dual/primal simplex, barrier, concurrent LP, first-order PDLP (GPU variants such as cuPDLP) | Concurrent algorithms; parallel linear algebra | `lp-ip.solve/1` with `inner_threads` up to 8 | Ready (simplex). PDLP/GPU: profile needed |
| Mixed-integer programming | Parallel and distributed branch-and-bound (deterministic parallel trees, racing ramp-up, UG/FiberSCIP), LNS portfolios (CP-SAT) | Tree parallelism; solver and start portfolios | `lp-ip.solve/1`: tasks are MIP starts (a start portfolio); claims `numerical_optimum`, `infeasible`, objective bounds | Ready. Tree workers are serial in the profile |
| Smooth nonlinear | SQP and interior point with multistart, parallel sparse linear algebra, ADMM, progressive hedging | Multistart across tasks; scenarios across tasks | `sqp.solve/1`, claim `first_order` with recomputed KKT residuals | Model needed (three benchmark models today) |
| Surrogate and Bayesian | Batch BO (qEI, qNEHVI), TuRBO (parallel trust regions), asynchronous BO, multi-fidelity | Batches of suggestions per round | Only indirectly: an external optimizer proposes a batch, `*.evaluate/1` evaluates it in parallel | Profile needed |
| HPO schedulers | ASHA/Hyperband, population-based training | Asynchronous trials with early stopping | Not a whole-run job pattern; trial-level services (Optuna, Vizier, Ax) fit better | Out of scope |
| Combinatorial heuristics | Random-key encodings with ES/DE, parallel tempering, LNS | Independent restarts | `fcmaes.retry/1` with a registered decoder model | Model needed |
| Decomposition | Benders, column generation, progressive hedging | Parallel subproblems | One job per subproblem batch, orchestrated outside (for example by an agent) | Partial: master/subproblem orchestration not built in |

## 3. Application areas

### Summary

| # | Area | Typical problems | Leading parallel methods | Profiles | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Space mission design | Multiple-gravity-assist trajectories, constellations, target assignment | BiteOpt/CMA-ES restarts, MODE, MIP | `fcmaes.retry/1`, `fcmaes.mode/1`, `lp-ip.solve/1` | Cassini1 ready; others: model needed; assignment MIPs ready |
| 2 | Structural and mechanical design | Truss sizing, mechanisms, lightweight structures | DE/CMA-ES restarts, MODE, MAP-Elites | retry, mode, map-elites | Model needed |
| 3 | Optics, photonics, antennas | Lens design, phased-array codebooks | CMA-ES/BiteOpt restarts, robust MODE | retry, mode | Model needed |
| 4 | Electronics and circuits | Component sizing, transient drivers, topology search | Restarts, MODE, agent-driven structure search | retry, mode, `fcmaes.evaluate/1` | Model needed |
| 5 | Energy and power systems | Unit commitment, capacity expansion, dispatch, bilevel sizing, voltage control | Parallel MIP, LP (PDLP at scale), NLP multistart, robust black-box | `lp-ip.solve/1`, `sqp.solve/1`, retry, mode | MIP/LP ready; NLP and bilevel: model needed |
| 6 | Water and utilities | Pump scheduling, pressure management | DE/CMA-ES with random keys, MIP, MODE | retry, mode, `lp-ip.solve/1` | Model needed; MIP ready |
| 7 | Logistics, routing, scheduling | Routing, field service, production lines, job shops | MIP, LNS portfolios, random-key metaheuristics | `lp-ip.solve/1`, retry, evaluate | MIP ready; heuristics: model needed |
| 8 | Network and facility design | Coverage, facility location, telecom design | Parallel MIP vs metaheuristic baselines | `lp-ip.solve/1`, retry | Ready (MIP) |
| 9 | Inverse problems and calibration | Source localization, model calibration, parameter estimation | Restarts, multistart NLP | retry, `sqp.solve/1` | Model needed |
| 10 | Robotics, control, policy search | Controllers, gaits, behavior repertoires | PGPE, CR-FM-NES, CMA-ES, MAP-Elites, accelerator QD | retry, map-elites | Model needed; PGPE/CR-FM-NES: profile needed |
| 11 | Machine learning | Hyperparameters, ensembles, small architectures | BO, ASHA, CMA-ES restarts | retry, evaluate | Model needed for cheap models; expensive training fits trial-level services |
| 12 | Computational biology, stochastic systems | Robust oscillators, regulatory networks, kinetics fitting | Restarts on seeded stochastic simulation, agent topology search | retry, mode, evaluate | Model needed |
| 13 | Finance and economics | Portfolios, model calibration, strategy search | MIP/QP, multistart NLP, CMA-ES, agent-proposed strategies | `lp-ip.solve/1`, `sqp.solve/1`, evaluate | LP/MIP formulations ready; rest: model needed |
| 14 | Agent-driven discovery | LLM proposes, engine verifies | Evolutionary code search, CMA-ES/LLM hybrids | `*.evaluate/1`, retry | Model needed per domain |
| 15 | Optimization research and benchmarking | Algorithm comparison, competitions | Restarts with fixed seeds, solver portfolios | All profiles | Ready for registered benchmarks |

Every area except 15 needs its models registered in a worker. fcmaes-rust
already has 22 application tutorials whose models could be registered; they
are named per area below.

### 3.1 Space mission design

**What is optimized:**
- Multiple-gravity-assist and low-thrust trajectories (GTOP problems such as
  Cassini1, GTOC1 "Save the Earth").
- Satellite constellation access.
- Target assignment and tour scheduling (for example the ESA SpOC Keplerian
  TSP), usually as MIP or TSPTW.

**Leading parallel methods:**
- Massively restarted BiteOpt and CMA-ES on the continuous trajectory
  variables; restarts are independent, so throughput scales with cores.
- MODE for trade-offs such as Δv against flight time.
- Parallel branch-and-bound for assignment.

**Job mapping:**
- `fcmaes.retry/1`, one task per retry, `8 / 8 / 1`, claim `best_found`; or
  `fcmaes.mode/1`.
- Assignment MIPs via `lp-ip.solve/1` with a CSR artifact. Tasks are MIP
  starts, so heuristic tours can be checked as candidates.
- Agents can propose flyby sequences and let `fcmaes.evaluate/1` or a retry
  job score them.

**Status:** Cassini1 is registered today. GTOC1 and constellation models need
registration (tutorials `gtoc1`, `gtoc1-route-search`, `brahe-constellation`).
Assignment MIPs are ready.

### 3.2 Structural and mechanical design

**What is optimized:** truss topology and catalogue-section sizing; mechanism
and device design driven by a physics engine, such as a trebuchet in Rapier.

**Leading parallel methods:**
- DE and CMA-ES restarts with penalty or repair constraint handling.
- MODE for mass against stiffness or cost.
- MAP-Elites for a diverse set of feasible designs.
- At scale: surrogate-assisted EAs whose expensive evaluations run in parallel.

**Job mapping:** retry and mode profiles, one run per task. Use `elite_archive`
when designers want alternatives, not one optimum. `fcmaes.evaluate/1` checks
hand-made or catalogue designs with the same model.

**Status:** model needed (tutorials `truss-sizing`, `rapier-trebuchet`).
Expensive finite-element models would also need `inner_threads > 1` or a
surrogate profile.

### 3.3 Optics, photonics and antennas

**What is optimized:** lens prescriptions (Cooke triplet); hardware-quantized
phased-array beam codebooks with robustness to element errors.

**Leading parallel methods:** CMA-ES and BiteOpt restarts on smooth but
multimodal merit functions; robust (worst-case or expected) objectives
evaluated over scenario sets; MODE for gain against side-lobe level.

**Job mapping:**
- Retry and mode profiles.
- The scenario set of a robust objective belongs inside the registered model,
  so the evaluation stays deterministic and re-checkable.
- Phases with different questions (continuous phases, quantization, robustness)
  become separate jobs that share a node.

**Status:** model needed (tutorials `optical-lens-design`,
`phased-array-codebook`).

### 3.4 Electronics and circuits

**What is optimized:** component values against frequency or transient
specifications, gate-driver transients, and circuit topologies.

**Leading parallel methods:**
- Restarted CMA-ES/BiteOpt over component values.
- MODE for competing specifications.
- Split-Brain structural search: an LLM proposes topologies, and an inner,
  fixed-budget optimizer tunes each proposal's parameters.

**Job mapping:**
- One retry job per proposed topology, each with its own request key, so an
  agent's retries never double the cost.
- `fcmaes.evaluate/1` for netlists the agent wants checked as given.
- Refutations prevent an agent from reporting a value the model does not
  reproduce.

**Status:** model needed (tutorials `sindr-circuit-design`,
`thevenin-gate-driver`, `oscillator-topology-search`).

### 3.5 Energy and power systems

**What is optimized:**
- Unit commitment and economic dispatch (MIP).
- Capacity expansion (very large LPs).
- Energy-hub sizing as a bilevel problem: black-box outer sizing over an inner
  LP dispatch.
- Robust voltage control.
- AC optimal power flow (smooth NLP).

**Leading parallel methods:**
- Parallel branch-and-bound for commitment.
- Concurrent simplex/barrier, and first-order PDLP (on GPUs) for the largest LPs.
- Multistart interior-point or SQP for AC-OPF.
- Progressive hedging and Benders decomposition across scenarios.
- Evolutionary outer loops for bilevel and robust problems.

**Job mapping:**
- Commitment and dispatch MIPs: `lp-ip.solve/1` today. Claims `numerical_optimum`
  or `infeasible`, `objective_bounds` with the proven gap, `lp-ip.evaluate/1`
  for operator schedules.
- Scenario decomposition: one job per subproblem batch, with an external
  coordinator.
- AC-OPF: `sqp.solve/1` once network models are registered.
- Bilevel sizing: retry with a registered model that solves the inner LP.

**Status:**
- Ready: LP and MIP.
- Model needed: NLP and bilevel (tutorials `energy-hub-bilevel`,
  `rustpower-voltage-control`).
- Profile needed: PDLP-scale LP and decomposition.

### 3.6 Water and utilities

**What is optimized:** pump schedules and pressure management with two
objectives (energy cost against service level) under demand uncertainty.

**Leading parallel methods:** DE or CMA-ES over random-key or continuous
schedule encodings, MODE for the trade-off, MIP for linearized models.

**Job mapping:** mode or retry with a registered hydraulic model; MIP
formulations through `lp-ip.solve/1`.

**Status:** model needed (tutorial `water-network-scheduling`); MIP ready.

### 3.7 Logistics, routing, scheduling and manufacturing

**What is optimized:** vehicle and field-service routing with time windows and
disruptions; production lines evaluated by discrete-event simulation; job-shop
and timetabling problems.

**Leading parallel methods:**
- Exact or near-exact MIP with parallel trees.
- Portfolios of large-neighborhood search (CP-SAT style).
- Random-key decoders driven by CMA-ES/DE restarts.
- Simulation-based optimization with parallel replications.

**Job mapping:**
- Formulations: `lp-ip.solve/1` (ready). Tasks are warm starts, so routes from
  dispatchers or heuristics are evaluated as candidates.
- Decoder-based search: retry with a registered decoder.
- Agent- or planner-proposed plans: `lp-ip.evaluate/1` and `fcmaes.evaluate/1`.
- Simulation models must be deterministic per seed for re-checks to hold.

**Status:**
- Ready: MIP.
- Model needed: decoders and simulators (tutorials `field-service-routing`,
  `nexosim-production-line`).
- Not provided: an LNS or CP portfolio.

### 3.8 Network and facility design

**What is optimized:** weighted coverage, facility location, telecom network
design.

**Leading parallel methods:** parallel MIP for proofs of optimality; restarted
metaheuristics as fast baselines.

**Job mapping:** run both and compare honestly. The MIP job reports
`numerical_optimum` with objective bounds, the metaheuristic job reports
`best_found`, and the result contract keeps them apart.
`certified_optimum` would need exported VIPR certificates.

**Status:** ready for MIP (tutorial `network-coverage` shows the value of a
certified baseline); metaheuristic models need registration.

### 3.9 Inverse problems and calibration

**What is optimized:** emission-source localization from sensor data; model
calibration in physics, hydrology and epidemiology; parameter estimation.

**Leading parallel methods:** many restarts of CMA-ES/BiteOpt, since
multimodal likelihoods need them; multistart least squares or SQP for smooth
models; Bayesian optimization for expensive simulators.

**Job mapping:**
- Retry jobs with deterministic seeds. The spread of restart results is a cheap
  identifiability check (distinct optima with equal value).
- `sqp.solve/1` for smooth calibration once models are registered.

**Status:** model needed (tutorials `dispersion-source-localization`,
`cfd-room-ventilation`).

### 3.10 Robotics, control and policy search

**What is optimized:** fixed-topology neural controllers, gait parameters, and
behavior repertoires for damage recovery or task variety.

**Leading parallel methods:**
- Natural evolution strategies (PGPE, CR-FM-NES) and CMA-ES with large
  populations.
- MAP-Elites, CMA-ME and CMA-MAE.
- Accelerator-parallel QD (thousands of rollouts per generation).

**Job mapping:**
- `fcmaes.map-elites/1` gives a re-evaluated `elite_archive`: every elite's
  descriptor and niche are rechecked, which matters when the archive is the
  deliverable.
- Retry for single controllers.

**Status:**
- Model needed: simulator models (tutorials `neural-controller-policy-search`,
  `rapier-quadruped-gait`).
- Profile needed: PGPE and CR-FM-NES, CMA emitters, and parallel population
  evaluation.

### 3.11 Machine learning

**What is optimized:** hyperparameters, ensemble weights, small architectures,
data-augmentation policies.

**Leading parallel methods:** batch and asynchronous Bayesian optimization,
ASHA/Hyperband with early stopping, population-based training, and CMA-ES and
TPE (which a 2026 study found ahead of pure LLM agents on an autoresearch
benchmark).

**Job mapping:**
- Cheap models trained inside a registered worker: retry with deterministic
  seeds.
- Expensive training: trial-level services (Optuna, Vizier, Ax) fit better.
  The job API can evaluate a proposed batch of configurations with
  `fcmaes.evaluate/1`.

**Status:** model needed (tutorial `ml-hyperparameter-tuning`); early-stopping
schedulers are out of scope.

### 3.12 Computational biology and stochastic systems

**What is optimized:** robust stochastic oscillators (Gillespie simulation),
regulatory-network topologies, kinetic parameter fitting.

**Leading parallel methods:** restarts over seeded stochastic simulations
(common random numbers), MODE for robustness against performance, and
agent-proposed topologies with an inner parameter optimizer.

**Job mapping:**
- Retry and mode with registered, seeded simulators.
- `fcmaes.evaluate/1` for agent-proposed networks.

**Caveat:** re-evaluation checks assume a deterministic model. Stochastic
models must fix their simulation seeds inside the model, or claims would be
refuted spuriously.

**Status:** model needed (tutorials `rebop-oscillator`,
`oscillator-topology-search`).

### 3.13 Finance and economics

**What is optimized:** portfolio selection (cardinality and lot-size
constraints), model calibration, trading-strategy search.

**Leading parallel methods:**
- MIP for combinatorial portfolio constraints.
- QP/SQP for smooth mean-variance and risk-parity models.
- Multistart NLP and CMA-ES for calibration.
- Agent-proposed strategies scored by deterministic backtests.

**Job mapping:**
- Linear and MIP portfolio models: `lp-ip.solve/1`, ready.
- Calibration: `sqp.solve/1` or retry with registered models.
- Strategy search: evaluate profiles with a registered backtester.

**Caveat:** evidence covers the computation, not out-of-sample validity.
Report the backtest protocol separately.

**Status:** ready for LP/MIP formulations; model needed otherwise.

### 3.14 Agent-driven discovery

**What is optimized:** anything where an LLM proposes structures (programs,
topologies, sequences, strategies) and an engine evaluates or tunes them.
Examples are AlphaEvolve-style evaluator loops and CMA-ES/LLM hybrids, in which
the LLM overrides part of the optimizer's proposals.

**Why the job API fits:**
- Durable job handles survive agent restarts.
- Content-checked request keys stop accidental re-runs.
- Cancellation and deadlines enforce budgets.
- Evaluate profiles give independent scores.
- Refutations make fabricated results visible instead of silent.

The MCP Tasks mapping in the shared guide shows how to expose this to agent
hosts.

**Status:** model needed per domain; an MCP server is not built yet.

### 3.15 Optimization research and benchmarking

**Uses:**
- Algorithm comparisons with fixed per-task seeds and complete work accounting.
- Competition entries (GTOC, ESA SpOC) whose results must be reproducible.
- Solver portfolios on one node.
- Mixed LP/MIP, black-box and NLP studies under one CPU budget.

**Job mapping:** all profiles. `compare` refuses mismatched experiments, and
journals and manifests are the raw data.

**Status:** ready for the registered benchmarks (sphere, Rosenbrock,
Cassini1, ZDT1, DTLZ2, Binh-Korn, the projection QD benchmarks, the SQP
benchmarks, and any LP/MIP artifact). Best-so-far traces for
IOHanalyzer/COCO are not emitted yet.

## 4. What the job API adds, by feature

| Feature | Why it matters | Areas where it matters most |
| --- | --- | --- |
| Content-checked request keys | A retried submission returns the same job and never silently runs or returns different content | Expensive simulations (3.2, 3.9, 3.10), agent loops (3.4, 3.14) |
| CPU-slot admission | Mixed workloads share one machine without oversubscription | Energy (3.5), logistics (3.7), research (3.15) |
| Deterministic task seeds | Reproducible results, independent of scheduling | Competitions (3.1), benchmarking (3.15), calibration (3.9) |
| Claims, evidence, refutations | What is asserted, how much was verified, and which solver claims failed | LP/MIP areas (3.5, 3.7, 3.8), agent loops (3.14), finance (3.13) |
| Set claims | Pareto fronts and elite archives as first-class, re-evaluated results | Design trade-offs (3.2, 3.3, 3.6), robotics (3.10) |
| Evaluate profiles | Independent checking of proposed candidates | Routing (3.7), circuits (3.4), strategies (3.13), discovery (3.14) |
| Journals and recovery | Committed tasks survive crashes, with honest partial accounting | Long restart campaigns (3.1, 3.9) |
| Cooperative cancellation and deadlines | Anytime algorithms return their best so far | MODE, MAP-Elites, MIP time limits (3.2, 3.5, 3.10) |

## 5. Gaps that limit coverage, by reach

1. **Application model registration.** Most areas need their own model. A
   documented path from an application crate to a registered model (or a
   strictly specified, hashed external-simulator protocol) would unlock areas
   2–4, 6, 7 and 9–13.
2. **Parallel population evaluation in fcmaes** (`inner_threads > 1`), for
   expensive simulations where independent restarts are not enough.
3. **More algorithms behind profiles:** CR-FM-NES, PGPE, BIPOP restarts, CMA-ME
   and CMA-MAE emitters, CVT archives.
4. **Accelerator profiles:** PDLP-class LP and vectorized ES/QD on GPUs.
5. **Multiple nodes:** a scheduler backend (Slurm, Flux) behind `cpu_slots`.
6. **Surrogate and batch Bayesian optimization** for evaluations that cost
   minutes or hours.
7. **Decomposition orchestration:** Benders, column generation and progressive
   hedging over jobs.
8. **Certificates:** exported Farkas rays for LP infeasibility; VIPR for MIP
   (`certified_optimum`).
9. **Stochastic objectives:** evidence semantics for seeded or averaged
   re-evaluation.
10. **Agent access:** an MCP server with the Tasks extension, using the mapping
    already documented.

## 6. Selected background

- Hansen and Ostermeier (2001), CMA-ES; Auger and Hansen (2005), IPOP-CMA-ES;
  Hansen (2009), BIPOP-CMA-ES.
- Storn and Price (1997), differential evolution. Vaneev, BiteOpt
  (github.com/avaneev/biteopt).
- Deb et al. (2002), NSGA-II.
- Mouret and Clune (2015), MAP-Elites; Fontaine et al. (2020), CMA-ME;
  Fontaine and Nikolaidis (2023), CMA-MAE; Lim et al. (2022), accelerated
  quality diversity (QDax).
- Salimans et al. (2017), evolution strategies at scale; Sehnke et al. (2010),
  PGPE; Nomura and Ono (2022), CR-FM-NES.
- Applegate et al. (2021), PDLP; Lu and Yang (2023), cuPDLP.
- Shinano et al., UG/ParaSCIP/FiberSCIP parallel branch-and-bound; OR-Tools
  CP-SAT portfolio search.
- Eriksson et al. (2019), TuRBO; Balandat et al. (2020), BoTorch.
- Li et al. (2020), ASHA; Jaderberg et al. (2017), population-based training.
- Rockafellar and Wets (1991), progressive hedging.
- ESA GTOP database; GTOC and SpOC competitions.
- Ferreira et al. (2026, arXiv 2603.24647), classical optimizers against LLM
  agents and CMA-ES/LLM hybrids.
- Companion report: *Optimization job APIs for AI agents* (prior art and the
  result contract 0.2).
