# Optimizing native Rust applications with fcmaes-rust

## 1. Introduction

These tutorials show how to put a native Rust simulation or model-fitting
pipeline directly inside an `fcmaes-rust` objective. There is no Python
callback or serialization boundary in the expensive path: a candidate is
decoded and scored in Rust, and independent candidates are distributed across
native worker threads.

The fourteen applications deliberately cover different reasons for choosing
gradient-free optimization:

| Tutorial | Simulator | Problem property | Implemented formulations | QD decision |
|---|---|---|---|---|
| [Production line](nexosim-production-line/) | NeXosim | stochastic discrete events and mixed controls | MODE + MAP-Elites | accepted |
| [Trebuchet](rapier-trebuchet/) | Rapier | contact, release and joint-limit discontinuities | BiteOpt retry + MODE + MAP-Elites | accepted |
| [Biochemical oscillator](rebop-oscillator/) | ReBop | intrinsically noisy stochastic reaction paths | BiteOpt retry + MODE + MAP-Elites | accepted |
| [Satellite constellation](brahe-constellation/) | Brahe | access-window appearance/disappearance and worst-gap aggregation | BiteOpt retry + constrained MODE + MAP-Elites | accepted |
| [Voltage control](rustpower-voltage-control/) | RustPower | mixed-integer controls, contingencies and power-flow failures | constrained MODE + MAP-Elites | MODE primary, QD secondary |
| [Atmospheric source localization](dispersion-source-localization/) | ISC-3-derived native model | inverse inference, censoring, model mismatch, and non-identifiability | BiteOpt advanced retry + MODE + MAP-Elites | accepted |
| [Room ventilation](cfd-room-ventilation/) | Custom D2Q9/D2Q5 Rust backend | variable geometry, numerical constraints, worst-case releases, and resolution sensitivity | BiteOpt retry + MODE + MAP-Elites | accepted |
| [ML hyperparameter optimization](ml-hyperparameter-tuning/) | SmartCore decision trees | mixed variables, nested stochastic fitting, validation overfitting, and probability quality | BiteOpt retry + constrained MODE + MAP-Elites | rejected: coverage and diversity pass, niche retention fails |
| [Neural controller policy search](neural-controller-policy-search/) | Native stochastic cart-pole model | 118-dimensional fixed-topology policy, randomized rollouts, and validation variance | PGPE + CR-FM-NES + active CMA-ES/BiteOpt comparison | omitted: one robust controller is the deliverable |
| [GTOC1 “Save the Earth”](gtoc1/) | pykep-core astrodynamics | 87 variables, narrow equality constraints, two multi-revolution Lambert arcs, and low thrust | coordinated DE–CMA-ES + seeded CMA-ES/BiteOpt retry | omitted: one maximum-impact trajectory is the deliverable |
| [Circuit design](sindr-circuit-design/) | sindr AC modified-nodal analysis | log-scaled components, interpolated Bode features, competing filter goals, and E12 discreteness | CMA/DE/Bite retry + constrained MODE + MAP-Elites | accepted: a robust frequency/gain catalogue is the deliverable |
| [Transient gate driver](thevenin-gate-driver/) | thevenin transient modified-nodal analysis | interpolated edge measurements, ringing, current/settling constraints, and independent simulator validation | constrained MODE | omitted: one continuous engineering trade-off front is the deliverable |
| [Optical lens design](optical-lens-design/) | Pure-Rust sequential geometric ray tracer | multimodal bending, hard ray-loss boundaries, and a published prescription gate | CMA/DE/Bite retry + constrained MODE | omitted: a quality/size/material trade-off front is the deliverable |
| [Quadruped gait](rapier-quadruped-gait/) | Rapier 3D | contacts, falls, actual motor work, and terrain overfitting | BiteOpt retry + MAP-Elites | accepted: behavior diversity is the primary deliverable |

Each directory is a standalone Cargo workspace. This keeps large,
simulator-specific dependency sets out of the main `fcmaes-rust` workspace.
Most use the same local core:

```toml
fcmaes-core = { path = "../../crates/fcmaes-core" }
```

The two circuit tutorials instead pin the published
`fcmaes-core = "=0.1.3"` so their recorded pairings with the simulator crates
are exact; each manifest includes the local path as a commented development
override.

Run commands from the repository root, for example:

```bash
cd tutorials/rapier-trebuchet
cargo run --release -- --help
```

### What you will learn

Across the series you will learn how to:

- wrap an expensive, nonsmooth Rust simulation as an optimizer objective;
- preserve mixed continuous, integer and categorical decisions;
- choose scalar retry, MODE or MAP-Elites from the result the application
  needs;
- use common random numbers and disjoint validation for noisy objectives;
- assign one owner to the CPU worker budget; and
- export native results for reproducible Python/Matplotlib analysis without
  putting Python in the hot evaluation path.

Prerequisites are a current stable Rust toolchain and a platform linker. Each
tutorial pins its standalone dependency graph in `Cargo.lock`. Figure
regeneration additionally needs Python 3.11 or newer:

```bash
cd tutorials/python
python -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python render_all.py --check
```

The quick commands are functional checks; the recorded publication commands
use 16 or 24 workers and range from seconds to several minutes per seed on the
documented Ryzen 9 9950X.

The examples follow four recurring rules:

1. Choose the required result first: one design, a Pareto front, or a
   behavior-diverse archive. Express genuinely distinct engineering goals as
   MODE objectives instead of hiding every preference in one penalty sum.
2. Report constraints explicitly, including simulation failure, whenever the
   formulation supports them.
3. Fix stochastic seed sets during optimization and validate against disjoint
   seeds afterward.
4. Give one layer ownership of parallelism. Benchmark simulator-internal
   parallelism against fcmaes candidate parallelism instead of enabling both
   pools blindly.

MODE and MAP-Elites answer different questions:

- MODE approximates a Pareto front: which feasible designs are nondominated
  over several competing objectives?
- MAP-Elites builds a repertoire: what is the best feasible design found in
  each user-defined behavior niche?

MAP-Elites is useful only when its descriptors express diversity that a user
would actually choose between. It is not a generic replacement for MODE, and
objective values should not be relabeled as descriptors without a reason.
Two tutorials record the failure modes with measurements. RustPower's first
descriptor pair were decision variables and reached 4% coverage, while emergent
behavior coordinates on the identical budget reached 68%. The ML tutorial shows
that emergent is not sufficient: its first pair were both emergent yet mutually
redundant (rank correlation +0.9997). On the same 271 feasible range-study
candidates and 20×20 grid, the original pair occupied 16 cells while the
replacement occupied 91. Descriptors must be jointly reachable and individually
reproducible. See the
[CVT-MAP-Elites and Diversifier guide](../docs/optimizers.md#cvt-map-elites-and-diversifier)
for the batch APIs, the [result schema](RESULT_SCHEMA.md), and the
[Python plotting package](python/) for reproducible figures.

The recorded timings are reproducibility checks from one machine, not
cross-library performance claims. Every detailed tutorial explains its model,
budgets, output files, limitations and validation strategy.

![A candidate is decoded and evaluated in an isolated native Rust simulator before scalar retry, MODE, or MAP-Elites consumes its outputs](images/native-pipeline.svg)

### Short glossary

- **Nondominated / Pareto point:** no retained feasible design is at least as
  good in every objective and strictly better in one.
- **Descriptor:** a measurable behavior or architecture coordinate used to
  organize a QD repertoire, not another name for the optimized objective.
- **Niche:** one archive cell or CVT region; it stores the best quality found
  for that descriptor region.
- **Coverage:** occupied niches divided by archive capacity.
- **QD score:** the sum of reciprocal positive minimized qualities used by the
  current archive implementation; higher is better only for the same
  descriptor bounds and archive geometry.

### Troubleshooting

- An empty or sparse QD archive usually means feasibility is too rare,
  descriptor bounds are wrong, or invalid simulations were correctly rejected.
  Inspect the invalid and clipping counts before increasing the budget. If the
  descriptors are themselves decision variables, no budget will help: the
  archive can only reach the part of that input space which is also feasible.
  See the RustPower tutorial for the measured before/after.
- If performance worsens as workers increase, check for a nested simulator
  thread pool. NeXosim and Brahe make the two ownership choices explicit.
- If stochastic elites move between niches on validation, increase replication
  count and show both training and holdout maps; do not hide the migration. If
  migration stays high at every grid resolution, the descriptor itself is the
  high-variance part. The ML tutorial measures this directly: its
  threshold-derived precision axis moves a median 1.74 cells between tuning and
  selection while its distributional sharpness axis moves 0.23, so retention
  fails at 20x20 and still fails at 4x4.
- Run `cargo test` and `cargo clippy --all-targets -- -D warnings` inside a
  standalone tutorial directory. The root workspace deliberately excludes
  these simulator-heavy crates.
- Run `python tutorials/python/check_docs.py` from the public repository root
  after moving result or image files.

## 2. NeXosim: production-line digital twin

[NeXosim](https://github.com/asynchronics/nexosim) is a component-oriented
discrete-event simulation framework. The tutorial models

```text
source → machine A → finite buffer → machine B → inspection
                                            ↘ rework or shipping
```

Orders, processing times, failures, repairs and quality outcomes are
stochastic. MODE chooses buffer capacity, machine speeds, maintenance
thresholds, rework routing, dispatch priority and staffing. It minimizes
negative throughput, lead time, work in progress, and energy/staffing cost.
Integer capacity and staffing decisions coexist with continuous policies.

MAP-Elites is implemented as the complementary operating-catalog view.
Achieved throughput and mean WIP form the two-dimensional behavior map;
lead time, cost and failure/overflow penalties determine within-niche quality.
Every retained policy is re-evaluated on disjoint stochastic seeds. Three
100,096-evaluation runs reached 30.583% mean coverage; only 38.908% of elites
remained in the same holdout niche, an important warning against hiding
simulation uncertainty.

This tutorial is primarily about parallel architecture. It compares:

- many serial NeXosim replications evaluated concurrently by fcmaes; and
- serial candidate evaluation where one NeXosim bench owns the thread pool.

Both modes use identical candidates, replications and total worker counts, so
the comparison does not hide nested oversubscription. For this small,
replication-heavy bench, outer fcmaes parallelism was much faster; a larger
model with more simultaneous component work can reverse that outcome.

```bash
cd tutorials/nexosim-production-line
cargo run --release -- \
  --strategy both --workers 16 \
  --evaluations 512 --popsize 32 \
  --replications 4 --horizon 240 --seed 42
```

See the [complete NeXosim tutorial](nexosim-production-line/README.md) for the
event model, common-random-number setup, benchmark interpretation and tests.

![Production-line event flow and the two mutually exclusive owners of parallelism](nexosim-production-line/images/architecture.svg)

## 3. Rapier: discontinuous mechanical design

[Rapier](https://github.com/dimforge/rapier) supplies the rigid-body dynamics
for a two-dimensional trebuchet/robotic thrower. fcmaes chooses arm and sling
lengths, masses, initial and release angles, damping and pivot friction.

The projectile is attached through a rope joint and released when the arm
crosses the selected angle. Contact, joint limits and removal of the joint make
small parameter changes capable of producing qualitatively different
trajectories. Those discontinuities are exactly where finite-difference
gradients become unreliable.

The tutorial presents both:

- BiteOpt parallel retry for a scalar target-error/resource score; and
- MODE for the Pareto trade-off among landing error, available energy and peak
  pivot load.

MAP-Elites is implemented because mechanically different throws can be useful
even when they have similar scalar scores. Trajectory apex and release time
are behavior descriptors; target error, energy and peak load determine quality
inside each niche. Invalid releases are rejected. Across three
200,192-evaluation runs the archive reached 40.583% mean coverage with a
0.022% descriptor-clipping rate.

Rapier's internal parallel feature is disabled. Each physics evaluation is
deterministic and serial, while fcmaes distributes evaluations across workers.
The selected design can be replayed in a self-contained browser animation.

```bash
cd tutorials/rapier-trebuchet
cargo run --release -- \
  --mode all --workers 24 --target 35 \
  --evaluations 20000 --retries 24 \
  --mo-evaluations 200000 --popsize 256 \
  --qd-evaluations 200000 --qd-capacity 400 \
  --qd-chunk-size 256 --seed 42
```

See the [complete Rapier tutorial](rapier-trebuchet/README.md) for the physical
model, serious-run results, Pareto extremes and replay output.

![Scalar, MODE, and MAP-Elites evaluate isolated Rapier worlds through one fcmaes worker pool](rapier-trebuchet/images/architecture.svg)

## 4. ReBop: robust stochastic oscillator design

[ReBop](https://github.com/Net-Mist/rebop) simulates well-mixed stochastic
chemical reaction networks. This tutorial optimizes the logarithms of all 15
kinetic rates in the Vilar oscillator using its compiled macro DSL.

Every candidate is evaluated on the same named stochastic seed set. This
common-random-number design reduces comparison noise without pretending the
objective is deterministic. A disjoint holdout seed set is used only after
optimization, so the report exposes designs that overfit the training paths.

The scalar BiteOpt formulation combines period error, spectral impurity,
amplitude, autocorrelation, molecule count and failure rate. MODE separately
minimizes oscillation error, molecular resource use and stochastic fragility.
This is a native Rust example of robust optimization under intrinsic
simulation noise.

Mean oscillation period and amplitude are the implemented phenotype
descriptors, while spectral purity, molecule cost, fragility and failed-run
fraction determine cell quality. The archive answers “which robust circuit is
best for every reachable period/amplitude combination?” MODE remains present
for the three explicit objective trade-offs. Three 20,096-evaluation QD runs
reached 97.833% mean training coverage; every elite was then checked with
disjoint holdout seeds, and only 37.843% stayed in the same niche.

```bash
cd tutorials/rebop-oscillator
cargo run --release -- \
  --mode both --workers 24 --target-period 20 \
  --replications 4 --validation-replications 8 \
  --evaluations 2000 --retries 24 \
  --mo-evaluations 20000 --popsize 256 --seed 42
```

See the [complete ReBop tutorial](rebop-oscillator/README.md) for seed control,
signal metrics, training/holdout results and reaction-rate bounds.

![Common training paths, period-amplitude repertoire search, and disjoint stochastic validation](rebop-oscillator/images/architecture.svg)

## 5. Brahe: constellation access optimization

[Brahe](https://github.com/duncaneddy/brahe) provides orbit propagation,
ground-station data and access-window calculations. The tutorial chooses the
shared altitude and inclination plus independent RAAN and mean anomaly for four
satellites.

For six ground stations it propagates 24 hours, finds and merges access
windows, rejects short passes, then evaluates worst communication gap, total
contact duration and launch-complexity proxies. An access window appearing or
disappearing changes the objective discontinuously; the maximum-gap operation
adds another nonsmooth layer.

BiteOpt retry handles a scalar design score. Constrained MODE exposes the
gap/contact/cost trade-off while enforcing a minimum number of accepted passes
per station. The tutorial also compares fcmaes-owned outer parallelism with
Brahe-owned access parallelism without allowing both pools to expand together.
Finalists can optionally be checked with Brahe's higher-fidelity numerical
propagator.

The QD pilot passed: altitude and circular RAAN spread distinguish
architecturally meaningful constellations without duplicating MODE objectives.
Candidates missing the per-station pass constraint are rejected. Three
4,096-evaluation runs reached 99.167% mean coverage of a 400-cell archive.
Constrained MODE remains the direct view when the question is the three-way
gap/contact/launch-cost trade-off.

```bash
cd tutorials/brahe-constellation
cargo run --release -- \
  --mode all --workers 24 --parallel outer \
  --evaluations 500 --retries 24 \
  --mo-evaluations 8192 --popsize 256 \
  --qd-evaluations 4096 --qd-capacity 400 \
  --qd-chunk-size 128 --seed 42 \
  --numerical-validation
```

See the [complete Brahe tutorial](brahe-constellation/README.md) for station
selection, access rules, thread-pool ownership and validation limitations.

![Four-satellite propagation, access aggregation, MODE and the accepted architecture archive](brahe-constellation/images/architecture.svg)

## 6. RustPower: robust mixed-integer voltage control

[RustPower](https://github.com/chengts95/rustpower) performs steady-state AC
power-flow analysis. The tutorial uses its embedded IEEE-39 case and pure-Rust
RSparse solver path over six deterministic load, renewable, battery and N−1
line-outage scenarios.

The 20 controls include generator voltage setpoints, integer transformer taps,
categorical capacitor and battery locations, capacitor steps, battery capacity
and renewable curtailment. Constrained MODE minimizes mean line loss, voltage
deviation, lifecycle cost and worst security stress.

Voltage breaches, thermal overload and non-convergence are separate
constraints. A failed Newton solve is not replaced with an arbitrary enormous
objective value: its failed fraction and clipped residual form a calibrated
constraint violation. This preserves objective scales and makes feasibility
auditable.

Constrained MODE is the primary formulation because losses, voltage quality,
investment and security are explicit competing goals with hard feasibility
limits. This tutorial also records the clearest worked example of the descriptor
rule stated above. The first QD attempt used continuous battery MW and capacitor
MVAr as descriptors; after 100,097 evaluations only 16/400 niches were occupied
and battery capacity stayed in a 269.9–274.2 MW band. Both axes were decision
variables, so the archive re-plotted its own search box instead of illuminating
behavior. Replacing them with emergent coordinates measured from the solved
scenarios — weighted-mean bus voltage and the security-utilization spread across
the six scenarios — raised mean coverage from 4.0% to 68.0% over three seeds at
the identical budget, with no descriptor clipping. The corrected archive is a
useful operating-strategy repertoire; asset siting and sizing remain nearly
unique on this network, which the descriptor fix confirms rather than removes.
MODE was retained, not replaced.

```bash
cd tutorials/rustpower-voltage-control
cargo run --release -- \
  --mode optimize --workers 24 \
  --evaluations 8192 --popsize 128 --seed 42
```

See the [complete RustPower tutorial](rustpower-voltage-control/README.md) for
scenario definitions, rating calibration, licensing/build caveats and the
recorded Pareto result.

![Six-scenario voltage-control evaluation, constrained MODE, and the recorded MAP-Elites archive](rustpower-voltage-control/images/architecture.svg)

## 7. Atmospheric dispersion: robust source localization

The [dispersion tutorial](dispersion-source-localization/) is an inverse
problem rather than another forward engineering design. A native Rust
Gaussian-plume model infers the positions, emission rates, and release heights
of two sources together with wind and spread corrections from censored
receptor observations.

Its numerical equations are adapted from the MIT-licensed
[`really-simple-dispersion-wasm`](https://github.com/joshuanunn/really-simple-dispersion-wasm),
but the browser and WebAssembly layers are not used. Concentration is evaluated
only at the receptors required by the objective. The model is derived from the
superseded ISC-3 model and is explicitly educational, not suitable for
regulatory or safety decisions.

The deterministic synthetic dataset has separate sensors and weather for
training and validation. Observation-specific spread perturbations, relative
noise, background concentration, and a detection limit prevent the inverse
problem from being an exact replay of its own forward model.

BiteOpt coordinated advanced retry searches for one robust estimate. MODE
retains the trade-off among mean reconstruction error, tail/detection error,
and total emission. MAP-Elites remains complementary: emission-weighted source
centroid coordinates organize a map of alternative hypotheses, while the
robust reconstruction score chooses the elite inside each cell. Archive
coverage is not interpreted as a confidence region.

Recorded 24-worker runs reduced the mean hidden-source position error from
860.4 m for the initial baseline to 15.2 m for scalar retry and 11.7 m for the
selected MODE point. Three 200,192-evaluation QD runs filled all 400 centroid
niches; the quality gradient across that deliberately complete map, not
coverage by itself, is the useful result.

```bash
cd tutorials/dispersion-source-localization
cargo run --release -- \
  --mode all --workers 24 \
  --evaluations 5000 --retries 24 --depth 6 --max-eval-fac 6 \
  --mo-evaluations 200000 --popsize 256 \
  --qd-evaluations 200000 --qd-capacity 400 \
  --qd-chunk-size 256 --seed 42
```

See the
[complete atmospheric dispersion tutorial](dispersion-source-localization/README.md)
for the decision vector, model attribution, held-out results, serious-run
budgets, QD interpretation, and limitations.

![Receptor observations feed a native inverse model, three complementary optimization formulations, and disjoint holdout validation](dispersion-source-localization/images/architecture.svg)

## 8. Room ventilation: optimization with a purpose-built backend

The [room-ventilation tutorial](cfd-room-ventilation/) demonstrates a case
where the objective needs a small simulation kernel tailored to optimization.
Every candidate changes two wall vents and an internal baffle. A custom native
Rust backend rebuilds that geometry, solves D2Q9 lattice-Boltzmann flow, and
reuses the flow field for three D2Q5 pollutant releases.

The custom implementation keeps solver state isolated and deterministic while
fcmaes evaluates candidates in parallel. It also makes the backend part of the
tutorial model rather than an independently validated engineering package.
The tutorial therefore pairs optimization results with a straight-channel
property check, three-grid sensitivity, three optimizer seeds, and three
held-out pollutant releases.

MODE preserves exposure, receptor concentration, fan-proxy, and final-mass
trade-offs. MAP-Elites organizes feasible alternatives by fresh-air flow and
occupied-zone low-velocity fraction. Across seeds 42–44, MODE obtained
training quality `1.122344 ± 0.004929` and held-out quality
`1.492211 ± 0.004592`; MAP-Elites filled `76.00% ± 1.56%` of 400 niches.
The coarse-grid MODE representative slightly violates the flux constraint, so
the result is explicitly described as resolution sensitivity rather than grid
convergence.

```bash
cd tutorials/cfd-room-ventilation
cargo run --release --bin cfd-room-ventilation -- \
  --mode all --workers 16 \
  --mo-evaluations 2048 --popsize 128 \
  --qd-evaluations 2048 --qd-capacity 100 --qd-chunk-size 128 \
  --seed 42 --output results/smoke
```

See the [complete room-ventilation tutorial](cfd-room-ventilation/README.md)
for robust source sets, equal-budget publication commands, backend scope,
verification evidence, and deterministic figure regeneration.

![Baseline and optimized velocity and pollutant fields from the custom Rust backend](cfd-room-ventilation/images/flow-fields.svg)

## 9. ML hyperparameter optimization: validation-aware search

The [hyperparameter-tuning tutorial](ml-hyperparameter-tuning/) turns a
SmartCore decision tree into a deterministic, bagged probability forest and
optimizes eight mixed hyperparameters. The entire expensive path—bootstrap
sampling, feature subspaces, fitting, probability prediction, and metric
calculation—runs in Rust. Python is used only to render recorded artifacts.

This tutorial focuses on experimental discipline as much as optimizer speed.
It uses fixed stratified tuning folds with common model seeds, a disjoint
selection set for top-candidate re-ranking, and an independently seeded final
test set that is not evaluated until a hash-bound study plan is frozen.
Reported budgets count both candidate evaluations and underlying model fits.

Three formulations answer different questions:

- BiteOpt retry minimizes constrained cross-validated log-loss and returns one
  selected configuration;
- constrained MODE exposes probability quality, ranking quality, serialized
  model size, and a structural prediction-work proxy; and
- MAP-Elites explores alternatives by precision at threshold 0.5 and
  predicted-probability sharpness.

The descriptor pair is itself a recorded lesson, and a different one from the
RustPower case. Both of the originally proposed axes — predicted-positive rate
and the log false-positive/false-negative ratio — are emergent model behavior,
so neither repeats the decision-variable mistake. They still failed, because
with the threshold fixed at 0.5 they are a monotone function of each other
  (rank correlation +0.999715 over 271 feasible recorded candidates) and the
  reachable region is a narrow ribbon inside a two-dimensional grid. The raw
  range-study candidates and deterministic summary are checked in. Two
  emergent descriptors can still be the same axis twice; check that a pair is
  *jointly reachable* before spending a campaign on it.

The checked-in quick run is a functional smoke study, not publication
evidence. It exercises the complete protocol, baselines, budget sweep,
isolated prediction-latency benchmark, finalization guard, and deterministic
figures. The QD formulation was then run at the publication profile over three
independent outer seeds and **rejected**: coverage (49.0%) and configuration
  diversity (196) pass their pre-registered thresholds comfortably, but niche
  retention (6.8%) fails the 50% requirement because precision does not
  reproduce between fixed-fold tuning and the disjoint selection set. The
  saved elites were revalidated after correcting a validation-only aggregation
  defect so both sides describe single-forest behavior; the training archives
  and optimizer budgets are unchanged, and every manifest records that scope.

```bash
cd tutorials/ml-hyperparameter-tuning
cargo run --release -- \
  --preset smoke --mode all --workers 4 --seed 42 \
  --output results/quick
```

See the
[complete ML hyperparameter-optimization tutorial](ml-hyperparameter-tuning/README.md)
for the parameter decoder, probability-forest adapter, objective protocol,
fair-comparison rules, commands, results, and limitations.

![A validation-aware optimization protocol separates tuning, selection, and final reporting](ml-hyperparameter-tuning/images/architecture.svg)

## 10. PGPE and CR-FM-NES: fixed-topology policy search

The [neural-controller tutorial](neural-controller-policy-search/) gives PGPE
and CR-FM-NES a direct-policy-search showcase. A native Rust cart-pole model
randomizes plant parameters, initial conditions, sensor perturbations and
disturbances. The optimized `5 → 16 → 1` neural policy has 118 continuous
weights and receives no gradients.

The experiment contrasts fixed common training scenarios with deterministically
rotating common scenarios, then evaluates every final policy on disjoint
plants. Active CMA-ES and BiteOpt use the same bounds, population size,
candidate and rollout budgets, scenario schedules, and validation protocol as
comparison points. Five roots and exactly 20,480 candidates per run show PGPE
leading under this recorded budget: rotating-scenario PGPE reached a
`0.620 ± 0.376` validation score and `68.3% ± 39.9%` holdout success. The
selected controller then achieved `97.8%` success on a one-time frozen
1,024-scenario final test.

Candidate-level parallelism reduced PGPE wall time from `2.051 s` at one
worker to `0.155 s` at 24 workers for the identical run. The simulator remains
serial and isolated inside each objective call, so fcmaes owns the worker
budget.

```bash
cd tutorials/neural-controller-policy-search
cargo run --release -- \
  --experiment single --algo all \
  --evaluations 2048 --popsize 64 --workers 16 \
  --train-scenarios 2 --validation-scenarios 32 \
  --horizon 200 --seeds 1 --output results/smoke
```

This tutorial deliberately omits MODE and MAP-Elites. Its question is which
optimizer can find one robust policy for a fixed architecture and scalar
control criterion. A Pareto formulation would be justified if control effort,
robustness and settling time were independent deliverables; QD would be
justified if users needed a repertoire indexed by meaningful controller
behaviors. Neither is needed to demonstrate the intended PGPE/CR-FM-NES use
case.

![Validation score and holdout success for fixed and rotating training scenarios](neural-controller-policy-search/images/publication/quality.svg)

See the [complete neural-controller policy-search tutorial](neural-controller-policy-search/README.md)
for the model, objective, fixed-versus-rotating noise protocol, baselines,
frozen test, scaling experiment, raw results and limitations.

## 11. GTOC1 “Save the Earth”: multi-fidelity trajectory optimization

The [GTOC1 tutorial](gtoc1/) combines `pykep-core` and `fcmaes-core` to
reproduce the EVEEEJSJA asteroid-impact trajectory disclosed by JPL. Its
87-variable objective contains encounter dates, endpoint geometry, final mass,
and 24 spherical Sims-Flanagan controls. Seven selected Lambert arcs—two of
them multi-revolution—and unpowered flyby constraints complete the mission.

This example focuses on continuation across model fidelity and optimizer
scale. Its campaign record explains how an earlier 12-segment model was
repaired with coordinated DE–CMA-ES, refined with incumbent-seeded CMA-ES,
doubled to 24 segments, and finally repaired at the highest available
VSOP2013 coefficient precision. The current executable intentionally ships
only the final 24-segment, `1e-9` model and labels this history accordingly.

The chapter now places that fixed-order experiment inside the complete GTOC1
search: propose planet orders, rank them with a cheap Lambert/flyby model, and
promote selected candidates into costly low-thrust validation. It also maps
the `autoresearch-circuit` split-brain architecture onto this task: an AI agent
selects discrete route structures while `fcmaes-core` optimizes their
continuous timings under equal budgets.

The stored solution scores `1,850,730.667522` inside the Rust model, versus
JPL's reported `1,850,000`, with a `3.58e-9` normalized low-thrust mismatch
and a sampled minimum solar distance of `0.671463 AU`. The chapter is explicit
that this is not a new official competition result: pykep-core supplies
VSOP2013 Earth-Moon-barycentre states rather than the required DE405-equivalent
Earth-centre ephemeris. Its threshold-sensitivity table also shows the active
Venus periapsis margin changing sign one truncation level away.

![The GTOC1 split-brain architecture separates AI planet-order proposals, continuous timing optimization, cheap scoring, and costly low-thrust validation](gtoc1/images/split-brain.svg)

```bash
cd tutorials/gtoc1
cargo run --release -- --algorithm inspect
cargo test
```

See the [complete GTOC1 tutorial](gtoc1/README.md) for the competition
background, multi-fidelity and split-brain architectures, mission
transcription, objective construction, staged search, parallel retry commands,
measured wall times, feasibility checks, and scoring limitation.

## 12. sindr: smooth features and manufacturable circuit catalogues

The [circuit-design tutorial](sindr-circuit-design/) puts `sindr` AC analysis
inside three native optimization formulations. It first demonstrates why a
sampled Bode-curve arg-max creates a staircase objective and replaces it with
log-frequency peak and crossing interpolation. Parallel retry then compares
CMA-ES, DE, and BiteOpt on the same requested budget.

The second module uses constrained MODE to retain cutoff/ripple/capacitance
trade-offs for a fourth-order low-pass. The third rounds continuous archive
coordinates to explicit E12 tables and uses common ±5% tolerance draws to
build a reproducible frequency/gain catalogue. A range study freezes the
descriptor box before MAP-Elites runs; out-of-range responses are counted and
rejected rather than hidden in boundary niches.

![Native decode, sindr AC solve, smooth feature extraction, and three fcmaes result types](sindr-circuit-design/images/architecture.svg)

```bash
cd tutorials/sindr-circuit-design
cargo run --release -- --preset smoke --mode all --workers 2 --no-output
```

See the [complete circuit-design tutorial](sindr-circuit-design/README.md) for
the netlists, feature tests, objectives, E12 encoding, recorded evidence,
deterministic visualizations, and the deliberately limited alpha-simulator
scope.

## 13. thevenin: validated transient gate-driver trade-offs

The [gate-driver tutorial](thevenin-gate-driver/) keeps the optimization hot
path pure Rust while adding the transient controls deliberately absent from the
companion `sindr` tutorial. MODE trades 10–90% rise time against gate
overshoot while enforcing peak-current and 2% settling constraints.

The tutorial uses a single checked-in SPICE template for optimization and
validation. Before publication, a boundary-inclusive 7×7 design grid is
replayed independently through libngspice, all five metric-error limits must
pass, and the 50 ps maximum timestep is checked against a 25 ps refinement.

![Pure-Rust optimization plus a publication-only ngspice validation branch](thevenin-gate-driver/images/architecture.svg)

```bash
cd tutorials/thevenin-gate-driver
cargo run --release -- --mode all --preset smoke --workers 2 --no-output
```

See the [complete transient tutorial](thevenin-gate-driver/README.md) for the
circuit model, objectives, measurement interpolation, MODE front, scaling
study, ngspice harness, validation limits, dependency notice, and exact
publication commands.

## 14. Pure-Rust optics: validated multimodal lens design

The [optical-design tutorial](optical-lens-design/) implements the compact
closed-form core of sequential geometric optics—sphere intersections, vector
Snell refraction, Sellmeier dispersion, and paraxial first-order
calculations—in Rust. A disclosed Optiland Cooke prescription and a pupil-grid
study must pass before CMA-ES, DE, BiteOpt, or constrained MODE results are
admitted.

![A serial auditable ray trace feeds scalar retry or constrained MODE](optical-lens-design/images/architecture.svg)

```bash
cd tutorials/optical-lens-design
cargo run --release -- --preset smoke --mode all --workers 4 --no-output
```

See the [complete optical tutorial](optical-lens-design/README.md) for the
prescription, validation limits, spot diagrams, and spot/length/glass front.

## 15. Rapier quadruped: contact-derived gait repertoires

The [quadruped tutorial](rapier-quadruped-gait/) makes MAP-Elites the primary
answer. A 25-variable CPG drives eight ideal motors in a deterministic
9-body Rapier model. Duty factor comes from narrow-phase foot contacts,
mechanical work comes from motor impulse and measured relative speed, and
every elite is replayed on five terrain seeds excluded from training.

![A CPG controls Rapier before MAP-Elites or BiteOpt consumes the measured rollout](rapier-quadruped-gait/images/architecture.svg)

```bash
cd tutorials/rapier-quadruped-gait
cargo run --release -- --preset smoke --mode all --workers 4 --no-output
```

See the [complete gait tutorial](rapier-quadruped-gait/README.md) for the range
study, contact strips, equal-budget scalar baseline, and held-out robustness.

## 16. Diffsol: why gradients are the better default

[Diffsol](https://github.com/martinjrobins/diffsol) is an MIT-licensed Rust
ODE/DAE solver with explicit and implicit integration, event/root detection,
resets, interpolation and dense output, quadrature, checkpointing, and forward
and adjoint sensitivity analysis. Equations can be supplied through Rust
traits/closures or the DiffSL DSL.

We deliberately do **not** add another fcmaes tutorial for Diffsol's standard
smooth parameter-fitting problems. Diffsol can calculate the gradients that
those problems need, so a gradient-based optimizer is the natural first
choice. The Diffsol book demonstrates this directly:

- [predator–prey parameter fitting](https://martinjrobins.github.io/diffsol/primer/population_dynamics_fitting.html)
  obtains gradients with forward sensitivities and uses L-BFGS;
- [spring–mass parameter fitting](https://martinjrobins.github.io/diffsol/primer/spring_mass_fitting.html)
  obtains gradients with an adjoint backward pass and uses L-BFGS;
- [weather prediction with a neural ODE](https://martinjrobins.github.io/diffsol/primer/weather_neural_ode.html)
  uses adjoint gradients and AdamW.

Using fcmaes for those smooth fits would discard valuable derivative
information and usually spend many more simulation evaluations. The right
lesson is not “apply gradient-free optimization to every Rust simulator”; it
is “match the optimizer to the mathematical structure of the problem.”

Diffsol could still support a worthwhile fcmaes application when the outer
decision problem is no longer smooth or purely continuous. Candidate topics
include:

- a fed-batch reactor with discrete feed-policy stages;
- pharmacological dosing with integer dose counts and event-triggered timing;
- hybrid control with resets and mode switches;
- energy-system operation under discrete commitment decisions;
- robust controller tuning that minimizes a worst case over uncertain plant
  models.

Events alone do not automatically rule out sensitivities. An fcmaes tutorial
would be justified when resets, discrete policies, solver failures, robust
maxima or other discontinuities make the end-to-end gradient unavailable or
misleading. Until such a formulation adds something distinct from the
repository's existing dynamic-control examples, the Diffsol book's
sensitivity-aware gradient workflows are the better tutorial.

MAP-Elites would change that conclusion only when diversity is itself the
deliverable—for example, an archive indexed by oscillation period and energy
use or by qualitatively different event-triggered policies. Smooth
parameter estimation with one best fit is still a gradient-based problem, even
when the simulator happens to be written in Rust.

![Diffsol optimizer selection based on sensitivity validity and the result the user needs](images/diffsol-selection.svg)
