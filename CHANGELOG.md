# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/) during its pre-1.0 development
phase, with breaking changes called out explicitly.

## [Unreleased]

### Added

- Added a dependency-isolated GTOP experiment that runs external serial
  `cmaes` through one serial restart lane and through physical-core restart
  lanes coordinated by generic `fcmaes_core::retry`. The primary 100-pair,
  seven-GTOP protocol gives both arms the same four-second wall allowance and
  reports best-objective mean/sdev plus paired retry wins; measured wall
  mean/sdev, CPU time, active cores, starts, and evaluations audit fairness and
  expose the deliberately higher parallel work. Deterministic seed streams,
  immediate restart after protective termination, alternating arm order,
  resume validation, and generated reports complete the protocol. A separate
  five-pair fixed-work pilot retains 11.18×–12.23× scheduling evidence at 16
  physical cores but is explicitly secondary to equal-wall solution quality.
  The completed campaign improves mean objective on all seven problems and
  wins 86–96 of 100 pairs; sdev improves on five problems but increases on
  SAGAS and Tandem. Cassini1 success rises from 38/100 to 100/100, while the
  other six cases honestly remain at zero target successes.
- Added a dependency-isolated active-CMA-ES implementation diagnostic against
  `cmaes` 0.2.2. Its shared reflected objective, explicit three-arm topology,
  physical-core default, paired deadline ordering, direct cost calibration,
  termination accounting, resume validation, raw CSV/JSON artifacts, and
  three-seed smoke bundle establish the protocol without treating smoke data
  as publication evidence or the easy controls as recommended CMA-ES
  applications.
- Completed its 20-pair, 7,920-row publication campaign on a 16-core Ryzen 9
  9950X, recording about 24.96 billion objective calls, 5.327 active wall
  hours, exhaustive generated tables, and deterministic quality/throughput
  figures. The scoped result reports mixed quality, a `cmaes` serial advantage
  on cheap/high-dimensional cases, a fcmaes-core aggregate-throughput
  advantage in equal multistart, and the observed 61.5% external protective-
  stop rate.

### Fixed

- Rejected non-finite and physically impossible negative time-to-50-AU values
  in the Rust SAGAS port. The equal-wall campaign exposed one false negative
  objective; a focused regression guard and complete same-seed SAGAS rerun
  replace it with 200 validated rows and zero false target successes.

### Changed

- Promoted equal-wall solution quality to an explicit retry design principle
  in the main README and retry guide. The external-CMA GTOP benchmark now
  carries complete quality, wall/core fairness, and starts/evaluations tables,
  and documents how the generic retry closure can coordinate compatible
  single-threaded third-party or FFI optimizers.
- Promoted population ask/tell as the complementary fixed-wall-time strategy
  for expensive objectives. The documentation now enumerates each core
  optimizer's batch boundary, records sequential Dual Annealing as the one
  non-population exception, distinguishes external `cmaes` 0.2.2's internal
  `run_parallel()` from public ask/tell, and warns against nested-worker
  oversubscription.
- Audited the Foundations evidence narrative against its artifacts: clarified
  initial/requested/actual budgets, the scalar DE batch overshoot, exact timing
  scope, the ten-seed Lennard-Jones scale, and L-BFGS's early-stop traversal
  advantage.
- Corrected split-brain attribution in both agent tutorials. GTOC1 now exposes
  the assisted run's 4/95/1 ranked-choice split, incremental versus prior-chain
  cost, 7–9-route concentration, JPL calibration gap, and reconstructible
  provider prompts. Oscillator search now records that the unlabeled
  repressilator was offered 17 times, attributes results to the complete v4
  menu–model policy, and names the missing menu-matched controls.
- Clarified that every Foundations MODE example and publication result uses
  the default NSGA-II-style population update, while the supported DE update
  is not part of that evidence; froze the setting and intentional serial
  evaluation policy in the benchmark contract and run manifest, and documented
  ordered `parallel_batch` evaluation for costly objectives.
- Hardened the oscillator topology-search live-agent protocol with forced
  structured output, an 8,000-token MiniMax cap, and local unauthenticated
  llama.cpp support. Protocol v4 now gives OpenAI-compatible models a balanced
  deterministic menu of unseen elite mutations, underrepresented structures,
  and random immigrants; deduplicates rejection feedback; and rejects older
  agent archives at resume time.
- Replaced the oscillator tutorial's preliminary 20-candidate serial evidence
  with a matched seed-42 comparison of 200 random, evolutionary, and Gemma 4
  31B Q8 proposals at 16 × 12,000 inner evaluations. The v4 agent completed
  without proposal failures, achieved the best and median scores, and exactly
  rediscovered the held-out repressilator at proposal 188.
- Reframed the GTOC1 route-search tutorial around impulsive-MGA portfolio
  discovery and replaced its interim cold-Gemma snapshot with complete
  100-route seed-42 random, evolutionary, and Gemma 4 evidence. Cold Gemma
  concentrates 90 routes at the 14-encounter limit and trails evolutionary
  search on the declared best-20 sum while using more than twice its worker
  time.
- Added the separately versioned `gemma4-assisted-v1` follow-up. It verifies
  and consumes the completed baseline archives, presents length-stratified
  unseen candidate menus, and uses ranked fallbacks; the completed run raises
  the best-20 MGA sum from 19.676 M to 26.964 M and cuts wall time from 11.66
  to 4.70 hours. Documentation explicitly treats it as prior-informed protocol
  evidence rather than an independent model-comparison arm.

## [0.1.4] - 2026-07-31

### Added

- Added a standalone Foundations guide with seven compact, size-tested
  lessons; eight classic scalar functions; ZDT1–4/ZDT6 and DTLZ1–7 with
  deterministic analytic fronts; a strict local CEC transform loader; and
  explicit WFG/BBOB evidence-gate skip artifacts.
- Added reusable `fcmaes-core::indicators` APIs for hypervolume, IGD/IGD+,
  GD/GD+, additive epsilon, spacing, and spread, including exact-vs-sampled
  provenance, explicit reference points, sampling uncertainty, and front
  cleanup accounting.
- Added schedule-independent `RetryContext::run_seed`, exact ragged
  `GridLayout` reporting for MAP-Elites, strict/excluding hypervolume
  reference-box policies with exclusion counts, and public non-dominated-sort
  and crowding-distance utilities.
- Added a dependency-isolated, 20-seed optimizer-boundary experiment with
  exact wall-resource accounting, held-out stochastic validation, equal-wall
  DE/EGO traces, and raw per-seed artifacts. Its public guide defines retry's
  closure as the supported adapter point for external local, Bayesian,
  gradient, and structured solvers.
- Added schema-v2 Foundations conformance evidence with equal-budget random
  controls, initial-population baselines, stored front decisions,
  deterministic same-evaluator rechecks, recorded normalization extrema, MODE
  convergence checkpoints, and four deterministic explanatory/result SVGs.
- Extended Foundations with a 33–294-dimensional Lennard-Jones cluster study:
  analytic gradients, finite C1 overlap handling, free and fixed-frame
  encodings, independently audited source-cited putative targets without
  redistributed coordinates, schedule-independent retry arms, external
  `argmin` L-BFGS and basin-hopping references, tiered target attainment,
  explicit pair/population accounting, a CR-FM-NES configuration sensitivity
  check, ten-seed scaling evidence, and a pre-registered descriptor-pilot
  rejection with an explicit QD skip.

- Added the reviewed GTOC1 “Save the Earth” tutorial with the real
  `EVEEEJSJA` sequence, staged global optimization, a finite-thrust ZOH
  transcription, accelerated Taylor propagation, independent DOP853
  validation, and a 5–8-segment-per-leg whole-tour formulation.
- Added a work-in-progress split-brain GTOC1 route-search tutorial with a
  provider-independent agent subprocess, deterministic grammar and duration
  decoding, equal-budget random and evolutionary controls, crash-safe replay
  artifacts, feasibility-first Lambert L0 evaluation, Sims–Flanagan L1
  promotion, and optional Taylor/DOP853 L2 validation.
- Added the reviewed MiniMax-M3 seed-42 L0 audit bundles: all three arms
  complete 40 candidates after repairing evolutionary bootstrap and
  exploration, random supplies 15 L0-admissible routes, and evolutionary
  supplies 24.
- Added a predeclared random-arm L1 follow-up promoting L0 ranks 1, 8, and 15;
  none passes closure, the leader yields a finite surrogate-gap diagnostic,
  and both lower-ranked controls retain typed propagation failures.
- Added GitHub-only tutorials for `sindr` AC circuit design, `thevenin`
  transient gate-driver design, pure-Rust optical lens design, and Rapier
  quadruped gait repertoires.
- Added GitHub-only tutorials for hardware-quantized phased-array codebooks,
  bilevel energy-hub sizing with an embedded dispatch LP, robust random-key
  field-service routing, and `epanet-rs` water-network pump scheduling.
- Added a GitHub-only truss topology and catalogue-section sizing tutorial
  with exact-k decoding, a validated native FEM, typed mechanism/conditioning
  failures, equal-budget scalar retry, constrained MODE, removal reanalysis,
  and a pre-registered descriptor-gate rejection.
- Added a GitHub-only weighted network-coverage tutorial with deterministic
  synthetic fixtures, exact native group-pair scoring, separate matching and
  primal-dual certificates, exact tiny ILPs, a pre-optimization throughput
  gate, integer-aware DE/MODE, and an honestly dominant marginal-greedy
  baseline.
- Added a GitHub-only split-brain oscillator topology-search tutorial with a
  signed three-gene grammar, runtime ReBop Hill propensities, variable
  10–18-dimensional fixed-budget BiteOpt tuning, held-out motif rediscovery,
  equal-budget random/evolutionary controls, an optional live-agent boundary,
  and a fresh descriptor-pilot rejection with explicit QD skip.
- Added shared result-schema, descriptor-pilot, and robustness/holdout
  protocols for application tutorials, with deterministic custom renderers
  and compact checked-in publication evidence.
- Added dependency-policy files for application workspaces, including a
  crate-specific MPL-2.0 exception and notice for the unmodified
  `pykep-core` dependency used by GTOC1 route search.
- Added a rendered-mdBook link and anchor validator to the GitHub Pages build.

### Changed

- Bumped the unreleased Rust workspace version to 0.1.4 for the additive core
  evidence and reproducibility APIs. `Archive::grid_shape()` remains as a
  compatibility convenience, while exact rendering uses
  `Archive::grid_layout()`; refreshed standalone local-path tutorial lockfiles
  while leaving deliberate crates.io `=0.1.3` reproduction pins unchanged.
- Marked `RetryContext` as non-exhaustive while adding `run_seed`. Normal retry
  closures are unaffected, but downstream direct struct literals must be
  removed; retry owns construction of this input context.
- Kept Nelder–Mead, Bayesian optimization, gradient solvers, and their
  dependencies outside `fcmaes-core` after the corrected boundary experiment
  found no general DE→NM advantage and only a narrow low-budget BO regime.

- Expanded the tutorial CI matrix, rendered guide, main index, and mdBook
  navigation to cover all twenty-two application tutorials.
- Made result-driven figure checks use the CI-pinned plotting environment and
  allow tutorials with domain-specific artifacts to opt into custom
  renderers.
- Upgraded GTOC1 to `pykep-core` 0.1.4 and separated the fixed-sequence
  continuous-thrust chapter from the variable-order campaign protocol.
- Extended the route-search evidence from its mock transport fixture to a
  feasibility-first completed live seed-42 L0 audit and targeted random-arm
  L1 controls, while retaining independent seeds, matched three-arm L1
  promotions, and L2 validation as open requirements.
- Hardened the oscillator-topology descriptor pilot with an explicit
  two-of-three-arm failure, observed ranges and bound fractions, 6×6 coarse
  retention, and an additive eight-replication sensitivity measurement.
- Added deterministic worker-budgeted parallel BiteOpt retry inside each
  oscillator topology while preserving sequential agent/evolutionary feedback,
  fixed seeds, and explicit per-retry versus total evaluation accounting;
  physical-core retry and worker counts are the default.

### Fixed

- Replaced degenerate partial-front Foundations hypervolume with one
  union-front reference shared by every arm and checkpoint of a problem;
  retained fixed `[1.1; m]` hypervolume as a nullable secondary field without
  filtering points, and labeled same-evaluator repeats as deterministic
  rechecks rather than independent validation.
- Marked the one-seed Foundations `publication` preset as conformance evidence,
  completed WFG/BBOB skip manifests with command/protocol fields, and
  documented that the initial scalar baseline is nested in the random stream.

- Prevented an infeasible high raw-score Earth–Saturn route from being
  described or plotted as a campaign leader; its launch-excess violation is
  now documented and result figures follow the Rust archive ordering.
- Diagnosed the seed-42 evolutionary control as a protocol failure: one-edit
  mutations cannot clear the edit-distance-3 gate while the archive remains
  below its six-route bootstrap threshold.
- Repaired evolutionary route search with independent bootstrap seeds and
  random exploration immigrants, preserving elite mutations for exploitation;
  numerical L1 propagation failures now become archived promotion outcomes
  instead of aborting the campaign.
- Repaired oscillator evolutionary search to sample eight ranked elites plus
  deterministic 20% random immigrants instead of exhausting one incumbent's
  one-edit neighborhood, and added schema-v2 resume validation for the seed,
  optimizer protocol, worker count, proposal policy, and candidate budgets.
- Added an oscillator `--mode report` path that validates completed matched
  arms, preserves their manifests and archives, retains agent usage accounting,
  and writes only the comparison and descriptor-gate reports without launching
  optimization, an agent, or QD.
- Hardened the oscillator agent boundary with a no-request configuration
  preflight, typed transport versus response errors, format repair only for
  malformed responses, a persistent three-failure circuit breaker with the
  final diagnostic, and cumulative attempts/failures/tokens across resume.
- Replaced the oscillator tutorial's visibility-patch claim with an accurate
  reduced-ReBop compatibility notice and a dual-source upstream replay test;
  its skipped-QD manifest, comparison direction, rediscovery cells, and exact
  random-sampling prior now match the shared documentation contracts.
- Fixed mdBook links to directory `README.md` chapters, published linked
  supporting chapters, and replaced non-browsable artifact-directory links
  with their GitHub tree targets.
- Corrected moved docs.rs, ReBop, Optiland, and release-history links.
- Reconciled all 22 tutorial inventories, corrected stale architecture and AI
  context counts, removed drift-prone tutorial ordinals, aligned the
  phased-array and energy-hub QD summaries with their accepted native-grid
  pilots, and made release commands derive the workspace version. Added a CI
  consistency check for tutorial inventories, repeated counts, registry pins,
  MODE-using tutorial counts, and workspace lockfile versions.

## [0.1.3] - 2026-07-25

### Added

- Added complete rustdoc for every public `fcmaes-core` item, runnable examples
  for all major solver families, and primary literature references for DE,
  active CMA-ES, CR-FM-NES, PGPE, Dual Annealing, MODE and MAP-Elites.
- Added complete runtime docstrings for every public PyO3 function, class,
  method, and property, with an installed-extension test preventing regression.
- Added an mdBook documentation site assembled from the canonical guides and
  tutorials, plus a GitHub Pages deployment workflow.
- Added a combinatorial encoding cookbook and tested, dependency-free example
  helpers for bounded and logarithmic integers, categories, Booleans,
  random-key permutations, exact-cardinality subsets, unique-selection repair,
  route partitions, and ordered breakpoints.
- Added a GitHub-only fixed-topology neural-controller policy-search tutorial
  with a native stochastic cart-pole model, PGPE and CR-FM-NES showcases,
  equal-protocol active CMA-ES and BiteOpt comparisons, common scenario sets,
  disjoint validation, a frozen final test, parallel-scaling measurements,
  tests, raw result data, and deterministic figures.
- Added a GitHub-only SmartCore hyperparameter-optimization tutorial with a
  native probability-forest adapter, mixed-variable decoding, fixed-fold
  tuning, disjoint selection and frozen final evaluation, BiteOpt retry,
  constrained MODE, a MAP-Elites pilot, fair baselines, cost accounting,
  latency/scaling measurements, tests, and deterministic figures.
- Added a GitHub-only room-ventilation optimization tutorial with a
  purpose-built D2Q9/D2Q5 Rust backend, worst-case training releases, held-out
  validation, BiteOpt retry, MODE, MAP-Elites, a straight-channel property
  check, three-grid sensitivity, three optimizer seeds, and deterministic
  field/result figures.
- Added a GitHub-only atmospheric dispersion source-localization tutorial with
  a native ISC-3-derived educational model, BiteOpt coordinated advanced retry,
  MODE, MAP-Elites, disjoint holdout validation, reproducible result artifacts,
  and deterministic figures.
- Added the GitHub-only `buckingham-pi` example with native dimension-matrix
  preprocessing, nullspace calculation, repeating-variable enumeration,
  π-group construction, deterministic train/holdout regression, independent
  BiteOpt retry, constrained MODE, tests, and a dedicated guide.

### Changed

- Raised `fcmaes-core` rustdoc coverage from 37.34% to 100% and made missing
  public API documentation a compile-time and CI error.
- Reframed module documentation around algorithms and primary publications
  instead of historical implementation provenance.
- Pointed the crates.io and PyPI package documentation paths at the generated
  API reference and rendered guide, respectively.
- Extended tutorial artifact validation to check the neural policy-search
  comparison, scaling, convergence, and replay figures byte for byte.
- Extended tutorial artifact validation to check the room-ventilation
  multi-seed, field, and resolution figures byte for byte.
- Separated Pareto titles and legends in the shared tutorial renderer and
  regenerated the byte-for-byte checked SVG figures.

## [0.1.2] - 2026-07-24

First synchronized crates.io and PyPI release.

### Released

- Published `fcmaes-core` 0.1.2 on crates.io with API documentation on
  docs.rs.
- Published `fcmaes-rust` 0.1.2 on PyPI with a source distribution, CPython
  3.11 through 3.13 wheels for Linux x86-64, Windows x86-64, macOS x86-64 and
  macOS ARM64, and a PyPy 3.11 wheel for Linux x86-64.
- Kept `fcmaes-gtop`, the example crate and the five simulator tutorials as
  GitHub-only source; they are not separate registry packages.

The synchronized release contains the pure-Rust implementations of
Differential Evolution, active CMA-ES, CR-FM-NES, PGPE, Dual Annealing,
BiteOpt, MODE, weighted multi-objective retry, CVT MAP-Elites and the
Diversifier. It also contains independent and coordinated retry, reproducible
per-worker random streams, ask/tell interfaces, parallel batch evaluation and
the optional `fcmaes_rust` Python facade.

### Packaging

- Added a tag-gated crates.io workflow for publishing `fcmaes-core` with
  short-lived GitHub OIDC credentials after package validation.
- Released both public packages from the `v0.1.2` source revision through
  registry trusted publishing.
- Rust 1.88 is the tested minimum supported Rust version.

## [0.1.1] - 2026-07-24

Initial crates.io bootstrap release.

### Added

- Pure-Rust implementations of Differential Evolution, active CMA-ES,
  CR-FM-NES, PGPE, Dual Annealing and BiteOpt.
- MODE, weighted multi-objective retry, CVT MAP-Elites and the Diversifier.
- Independent and coordinated retry with reproducible per-worker random
  streams.
- Ask/tell and parallel batch-evaluation interfaces.
- `fcmaes-core` crates.io package metadata and docs.rs documentation.

### Packaging

- Published `fcmaes-core` 0.1.1 manually to establish crates.io ownership
  before configuring trusted publishing.
- There was no corresponding `fcmaes-rust` 0.1.1 release on PyPI.

[Unreleased]: https://github.com/dietmarwo/fcmaes-rust/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/dietmarwo/fcmaes-rust/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/dietmarwo/fcmaes-rust/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/dietmarwo/fcmaes-rust/releases/tag/v0.1.2
[0.1.1]: https://crates.io/crates/fcmaes-core/0.1.1
