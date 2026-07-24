# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/) during its pre-1.0 development
phase, with breaking changes called out explicitly.

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/dietmarwo/fcmaes-rust/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/dietmarwo/fcmaes-rust/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/dietmarwo/fcmaes-rust/releases/tag/v0.1.1
