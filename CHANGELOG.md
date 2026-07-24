# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/) during its pre-1.0 development
phase, with breaking changes called out explicitly.

## [Unreleased]

## [0.1.1] - 2026-07-24

Initial dual-registry release candidate.

### Added

- Pure-Rust implementations of Differential Evolution, active CMA-ES,
  CR-FM-NES, PGPE, Dual Annealing and BiteOpt.
- MODE, weighted multi-objective retry, CVT MAP-Elites and the Diversifier.
- Independent and coordinated retry with reproducible per-worker random
  streams.
- Ask/tell and parallel batch-evaluation interfaces.
- Native GTOP and application examples, reproducible optimizer comparisons,
  and five Rust simulator-optimization tutorials.
- `fcmaes-core` crates.io package metadata and docs.rs documentation.
- `fcmaes-rust` Python distribution with the `fcmaes_rust` facade.

### Packaging

- Rust 1.88 is the tested minimum supported Rust version.
- CPython 3.11 through 3.13 wheels are prepared for Linux x86-64, Windows
  x86-64, macOS x86-64 and macOS ARM64.
- Registry workflows use GitHub environments and OpenID Connect trusted
  publishing after the required registry-side setup.

[Unreleased]: https://github.com/dietmarwo/fcmaes-rust/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/dietmarwo/fcmaes-rust/releases/tag/v0.1.1
