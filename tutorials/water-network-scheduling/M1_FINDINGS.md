# `epanet-rs` 0.2.3 reconnaissance

Date: 2026-07-30. This gate was completed against the exact source selected by
`Cargo.lock`, before the tutorial driver was implemented.

| Required surface | Result | Confirming API or decision |
|---|---|---|
| Stepwise extended-period simulation | yes | `Simulation::initialize_hydraulics`, `run_hydraulics`, `next_hydraulic_timestep`, and public `time` |
| Pump status and relative speed | yes | independent candidate state in `SolverState::statuses` and `settings` |
| PRV setpoint | yes | `ValveType::PRV`; settings are pressure in internal feet |
| Heads, flows and delivered demand | yes | public `SolverState` vectors |
| Velocity | derived | pipe flow divided by the pipe's circular area |
| Tank level | derived | tank head minus tank elevation |
| Typed hydraulic failure | yes | `SolverError`; `MaxIterations` does not carry a usable trial-limited result |
| Pressure-driven demand | yes | `DemandModel::PDA`, minimum/required pressure and exponent |
| EPANET ENERGY report | **no** | ENERGY is listed as unsupported; the tutorial integrates `ρgQH/η` explicitly |
| RULES | **no** | the generated network has no CONTROLS or RULES; one tested Rust function owns precedence |
| LEAKAGE | **no** | the MO formulation uses an “excess-pressure proxy”, not invented leakage physics |
| Water quality | **no** | outside this tutorial's scope |
| `.inp` input/output | yes | `Network::from_file` and `io::inp::write_inp` |
| Candidate thread safety | yes | compile-time `Send + Sync` assertions for `Network` and `Simulation` |
| Internal parallel EPS | conditional | only legal without tanks and pressure controls; measured on a separate legal variant |

## Unit boundary

The high-level result collector converts back to user units, but public
stepwise `SolverState` remains in EPANET's internal feet and cubic-feet-per-
second representation. `driver.rs` therefore has one explicit conversion
boundary:

```text
head/pressure: ft × 0.3048
flow:          ft³/s × 0.028316846592
PRV pressure:  m ÷ 0.3048 before assigning state.settings
```

Pump relative speed is dimensionless.

## Consequences

- The pure-Rust backend is sufficient; no C binding or native build dependency
  is required.
- A solver failure is a typed finite constraint. Failed steps never produce
  fabricated pressure or energy values.
- Power arithmetic is checked against a checked-in offline oracle. Trace replay
  separately verifies stored-power accumulation; it is not described as an
  independent power calculation.
- Numerical validation in this tutorial is internal. It does not claim
  upstream EPA EPANET equivalence.

## Lockfile licence result

`epanet-rs` itself is dual MIT/Apache-2.0. Its enabled `simplelog` feature pulls
in unmodified `paris = 1.5.15` under MPL-2.0. The tutorial records a
crate-specific `cargo-deny` exception and a
[`DEPENDENCY_NOTICE.md`](DEPENDENCY_NOTICE.md); MPL-2.0 is not generally
allowed.
