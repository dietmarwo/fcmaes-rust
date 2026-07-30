# Provenance

Everything below is tutorial-owned synthetic data. No utility network, customer
record or tariff dataset is redistributed.

| Input class | Origin | Frozen representation |
|---|---|---|
| topology and coordinates | hand-designed single-zone teaching network | `network/synthetic-zone.inp` |
| elevations and base demands | synthetic values chosen to admit normal and stressed operation | `[JUNCTIONS]`, `[RESERVOIRS]`, `[TANKS]` |
| pipe dimensions and roughness | synthetic metric Darcy–Weisbach inputs | `[PIPES]` |
| pump head curves | synthetic three-point curves | `[CURVES]` C1 and C2 |
| pump efficiency curves | smooth tutorial functions, not EPANET ENERGY data | `src/energy.rs` |
| power oracle | four pump/flow/head values calculated independently outside the Rust implementation and rounded to six decimals | `scenarios/energy-power-oracle.csv` |
| daily demand pattern | synthetic 24-hour multiplier profile | `[PATTERNS]` DAILY |
| tariff | synthetic off-peak/shoulder/peak bands | `src/scenarios.rs::tariff` |
| scenarios | named deterministic perturbations | `src/scenarios.rs` |
| generator identity | fixed template seed `20260730` | publication `generated/generator.json` |

The one-pipe validation input and tank-free benchmark input are also
tutorial-owned:

- `network/analytic-pipe.inp` isolates a laminar Darcy–Weisbach case;
- `network/benchmark-zone.inp` removes tanks and pressure control so both
  parallelism arrangements are legal.

`epanet-rs = 0.2.3` is dual MIT/Apache-2.0. `fcmaes-core = 0.1.3` is MIT.
The exact dependency graph is frozen by `Cargo.lock`. Its transitive logging
stack includes unmodified `paris = 1.5.15` under MPL-2.0; see
[`DEPENDENCY_NOTICE.md`](DEPENDENCY_NOTICE.md) and the named `cargo-deny`
exception.
