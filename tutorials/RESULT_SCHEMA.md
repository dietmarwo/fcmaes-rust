# Tutorial result schema

The simulator tutorials write a small, versioned set of machine-readable
artifacts. Native Rust performs simulation and optimization; Python reads these
artifacts to create documentation figures.

Most individual optimizer runs use the manifest schema below. A tutorial may
also publish a documented aggregate evidence bundle when one figure combines
multiple runs or validation studies. The room-ventilation tutorial is the
current example: its per-seed Pareto/archive and convergence CSVs are combined
with held-out evaluations, CFD fields, and a resolution study by a
byte-for-byte checked `plot_results.py`. Aggregate bundles retain the same
minimization, feasibility, full-precision CSV, seed, budget, and artifact-link
conventions even though no single `run.json` can describe the combined figure.

Schema version `1` uses one `run.json` manifest per optimization run:

```json
{
  "schema_version": 1,
  "tutorial": "rapier-trebuchet",
  "formulation": "mo",
  "command": "cargo run --release -- --mode multi ...",
  "seed": 42,
  "workers": 24,
  "requested_evaluations": 20000,
  "actual_evaluations": 20096,
  "elapsed_seconds": 12.34,
  "objectives": [
    {"column": "objective_target_error", "label": "Target error", "unit": "m"}
  ],
  "descriptors": [],
  "artifacts": {
    "pareto": "pareto.csv",
    "convergence": "convergence.csv"
  }
}
```

All objective values are minimized. Constraints are feasible at values less
than or equal to zero. A manifest may add simulator-specific metadata, but
must not change those conventions.

## Pareto data

`pareto.csv` contains one row per retained point:

```text
point_id,feasible,selected,objective_*,constraint_*,decision_*
```

`selected` identifies tutorial representatives. More than one representative
may be selected when the text discusses extremes and a compromise.

## Quality-diversity data

`qd_archive.csv` contains one row per occupied niche:

```text
niche_id,grid_x,grid_y,quality_train,quality_validation,
descriptor_*_train,descriptor_*_validation,visit_count,decision_*,constraint_*
```

For a regular two-dimensional archive, `run.json` includes
`qd.grid_shape = [columns, rows]`. CVT archives instead export their center
coordinates. Invalid or infeasible simulations are represented by non-finite
fitness while optimizing and never become archive elites.

## Convergence data

The first columns are:

```text
evaluations,elapsed_seconds
```

MODE runs report `best_quality` after those axes. It is a documented scalar
summary used for convergence visualization, not a replacement for the Pareto
front. Constrained MODE may add `feasible_population` and
`pareto_population`; other metric columns must be named in
`run.json.convergence_metrics`. MAP-Elites runs report `coverage`, `qd_score`,
`best_quality` and `invalid_fraction`. Evaluation count, rather than generation
number, is the common comparison axis.

## Reproducibility

Manifests record the exact command, optimizer seed, simulation training and
validation seeds, effective workers, requested and actual budgets, elapsed
wall time, descriptor bounds and relevant optimizer parameters. Values are
written at full `f64` precision; Markdown tables and plot labels perform
display rounding.
