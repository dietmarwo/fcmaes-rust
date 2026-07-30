# Tutorial result schema

The application tutorials write a small, versioned set of machine-readable
artifacts. Native Rust performs simulation and optimization; Python reads these
artifacts to create documentation figures. For model-fitting tutorials, native
Rust likewise owns fitting, prediction, and metric evaluation.

Most individual optimizer runs use the manifest schema below. A tutorial may
also publish a documented aggregate evidence bundle when one figure combines
multiple runs or validation studies. The room-ventilation tutorial combines
per-seed Pareto/archive and convergence CSVs with held-out evaluations, CFD
fields, and a resolution study. The neural policy-search tutorial combines
fixed/rotating-scenario comparisons, parallel-scaling repeats, baselines,
monitor histories, a selected policy, and a frozen final test. Both use a
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

An independently executed comparison arm that cannot produce a feasible
candidate is not omitted. If it ran, it writes `status: "completed"`, its
actual evaluations, and a zero feasible-result count. `status: "skipped"`
with a machine-readable `reason`, `actual_evaluations: null`, and an empty
`artifacts` object is reserved for an arm that was not executed. A frozen
study plan may list such arms under `excluded_arms`; excluded arms are never
evaluated on final-test data.

When one objective call expands into several simulations or model fits, the
manifest should record both logical candidate calls and physical work. The HPO
tutorial, for example, separates `tuning_model_fits` from
`selection_model_fits` and also reports their total as `model_fits`.

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
Validation-aware tutorials may use `selection_feasible` and `retained_niche`
to distinguish failure of a held-out constraint from movement to a different
behavior niche.

When a QD archive is a machine-consumed control catalogue, publish decoded
integer controls alongside normalized decisions. The phased-array codebook,
for example, adds semicolon-separated `phase_codes` and `attenuator_codes` to
`codebook.csv`; replay tests reconstruct the physical state from those codes.
The image is a visualization, while the decoded table is the product.

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

## Agent-guided route-search data

The GTOC1 route-search tutorial uses one schema-v1 manifest per agent, random,
or evolutionary arm. Its full `configuration` object is the replayable
invocation contract and includes grammar, optimizer, promotion, fidelity,
transport, and result-path settings. `budget` separates proposal attempts,
accepted L0 candidates, cache hits, requested/actual L0 and L1 evaluations,
allocated worker-seconds, promotions, and token usage.

Its additional artifacts are:

```text
archive.jsonl       checksummed append-only candidate and promotion revisions
archive.json        atomic latest-state snapshot
archive.csv         one row per logical candidate
proposal_log.jsonl  invalid, duplicate, repaired, diversity, and transport events
agent_log.jsonl     redacted requests/responses usable by replay transport
promotions.csv      paired L0/L1 scores, surrogate gap, threshold and failure
convergence.csv     accepted candidates, best L0/L1, niches and worker-seconds
```

L0 is a proxy and never establishes feasibility. An executed L1 arm with no
threshold-passing refinement remains `status: "completed"` and records
`l1_threshold_passed: 0`; budget exhaustion is not physical infeasibility.
Allocated worker-seconds are wall time multiplied by resolved workers, not
measured CPU time.

## Reproducibility

Manifests record the exact command, optimizer seed, simulation or model
training and validation seeds, effective workers, requested and actual
budgets, elapsed wall time, descriptor bounds and relevant optimizer
parameters. Values are written at full `f64` precision; Markdown tables and
plot labels perform display rounding.
