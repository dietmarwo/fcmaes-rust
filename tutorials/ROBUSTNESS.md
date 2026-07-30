# Robust optimization and holdout protocol

Robust optimization needs named, reproducible scenarios. An anonymous seed
range makes it impossible to tell what was optimized, what was independently
validated, or whether one arm received easier cases.

## Freeze training before search

Define every perturbation family, magnitude, aggregation rule, and seed table
before optimizer comparison. Check fixed stochastic draws into the repository
when practical. Every arm must evaluate the same candidate against the same
training cases.

Record both logical candidate calls and expanded physical work:

```text
physical evaluations = candidate calls × scenarios per candidate
```

For worst-case design, the objective or constraint must use the maximum across
the complete scenario set. Do not randomly subsample adversarial cases per
candidate unless stochastic objective noise is itself part of the declared
protocol.

## Make holdout structurally different

Changing only a random seed tests replication noise, not model robustness.
When the application permits it, hold out a different:

- perturbation magnitude or family;
- failure topology;
- operating condition or terrain;
- data split or distribution;
- simulator fidelity; or
- discretization/resolution.

The holdout must not influence optimizer selection, descriptor bounds, penalty
calibration, or stopping. If a validation replay changes a conclusion, publish
the change.

## Treat failure as a constraint

Numerical or physical failure should produce a named, calibrated constraint
violation feasible at `<= 0`, not a magic objective value mixed with valid
physics. Count failures by category and preserve representative diagnostics.
An optimizer arm that finds no feasible result is still an executed arm and
must not disappear from the comparison.

## Report

For the selected result report:

- nominal, aggregate, quantile, and worst-case metrics as appropriate;
- training and holdout scenario counts;
- failed-scenario count and failure taxonomy;
- constraints at full precision in artifacts;
- candidate and physical evaluation budgets;
- holdout degradation; and
- a sensitivity check when a grid or timestep can change feasibility.

The [phased-array tutorial](phased-array-codebook/README.md#robustness-protocol)
uses 49 deterministic training patterns and a 22-case holdout that changes
phase-error magnitude, failure adjacency, element spacing, and angular
resolution.
