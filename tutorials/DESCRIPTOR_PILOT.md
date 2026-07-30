# Descriptor-pilot protocol

MAP-Elites should be run because a measured behavior repertoire is the desired
product, not because a two-dimensional heatmap is convenient. Before a
tutorial makes a primary QD claim, run and publish a descriptor pilot.

## Pre-register

Before looking at a full archive, record:

1. the primary descriptor pair and why a user would select designs by it;
2. one physically meaningful fallback pair;
3. descriptor bounds and archive resolution;
4. the feasible-candidate generator, including seeds and budget;
5. a holdout that changes data, scenario, or perturbation kind; and
6. numeric pass/fail gates.

Decision variables may be included as a negative control, but a pair of raw
controls is not an emergent behavior map.

## Measure

Use feasible candidates only and report:

- reachable minimum and maximum for every coordinate;
- rank correlation between descriptor axes;
- lower- and upper-bound clipping fractions;
- occupied-niche coverage over at least three deterministic seed arms;
- same-niche retention from training to holdout;
- retention at one coarser resolution; and
- sensitivity to the numerical grid or measurement resolution when a
  descriptor is derived from sampled data.

Uniform random candidates are not automatically a fair pilot. When most of
the raw space is physically degenerate, mix structured seeds, known feasible
designs, or short search trajectories into a documented candidate generator.
Do not tune that mixture after observing the descriptor verdict.

## Decide

Apply the registered gates and write exactly one status:

- `accepted`: the primary pair clears every gate;
- `primary-secondary`: the pair remains useful as supporting evidence but
  misses a primary gate, or only a registered fallback clears the gates; or
- `rejected`: no registered pair is sufficiently reachable and stable.

Publish failed diagnostics. A rejected pilot is evidence about the
formulation, not a failed experiment. Do not enlarge bounds, coarsen the map,
or change the candidate mixture after seeing the result without declaring a
new protocol.

The [phased-array tutorial](phased-array-codebook/README.md#pre-registered-descriptor-pilot)
is a complete primary-secondary example. The ML tutorial records a rejection;
the RustPower tutorial shows why decision-led descriptors can be misleading.

## Artifacts

At minimum write:

```text
pilot.csv     one feasible observation per row, including train and holdout descriptors
pilot.md      human-readable gates, measurements, verdict, and reason
run.json      schema version, seeds, attempted/feasible counts, bounds, grid, and verdict
```

Keep full-precision values in CSV/JSON and round only the documentation table.
The archive campaign may proceed after a primary-secondary or rejected verdict
for educational comparison, but its claim ceiling must remain visible beside
every QD result.
