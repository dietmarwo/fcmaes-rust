# Tutorial: durable fcmaes optimization jobs

The `fcmaes` worker accepts four profiles of the common job envelope:

| Profile | One task is | Result claim |
| --- | --- | --- |
| `fcmaes.retry/1` | one BiteOpt, DE or CMA-ES retry on a registered scalar objective | `best_found` |
| `fcmaes.mode/1` | one MODE run on a registered multi-objective model | `non_dominated_set` |
| `fcmaes.map-elites/1` | one MAP-Elites run on a registered quality-diversity model | `elite_archive` |
| `fcmaes.evaluate/1` | one given point, evaluated on any of these models | `feasible` or `none` |

Every optimizing task has its own seed, derived from the job seed and the task
ID, so results do not depend on scheduling. Every outcome reports `evidence`:
`rechecked` for every claim, because the worker re-evaluates what it returns
outside the optimizer, and `claimed` for `none` without a refutation (a
penalty value, an empty front or archive); see
[result contract 0.2](../../docs/job-api.md#termination-claim-and-evidence). The original `fcmaes.job/0.1` workflow
remains available; see [legacy experiment runner](../../docs/experiment-runner.md).

## Build and run the checked examples

From the repository root, on Linux with Rust 1.88+ and Python 3:

```sh
cargo build --release --locked -p optimization-runner -p fcmaes-job-worker
python3 tutorials/job-api/run.py --root /tmp/fcmaes-job-tutorial
```

The script initializes an eight-slot node, registers the executable, validates
and runs every example, then checks the results independently in Python. It
prints the attempt directories and writes `tutorial-report.json` under the
node. Re-running it with the same node and unmodified inputs returns the
existing attempts.

| Example | Profile | What the script checks |
| --- | --- | --- |
| `sphere.json`, `rosenbrock.json`, `cassini1.json` | BiteOpt, 16 × 1,000 evaluations | `best_found`, the hard budget |
| `rosenbrock-cmaes.json` | CMA-ES, 16 × 20,000 evaluations | `best_found`, the soft budget, best value below 1e-10 |
| `zdt1-mode.json` | MODE, 8 runs × 250 generations × 64 | the recomputed ZDT1 values, non-dominance, nothing below the true front, hypervolume above 0.85 (the true front has 0.8767) |
| `binh-korn-mode.json` | MODE, 8 runs × 100 generations × 64 | feasibility of every front point and non-dominance |
| `rastrigin-map-elites.json` | MAP-Elites, 8 runs × 1,000 generations × 20, 400 niches | the recomputed fitness and descriptors of every elite, coverage above 50% |
| `binh-korn-evaluate.json` | three given Binh-Korn points | the recomputed objectives and constraints; two points feasible, one not |

None of these claims is a global optimum or Pareto-optimality certificate.
Cassini1 uses the native GTOP model with its canonical bounds; nothing invokes
Python during optimization. If the `jsonschema` package is installed, the
script also validates every job and result against the published
[job](../../docs/job-api/schemas/optimization.job-0.1.schema.json) and
[result](../../docs/job-api/schemas/optimization.result-0.2.schema.json) schemas.

## Run one job manually

```sh
./target/release/optimization-run --root /tmp/fcmaes-manual node init --capacity 8
./target/release/optimization-run --root /tmp/fcmaes-manual workers add \
  --id fcmaes --executable ./target/release/fcmaes-job-worker
./target/release/optimization-run --root /tmp/fcmaes-manual problems describe fcmaes
./target/release/optimization-run --root /tmp/fcmaes-manual validate \
  --spec tutorials/job-api/zdt1-mode.json
./target/release/optimization-run --root /tmp/fcmaes-manual run \
  --spec tutorials/job-api/zdt1-mode.json
```

`problems describe fcmaes` lists each registered model with its `profile`, the
optimizers with their defaults and ranges, the evaluation profile, the claims
and the cancellation behavior. Read `run.json` in the returned `directory`, or use
`optimization-run result JOB_DIRECTORY`. `work.evaluations` sums optimizer
work. The additional `validation_evaluations` counter counts independent
re-evaluations: the best point of a retry, every member of MODE's final
population, every MAP-Elites elite.

Copy an example, choose a new `request_key`, and change the typed payload.
Unknown fields are rejected, not ignored. Bounds may be broadcast from a single
lower/upper pair; validation expands them and resolves every default into the
payload. Keep `inner_threads=1`: the parallelism is across tasks. Set
`outer_workers=1, cpu_slots=1` for a serial run with the same task seeds.

## Scalar retries: BiteOpt, DE and CMA-ES

`problem` selects `benchmark.sphere`, `benchmark.rosenbrock` or
`gtop.cassini1`; `retries` and `evaluations_per_retry` must be positive. The job's
wall limit bounds the entire batch after admission.

- **BiteOpt** (`kind: biteopt`) uses a hard limit: `evaluation_limit_mode: hard`.
- **DE** (`kind: de`) needs `evaluation_limit_mode: soft`. For population P, it
  may spend at most `2P-1` extra evaluations per retry.
- **CMA-ES** (`kind: cmaes`) needs `soft` too: it evaluates whole generations,
  so it may spend at most `population-1` extra evaluations. Its parameters are
  `population` (default 31) and `sigma` (default 0.3), the initial step size in
  the box normalized to `[-1, 1]` (0.3 is about 15% of each range). Each retry
  starts from a uniform random point drawn from its seed. CMA-ES can stop
  before its budget on its own convergence tests; the task then reports
  `termination: "optimizer_stop"` and `details.stop_code`. `sigma` is rejected
  for BiteOpt and DE.

BiteOpt and DE jobs resolve through the same definitions as legacy
`fcmaes.job/0.1` jobs, so both give identical retries.

`best_found` is the best point the optimizer evaluated, not a model mean or a
recommended point. The worker evaluates it again (`evidence.checks`:
`objective`, `bounds`). A point outside the box is reported with claim `none`
and `evidence.refuted` (`best_found`, verdict `infeasible_point`) and is not
evaluated, since the registered models are defined on their box only. A value
the point does not evaluate to gives verdict `objective_mismatch` and leaves
claim `feasible` with the recomputed value, or `none` if that value is itself
unusable. A value of `1e99` or more, the fcmaes penalty, is no answer: claim
`none`.

## Multi-objective MODE runs

`fcmaes.mode/1` runs MODE (multi-objective differential evolution with the
NSGA-II population update by default) for `generations` asked populations,
`runs` times. All objectives are minimized; constraints are feasible at or
below zero.

| Problem | Objectives / constraints | Dimension and box |
| --- | --- | --- |
| `benchmark.zdt1` | 2 / 0; front `f2 = 1 - sqrt(f1)` | 2–10,000, inside `[0,1]` |
| `benchmark.dtlz2` | 3 / 0; front on the unit sphere | 3–10,000, inside `[0,1]` |
| `benchmark.binh-korn` | 2 / 2 | 2, inside `[0,5] × [0,3]` |

`optimizer.parameters` takes `population` (4–10,000, or 4–1,000 with three
objectives because the exact hypervolume grows cubically; default 64),
`nsga_update` (default true), `pareto_update`, `f`, `cr`, `pro_c`, `dis_c`,
`pro_m` and `dis_m`, all defaulting to `fcmaes-core`'s `ModeParams`.
`reference_point` is optional and defaults to the problem's point in
discovery. Each run spends exactly `generations × population` evaluations.

A run's outcome has `claim: "non_dominated_set"` with no single `x` or
`objective`. `details.front` lists the feasible, mutually non-dominated members
of the final population, each with `x`, `objectives` and `constraints`, sorted
by the first objective. Every member was re-evaluated outside the optimizer.
`details.hypervolume` gives the exact hypervolume of the front with the
reference point and `normalization: "none"` (raw objectives); points outside it
are counted in `outside_reference`, not clipped. `details.front_source` is
`final_population`: unlike COCO's bi-objective archive of every non-dominated
point evaluated, the front holds the final population's survivors, so compare
hypervolumes only with the same reference point and source. A run without a
feasible member reports `claim: "none"`.

## MAP-Elites archives

`fcmaes.map-elites/1` fills a regular two-dimensional grid of
`archive.capacity` niches (1–10,000). The registered models are the linear
projection benchmarks of CMA-ME (Fontaine et al. 2020):
`benchmark.sphere-projection` and `benchmark.rastrigin-projection`, with an even
dimension, bounds inside `[-5.12, 5.12]`, minimized fitness, and as descriptor
the sums of the first and second halves of `x`.

`optimizer.parameters` takes `chunk_size` (even, default 20), `variation`
(`sbx`, the default, with `dis_c`/`dis_m`; or `iso_dd`, with
`iso_sigma`/`line_sigma` in raw variable units). Each run spends exactly
`generations × chunk_size` evaluations. CMA-ES emitter generations and CVT
archives are not part of this profile: their work is unbounded or parallel.

A run's outcome has `claim: "elite_archive"`. `details.elites` lists every
occupied niche with `x`, `fitness`, `descriptor` and `visits`; each elite was
re-evaluated and maps to its niche. `x` and `objective` are the best elite.
`details` also reports `coverage`, `qd_score` (the `fcmaes-core` definition),
`mean_fitness` and the grid layout. Compare `qd_score` only between archives
with the same fitness, descriptor bounds and capacity.

## Evaluating given points

`fcmaes.evaluate/1` evaluates points you supply, for example candidates an
agent proposed, on any registered model, without optimizing:

```json
"payload_schema": "fcmaes.evaluate/1",
"payload": {
  "problem": {"id": "benchmark.binh-korn", "version": "1", "config": {"dimension": 2},
              "bounds": {"lower": [0.0, 0.0], "upper": [5.0, 3.0], "broadcast": false}},
  "points": [[1.0, 1.0], [3.0, 2.0], [0.0, 3.0]]
}
```

Each point is one task with `termination: "evaluated"`. A scalar or
quality-diversity model gives claim `feasible` and the objective (fitness)
for a usable value; a multi-objective model gives `details.objectives` and
`details.constraints`, claim `feasible` when every constraint is at most zero,
and no single objective. Points must lie inside the problem's bounds, where
the registered models are defined; validation rejects any other point.
`work.evaluations` counts one per point. Points are inline in the job, which
the runner's validation echoes within 4 MiB: keep all points together well
below that (about 100,000 two-dimensional points).

## Cancellation and deadlines

Retries check cancellation between retries; an active retry may continue until
its evaluation budget. MODE and MAP-Elites runs check it between generations:
an active run stops and returns its current front or archive, re-evaluated as
usual, with `termination: "cancelled"`. The SDK uses the same signal at the wall
deadline, so read the job state (`cancelled` or `timed_out`) for the reason.
The runner's process deadline remains authoritative; forced termination
preserves committed tasks and marks total work unknown.

## Use existing jobs and extend the registries

```sh
cargo build --release --locked -p fcmaes-runner -p fcmaes-native-worker
./target/release/fcmaes-run --root /tmp/fcmaes-legacy node init --capacity 8
./target/release/fcmaes-run --root /tmp/fcmaes-legacy workers add \
  --id native --executable ./target/release/fcmaes-native-worker
./target/release/fcmaes-run --root /tmp/fcmaes-legacy run \
  --spec docs/runner-jobs/sphere-bite.json
```

The legacy manifest and `retries.jsonl` retain their original meaning. Both
workers evaluate scalar objectives through one registry, so a new scalar
objective needs three changes: its validation and discovery entry in
`crates/optimization-job-api/src/legacy/spec.rs` (a shared file), its function
in `objective_by_id` in `crates/fcmaes-native-worker/src/lib.rs`, and tests for
bounds, budgets and deterministic seeds. Multi-objective and quality-diversity
models live in `crates/fcmaes-job-worker/src/problems.rs`, which only this
worker uses. Add a versioned profile if the mathematical meaning of a job
changes. Jobs select registered models; they cannot deserialize a function or
execute arbitrary source text.

See the [shared API guide](../../docs/job-api.md) for mixed-worker nodes,
lifecycle inspection, cancellation and recovery. Run `python3
scripts/job-api/test_integration.py` for local adapter acceptance, and
`python3 scripts/runner/test_integration.py` for legacy compatibility.
