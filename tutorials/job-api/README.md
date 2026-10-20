# Tutorial: durable fcmaes optimization jobs

The `fcmaes` worker accepts these profiles of the common job envelope, plus
the experimental `fcmaes.surrogate/1`, which has
[its own tutorial](../surrogate/README.md):

| Profile | One task is | Result claim |
| --- | --- | --- |
| `fcmaes.retry/1` | one BiteOpt, DE, CMA-ES (optionally with IPOP or BIPOP restarts), CR-FM-NES or PGPE retry on a registered scalar model | `best_found` |
| `fcmaes.mode/1` | one MODE run on a registered multi-objective model | `non_dominated_set` |
| `fcmaes.map-elites/1` | one MAP-Elites run on a grid or CVT archive of a registered quality-diversity model, optionally followed by CMA-ME or CMA-MAE emitters | `elite_archive` |
| `fcmaes.evaluate/1` | one given point, evaluated on any of these models | `feasible` or `none` |

A job names a registered model; it never carries code. The stock worker
registers benchmark models. An application builds its own worker that
registers its models as Rust closures, which can read parameters and
SHA-256-checked data files from the job
([application models](../../docs/job-api/models.md)).

Every optimizing task has its own seed, derived from the job seed and the task
ID, so results do not depend on scheduling. Every outcome reports `evidence`:
`rechecked` for every claim, because the worker re-evaluates what it returns
outside the optimizer, and `claimed` for `none` without a refutation (a
penalty value, an empty front or archive); see
[result contract 0.2](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#termination-claim-and-evidence). The original `fcmaes.job/0.1` workflow
remains available; see [legacy experiment runner](../../docs/experiment-runner.md).

## Build and run the checked examples

From the repository root, on Linux with Rust 1.88+ and Python 3:

```sh
cargo build --release --locked -p optimization-runner -p fcmaes-job-worker
cargo build --release --locked -p fcmaes-job-worker --example custom_worker
python3 tutorials/job-api/run.py --root /tmp/fcmaes-job-tutorial
```

The script takes the binaries, the example's included, from `target/release`,
or from `$CARGO_TARGET_DIR/release` when that is set; `--bin-dir` overrides
both. It initializes an eight-slot node, registers the stock worker as
`fcmaes` and the example application worker as `curve-fit`, validates and runs
every example, then checks the results independently in Python. It
prints the attempt directories and writes `tutorial-report.json` under the
node. Re-running it with the same node and unmodified inputs returns the
existing attempts.

| Example | Profile | What the script checks |
| --- | --- | --- |
| `sphere.json`, `rosenbrock.json`, `cassini1.json` | BiteOpt, 16 × 1,000 evaluations | `best_found`, the hard budget |
| `rosenbrock-cmaes.json` | CMA-ES, 16 × 20,000 evaluations | `best_found`, the soft budget, best value below 1e-10 |
| `cassini1-bipop.json` | CMA-ES with BIPOP restarts, 8 × 50,000 evaluations, 2 workers × 4 threads | at least one restart per retry, the soft budget, best value below 5.4 (the best known is 4.9307) |
| `rosenbrock-crfmnes.json` | CR-FM-NES on 50 dimensions, 8 × 100,000 evaluations, 4 workers × 2 threads | `best_found`, best value below 1e-20 |
| `zdt1-mode.json` | MODE, 8 runs × 250 generations × 64 | the recomputed ZDT1 values, non-dominance, nothing below the true front, hypervolume above 0.85 (the true front has 0.8767) |
| `binh-korn-mode.json` | MODE, 8 runs × 100 generations × 64 | feasibility of every front point and non-dominance |
| `rastrigin-map-elites.json` | MAP-Elites, 8 runs × 1,000 generations × 20, 400 niches | the recomputed fitness and descriptors of every elite, coverage above 50% |
| `sphere-cvt-cma-mae.json` | MAP-Elites on a 200-niche CVT archive, then CMA-MAE: 100 × 20 + 300 × 4 emitters × 16 per run, 2 workers × 4 threads | the exact budget, every elite recomputed, coverage above 50% |
| `binh-korn-evaluate.json` | three given Binh-Korn points | the recomputed objectives and constraints; two points feasible, one not |
| `exponential-fit.json` | worker `curve-fit`: CMA-ES fits `a·exp(-b·t) + c` to the input `decay.csv` | the default `loss` recorded in the resolved job, the recomputed loss, `(a, b, c)` near `(3, 0.7, 0.5)` |

None of these claims is a global optimum or Pareto-optimality certificate.
Cassini1 uses the native GTOP model with its canonical bounds; nothing invokes
Python during optimization. If the `jsonschema` package is installed, the
script also validates every job and result against the published
[job](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api/schemas/optimization.job-0.1.schema.json) and
[result](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api/schemas/optimization.result-0.2.schema.json)
schemas, from Cargo's checkout of optimization-jobs at the pinned revision.

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
payload. Set `outer_workers=1, cpu_slots=1` for a serial run with the same task
seeds.

## Through an agent (MCP)

The runner's build also makes `optimization-mcp`, an MCP server over the same
node. With the node of the previous section, connect an MCP client, for
example Claude Code:

```sh
claude mcp add fcmaes-jobs -- "$PWD/target/release/optimization-mcp" --root /tmp/fcmaes-manual
```

The agent sees the worker with `describe_worker` (worker `fcmaes`), can pass
any job of this directory to `submit_job` as `spec_path`, follows it with
`wait_job` and reads the per-task claims and evidence with `get_result`;
`cancel_job` stops a run gracefully, keeping completed retries. With the MCP
Tasks extension, `submit_job` returns a task that completes with the job's
result. See [Agents and MCP](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#agents-and-mcp).

## Threads: `outer_workers` and `inner_threads`

`outer_workers` runs tasks (retries, runs, points) concurrently.
`inner_threads` evaluates the population of one task in parallel, on threads
of that task's own; `outer_workers * inner_threads` must not exceed
`cpu_slots`. Use `inner_threads > 1` for CMA-ES, CR-FM-NES, PGPE, MODE and
MAP-Elites, and for BiteOpt and DE with `batch`, when there are fewer tasks
than slots or one evaluation is expensive. A run's result does not depend on
`inner_threads`: populations are evaluated in parallel but returned in order,
and the optimizer's state stays serial. Serial BiteOpt and DE evaluate one
point at a time and `fcmaes.evaluate/1` one point per task, so they require
`inner_threads=1`.

## Scalar retries: BiteOpt, DE, CMA-ES, CR-FM-NES and PGPE

`problem` selects a registered scalar model: `benchmark.sphere`,
`benchmark.rosenbrock`, `gtop.cassini1`, `benchmark.branin` or
`benchmark.hartmann3` in the stock worker; `retries` and
`evaluations_per_retry` must be positive. The job's wall limit bounds the
entire batch after admission.

- **BiteOpt** (`kind: biteopt`) uses a hard limit: `evaluation_limit_mode: hard`.
- **DE** (`kind: de`) needs `evaluation_limit_mode: soft`. For population P, it
  may spend at most `2P-1` extra evaluations per retry.
- **BiteOpt and DE with `batch`** evaluate `batch` candidates concurrently
  (on `inner_threads` threads), with delayed feedback. BiteOpt asks, evaluates
  and tells batches of `batch` candidates. DE keeps `batch` candidates in
  flight, the delayed update of the Python fcmaes `de.minimize(workers=...)`
  that the Rust DE derives from: when the oldest candidate is evaluated, it is
  told and one new candidate is asked. Unlike the Python version, results are
  told in the order the candidates were asked, so the run depends on `batch`
  and the seed, never on thread timing. Neither exceeds
  `evaluations_per_retry`. Delayed feedback costs some convergence per
  evaluation, more for DE than for BiteOpt; keep `batch` small against the
  population (DE's default is 31) and use it when evaluations are expensive.
  Without `batch`, both run the serial algorithm of `fcmaes.job/0.1`.
- **CMA-ES** (`kind: cmaes`) needs `soft` too: it evaluates whole generations,
  so it may spend at most `population-1` extra evaluations. Its parameters are
  `population` (default 31) and `sigma` (default 0.3), the initial step size in
  the box normalized to `[-1, 1]` (0.3 is about 15% of each range). Each retry
  starts from a uniform random point drawn from its seed. CMA-ES can stop
  before its budget on its own convergence tests; the task then reports
  `termination: "optimizer_stop"` and `details.stop_code`.
- **CMA-ES with restarts** (`restarts: ipop` or `bipop`) restarts until the
  retry's budget is spent, each run from a new uniform random point. IPOP
  (Auger and Hansen 2005) doubles the population after every stop. BIPOP
  (Hansen 2009) alternates doubled populations with small populations and
  smaller step sizes, choosing the regime that has used fewer evaluations.
  Populations stop growing at `max_population` (default 512 × `population`,
  at most 10⁶ and 10⁷ / dimension), which bounds the overshoot to
  `max_population-1`. `details.restarts` and
  `details.largest_population` report what happened.
- **CR-FM-NES** (`kind: crfmnes`, Nomura and Ono 2022) is a natural evolution
  strategy with a restricted covariance whose cost is linear in the dimension,
  for problems with hundreds or thousands of variables. `population` (even,
  default 32) and `sigma` (default 0.3, normalized box) as for CMA-ES.
- **PGPE** (`kind: pgpe`, Sehnke et al. 2010) samples mirrored pairs around a
  center it moves with Adam; a common choice for neural-network weights.
  `population` (even, default 32), `sigma` (default 0.1), and the
  `center_learning_rate` (0.15) and `stdev_learning_rate` (0.1).

Settings of another optimizer are rejected, never ignored: `sigma` for
BiteOpt, `restarts` for CR-FM-NES, `center_learning_rate` for CMA-ES. CMA-ES,
CR-FM-NES and PGPE use a soft limit (`evaluation_limit_mode: soft`) and may
spend `population-1` extra evaluations. CR-FM-NES adds a bound penalty to the
values it ranks and keeps unclamped points, so for CR-FM-NES and PGPE the
worker takes the best point from the evaluated, clamped populations.
CMA-ES holds dimension × dimension matrices and accepts at most 3,162
variables; CR-FM-NES and PGPE scale further. `population × dimension` (with
restarts `max_population × dimension`, and `batch × dimension`) is at most
10⁷ for every optimizer, BiteOpt and DE included: they allocate their
population tables before the first evaluation, so the evaluation budget does
not bound that memory. BiteOpt's default population, `9 + 3 × dimension`,
exceeds the limit from dimension 1,825 on; set a smaller `population` there.

BiteOpt and DE jobs without `batch` resolve through the same definitions as
legacy `fcmaes.job/0.1` jobs, so both give identical retries.

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
`details.hypervolume` gives the hypervolume of the front with the reference
point and `normalization: "none"` (raw objectives): exact through four
objectives (`method: "exact"`, `exact: true`); for an application model with
more objectives a deterministic Monte Carlo estimate (`method: "monte_carlo"`,
`exact: false`, with `samples`, `seed` and `standard_error`). Points outside
the reference are counted in `outside_reference`, not clipped. `details.front_source` is
`final_population`: unlike COCO's bi-objective archive of every non-dominated
point evaluated, the front holds the final population's survivors, so compare
hypervolumes only with the same reference point and source. A run without a
feasible member reports `claim: "none"`.

## MAP-Elites archives

`fcmaes.map-elites/1` fills an archive of `archive.capacity` niches
(1–10,000). The stock worker registers the linear projection benchmarks of
CMA-ME (Fontaine et al. 2020): `benchmark.sphere-projection` and
`benchmark.rastrigin-projection`, with an even dimension, bounds inside
`[-5.12, 5.12]`, minimized fitness, and as descriptor the sums of the first and
second halves of `x`.

The archive is a regular grid (`archive.kind: grid`, the default for
two-dimensional descriptors) or a CVT archive (`kind: cvt`, the default for
any other descriptor dimension; Vassiliades et al. 2016): `capacity` niches
from k-means over `samples_per_niche` (default 20) uniform samples per niche,
with `capacity × samples_per_niche ≤ 10⁶` and `capacity² × samples_per_niche
× descriptor dimension ≤ 10⁹`. Descriptors have 1–100 dimensions. The tessellation comes from
`archive.seed` (default: the job seed), so every run of a job, and every job
with the same archive settings, shares its niches.

A run has two phases with exact budgets. First `generations` of the
`chunk_size` (even, default 20) emitter: `variation: sbx`, the default, with
`dis_c`/`dis_m`, or `iso_dd`, with `iso_sigma`/`line_sigma` in raw variable
units. Then, with `cma_generations > 0`, `emitters` CMA-ES instances (default
4; `emitters × dimension² ≤ 10⁷`) each evaluate `population` candidates
(default 16, initial `sigma` 0.1 in the normalized box) per generation, ranked
by archive improvement:

- `emitter: cma_me` (default): CMA-ME's improvement ranking (Fontaine et al.
  2020). Candidates that fill an empty niche come first, by fitness; then
  candidates that beat their niche's elite, by `fitness - elite`; then the
  rest.
- `emitter: cma_mae`: CMA-MAE (Fontaine and Nikolaidis 2023) ranks by
  `fitness - threshold` of the niche; a fitness below the threshold moves the
  threshold towards it by `archive_learning_rate` (default 0.01).
  `threshold_init`, the threshold of an empty niche, is required: a fitness
  worth keeping, comparable to pyribs' `threshold_min`.

An emitter restarts from a random elite when CMA-ES stops or a generation
improves nothing. A run spends exactly `generations × chunk_size +
cma_generations × emitters × population` evaluations; `generations` may be 0
if `cma_generations` is positive. The emitter settings are rejected without
`cma_generations` and absent from the resolved job then.

A run's outcome has `claim: "elite_archive"`. `details.elites` lists every
occupied niche with `x`, `fitness`, `descriptor` and `visits`; each elite was
re-evaluated and maps to its niche. `x` and `objective` are the best elite.
`details` also reports `coverage`, `qd_score` (the `fcmaes-core` definition),
`mean_fitness`, `archive_kind`, the grid layout (null for CVT) and, with
emitters, `cma_generations` and `emitter_restarts`. Compare `qd_score` only
between archives with the same fitness, descriptor bounds and niches.

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
its evaluation budget. MODE and MAP-Elites runs check it between generations,
in both MAP-Elites phases:
an active run stops and returns its current front or archive, re-evaluated as
usual, with `termination: "cancelled"`. The SDK uses the same signal at the wall
deadline, so read the job state (`cancelled` or `timed_out`) for the reason.
The runner's process deadline remains authoritative; forced termination
preserves committed tasks and marks total work unknown.

## Your own models

The stock worker's models are benchmarks. Real problems run in an
application worker: a small binary that registers its models in a
`Registry` and runs the fcmaes adapter. `examples/custom_worker.rs` in
`crates/fcmaes-job-worker` registers `app.exponential-fit`, which reads its
data from the job's input `data` and takes a `loss` parameter;
`exponential-fit.json` runs it:

```json
"worker": "curve-fit",
"inputs": [{"name": "data", "file": "decay.csv", "sha256": "d864222f..."}],
"payload": {"problem": {"id": "app.exponential-fit", "version": "1",
    "config": {"dimension": 3},
    "bounds": {"lower": [0.0, 0.0, -10.0], "upper": [10.0, 5.0, 10.0], "broadcast": false}},
  "optimizer": {"kind": "cmaes", "parameters": {}}, ...}
```

The runner copies the input into the attempt and checks its SHA-256; the
resolved job records `"parameters": {"loss": "squared"}`, the default the
model used. [Application models](../../docs/job-api/models.md) explains the
registry, the checks and how to build and register such a worker.

## Legacy jobs

```sh
cargo build --release --locked -p fcmaes-runner -p fcmaes-native-worker
./target/release/fcmaes-run --root /tmp/fcmaes-legacy node init --capacity 8
./target/release/fcmaes-run --root /tmp/fcmaes-legacy workers add \
  --id native --executable ./target/release/fcmaes-native-worker
./target/release/fcmaes-run --root /tmp/fcmaes-legacy run \
  --spec docs/runner-jobs/sphere-bite.json
```

The legacy manifest and `retries.jsonl` retain their original meaning. The
legacy worker and `fcmaes.retry/1` evaluate the three built-in scalar models
with the same functions (`objective_by_id` in
`crates/fcmaes-native-worker/src/lib.rs`), so BiteOpt and DE retries are
identical in both. Add a versioned profile if the mathematical meaning of a
job changes.

See the [shared API guide](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md) for mixed-worker nodes, lifecycle
inspection, cancellation and recovery, and [shared job API](../../docs/job-api.md)
for running the shared acceptance harness against this repository's worker.
Run `python3 scripts/runner/test_integration.py` for legacy compatibility.
