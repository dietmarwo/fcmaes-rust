#!/usr/bin/env python3
"""Round-4 candidate driver: ARRDE from the Minion library on `truss_sizing`.

Runs `minionpy.algorithms.ARRDE` (Minion 2.0.1, built from source at the
revision pinned below, default parameters) under protocol T and appends rows
with solver name `arrde` to a raw file in the format of the Rust runner, with
the same resume key (ROUND4.md sections 2 to 4; DEVIATIONS.md, round 4).

    python3 run_arrde.py run --out results/round4/stage1-cap125000.tsv
        --total 1000000 --run-cap 125000 --root-seed 7 [--reps 10]
        [--min-run 1000] [--problems truss_sizing] [--harness PATH]
        [--minion-src DIR]
    python3 run_arrde.py info [--minion-src DIR]

How it works:

- The library gets the box of the problem, the run's evaluation budget as
  `maxevals` and the seed, nothing else (no start point, default options).
  Its default population size depends on the budget and on the dimension;
  the first batch of every run is recorded in the detail file.
- The library evaluates whole populations. Each batch goes through
  `candidate-harness serve` in one request, bit-exact, as in
  `run_catcmawm.py`, whose bridge, seed derivation, guard, lock and raw-file
  helpers are imported.
- A `Guard` per run enforces the run allowance and the box and records the
  best value itself. The library overshoots its budget by a few evaluations;
  the guard refuses the points beyond the allowance (answering `+inf`,
  counted in `over_budget`), so a run never uses more than its allowance. A
  point outside the box is refused and counted in `out_of_bounds`; nothing is
  clipped.
- Protocol T as in the Rust runner: run k gets min(run cap, remaining)
  evaluations and the seed `run_seed(root, problem, 'T', rep, k)`, of which
  the library gets the low 31 bits (DEVIATIONS.md, round 4, entry 2).
  Restarting ends when fewer than `min_run` evaluations remain, or after a
  run without evaluations.

The Rust runner and this driver must not write one file at the same time;
both take the lock file `<out>.lock`.

Besides the raw rows, the driver appends one informational line per row to
`<out stem>.arrde.tsv` next to the raw file. The analysis does not read it.

Every check raises an exception; the script never uses `assert`, so it
behaves identically under `python -O`.
"""

from __future__ import annotations

import os

# One BLAS thread per process; the objective runs in the harness process.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import run_catcmawm as base  # noqa: E402
from run_catcmawm import DriverError  # noqa: E402

MINION_VERSION = "2.0.1"
MINION_COMMIT = "535504a902575bf734862f238c7483da39bbdbd5"
SOLVER = "arrde"
PROBLEMS = ["truss_sizing"]
MIN_RUN = 1000
SEED_MASK = 0x7FFFFFFF
DETAIL_HEADER = "\t".join([
    "problem", "total", "run_cap", "rep", "runs", "first_batch", "library_calls",
    "evaluations", "over_budget", "out_of_bounds", "objective_ms", "bridge_ms",
    "wall_ms",
])


# ------------------------------------------------------------------ library

def check_minion(source: str | None = None):
    """Return `minionpy.algorithms.ARRDE` after checking the version and the
    MIT license of the installed package; with `source` (a checkout of
    Minion), also the checked-out commit."""
    try:
        version = metadata.version("minionpy")
    except metadata.PackageNotFoundError as error:
        raise DriverError("minionpy is not installed (build Minion 2.0.1 from source, "
                          "README, section Round 4)") from error
    if version != MINION_VERSION:
        raise DriverError(f"minionpy {version} is installed; this driver is pinned to "
                          f"{MINION_VERSION}")
    meta = metadata.metadata("minionpy")
    license_text = (meta.get("License-Expression") or meta.get("License") or "").strip()
    classifiers = meta.get_all("Classifier") or []
    if license_text != "MIT" and "License :: OSI Approved :: MIT License" not in classifiers:
        raise DriverError(f"minionpy {version}: license '{license_text}' is not MIT")
    if source is not None:
        head = subprocess.run(["git", "-C", source, "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        if head.returncode != 0 or head.stdout.strip() != MINION_COMMIT:
            raise DriverError(f"{source} is not at commit {MINION_COMMIT}")
    from minionpy.algorithms import ARRDE
    return ARRDE


def library_seed(seed: int) -> int:
    """Low 31 bits of the run seed; never -1, which the library reads as
    'random seed' (DEVIATIONS.md, round 4, entry 2)."""
    return seed & SEED_MASK


# --------------------------------------------------------------- formatting

def format_row(problem: str, domain: str, dim: int, rep: int, settings: dict,
               result: dict) -> str:
    """One protocol-T raw row (with newline), as `format_row_t` of the
    harness; `settings` holds the total and the run cap."""
    fields = [
        SOLVER, problem, domain, str(dim), "T", str(rep), str(result["seed"]),
        str(settings["total"]), "1e0", str(result["evaluations"]),
        base.format_float(result["best_value"]), str(result["over_budget"]),
        str(result["out_of_bounds"]), f"{result['wall_ms']:.3f}", str(result["runs"]),
        str(settings["root_seed"]), str(settings["total"]), str(settings["run_cap"]),
        str(settings["min_run"]),
    ]
    return "\t".join(fields) + "\n"


# ----------------------------------------------------------------- protocol

def run_once(optimizer_class, info: dict, guard: base.Guard, seed: int,
             evaluate_batch) -> dict:
    """One ARRDE run behind `guard`: bounds, budget and seed only."""
    batches: list[int] = []

    def objective(points):
        vectors = [np.asarray(p, dtype=np.float64) for p in points]
        batches.append(len(vectors))
        return guard.evaluate(vectors, evaluate_batch)

    bounds = [(float(a), float(b)) for a, b in zip(info["lower"], info["upper"])]
    optimizer = optimizer_class(objective, bounds, maxevals=guard.budget,
                                seed=library_seed(seed))
    optimizer.optimize()
    return {"first_batch": batches[0] if batches else 0, "library_calls": sum(batches)}


def run_repetition(optimizer_class, problem: str, info: dict, rep: int, settings: dict,
                   evaluate_batch) -> dict:
    """One protocol-T repetition, as `run_repetition_t` of the harness."""
    lower, upper = info["lower"], info["upper"]
    result = {
        "seed": base.run_seed(settings["root_seed"], problem, rep, 0),
        "evaluations": 0, "best_value": math.inf, "over_budget": 0,
        "out_of_bounds": 0, "wall_ms": 0.0, "runs": 0, "stopped_empty": False,
        "first_batch": 0, "library_calls": 0, "seeds": [], "best_x": None,
    }
    remaining = settings["total"]
    while remaining >= max(settings["min_run"], 1):
        allowance = min(settings["run_cap"], remaining)
        seed = base.run_seed(settings["root_seed"], problem, rep, result["runs"])
        guard = base.Guard(lower, upper, allowance)
        started = time.perf_counter()
        outcome = run_once(optimizer_class, info, guard, seed, evaluate_batch)
        result["wall_ms"] += (time.perf_counter() - started) * 1e3
        result["seeds"].append(seed)
        if result["runs"] == 0:
            result["first_batch"] = outcome["first_batch"]
        result["library_calls"] += outcome["library_calls"]
        calls = guard.calls
        result["evaluations"] += calls
        result["over_budget"] += guard.over_budget
        result["out_of_bounds"] += guard.out_of_bounds
        if guard.best_y < result["best_value"]:
            result["best_value"] = guard.best_y
            result["best_x"] = guard.best_x
        result["runs"] += 1
        remaining -= calls
        if calls == 0:
            result["stopped_empty"] = True
            break
    return result


# ------------------------------------------------------------------ workers

def _task(serve: base.Serve, optimizer_class, task: tuple) -> tuple:
    problem, rep, settings = task
    info = serve.info(problem)
    objective_before, bridge_before = serve.objective_ns, serve.bridge_ns
    result = run_repetition(optimizer_class, problem, info, rep, settings,
                            lambda vectors: serve.evaluate(problem, vectors))
    result["objective_ms"] = (serve.objective_ns - objective_before) / 1e6
    result["bridge_ms"] = (serve.bridge_ns - bridge_before) / 1e6
    line = format_row(problem, info["domain"], info["dim"], rep, settings, result)
    detail = "\t".join(str(v) for v in [
        problem, settings["total"], settings["run_cap"], rep, result["runs"],
        result["first_batch"], result["library_calls"], result["evaluations"],
        result["over_budget"], result["out_of_bounds"], f"{result['objective_ms']:.3f}",
        f"{result['bridge_ms']:.3f}", f"{result['wall_ms']:.3f}"]) + "\n"
    return task, line, detail, result["stopped_empty"], result["runs"]


# --------------------------------------------------------------- commands

def detail_path(out: Path) -> Path:
    return out.with_name(out.stem + ".arrde.tsv")


def command_run(args) -> int:
    optimizer_class = check_minion(args.minion_src)
    out = Path(args.out)
    problems = [p for p in (v.strip() for v in args.problems.split(",")) if p]
    if not problems or len(set(problems)) != len(problems):
        raise DriverError("--problems needs distinct problem names")
    if min(args.reps, args.min_run, args.total, args.run_cap) <= 0 or args.root_seed < 0:
        raise DriverError("--reps, --min-run, --total and --run-cap must be positive")
    if args.run_cap > args.total:
        raise DriverError("--run-cap exceeds --total")
    settings = {"root_seed": args.root_seed, "total": args.total,
                "run_cap": args.run_cap, "min_run": args.min_run}
    print(f"arrde (minionpy {MINION_VERSION}, commit {MINION_COMMIT[:7]}), protocol T: "
          f"root seed {args.root_seed}, total {args.total}, run cap {args.run_cap}, "
          f"min run {args.min_run}, seed = low 31 bits of the run seed", flush=True)
    with base.RunLock(out):
        existing = base.load_or_create(out)
        expected = (settings["root_seed"], settings["total"], settings["run_cap"],
                    settings["min_run"])
        for row in existing:
            found = (row["root_seed"], row["total"], row["run_cap"], row["min_run"])
            if found != expected:
                raise DriverError(f"{out} holds rows with protocol-T settings {found} "
                                  f"(current {expected}); use another output file")
        done = {base.row_key(row) for row in existing}
        tasks = [(problem, rep, settings) for problem in problems for rep in range(args.reps)
                 if (SOLVER, problem, "T", rep, settings["total"]) not in done]
        print(f"{len(tasks)} runs to do ({len(done)} rows already present)", flush=True)
        detail = detail_path(out)
        if not detail.exists():
            detail.write_text(DETAIL_HEADER + "\n")
        serve = base.Serve(args.harness or base.default_harness())
        started = time.perf_counter()
        written = 0
        try:
            for task in tasks:
                _, line, detail_line, stopped_empty, runs = _task(serve, optimizer_class, task)
                problem, rep, _ = task
                base.append_line(out, line)
                base.append_line(detail, detail_line)
                written += 1
                if stopped_empty:
                    print(f"warning: arrde on {problem} rep {rep}: run {runs - 1} made no "
                          "evaluation", file=sys.stderr)
                print(f"done arrde {problem} {settings['total']}/{settings['run_cap']} "
                      f"rep {rep} ({written}/{len(tasks)}, elapsed "
                      f"{time.perf_counter() - started:.1f}s)", flush=True)
        finally:
            serve.close()
    print(f"wrote {written} rows to {out}")
    return 0


def command_info(args) -> int:
    check_minion(args.minion_src)
    print(f"minionpy {metadata.version('minionpy')} (MIT), pinned commit {MINION_COMMIT}")
    print("ARRDE default population: clamp(D * m, 4, 3000), at least 2 D, with "
          "m = 2 + 5.756 (log10(maxevals / D) - 2)^1.609 for maxevals / D > 100 "
          "(else 2); see the library's arrde.cpp")
    serve = base.Serve(base.default_harness())
    try:
        for problem in PROBLEMS:
            dim = serve.info(problem)["dim"]
            for evals in (100_000, 125_000, 500_000, 1_000_000, 4_000_000):
                print(f"{problem}\tD={dim}\tmaxevals={evals}\tpopulation={default_population(dim, evals)}")
    finally:
        serve.close()
    return 0


def default_population(dim: int, maxevals: int) -> int:
    """Default initial population of the library for `dim` and `maxevals`,
    restated from its source (informational; the driver records the first
    batch actually seen)."""
    logeta = math.log10(maxevals / dim)
    multiplier = 2.0 + 5.756 * (logeta - 2.0) ** 1.609 if logeta > 2.0 else 2.0
    population = int(min(max(dim * multiplier, 4.0), 3000.0))
    return max(population, 2 * dim, 4)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "info"):
        p = sub.add_parser(name)
        p.add_argument("--minion-src", default=None,
                       help="checkout of Minion whose commit is verified (optional)")
        p.add_argument("--harness", default=None,
                       help="candidate-harness binary (default $CARGO_TARGET_DIR/release/"
                            "candidate-harness)")
        if name == "run":
            p.add_argument("--out", required=True, help="raw TSV (one shape per file)")
            p.add_argument("--problems", default=",".join(PROBLEMS))
            p.add_argument("--total", type=int, required=True)
            p.add_argument("--run-cap", type=int, required=True)
            p.add_argument("--root-seed", type=int, required=True)
            p.add_argument("--min-run", type=int, default=MIN_RUN)
            p.add_argument("--reps", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return {"run": command_run, "info": command_info}[args.command](args)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except DriverError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
