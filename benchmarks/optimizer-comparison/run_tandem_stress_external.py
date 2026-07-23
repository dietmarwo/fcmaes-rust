#!/usr/bin/env python3
"""Run the strongest equal-budget alternative on a large Tandem retry test."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC_ROOT = ROOT.parent.parent
EXTERNAL_ROOT = Path(
    os.environ.get("FCMAES_COMPARISON_WORKSPACE", PUBLIC_ROOT.parent / "fcmaes-optimizer-bench")
)
RAW = ROOT / "raw"
HEADER = [
    "library",
    "version",
    "algorithm",
    "parallel_mode",
    "problem",
    "run",
    "seed",
    "workers",
    "population_or_batch",
    "optimizer_runs",
    "configured_evaluations",
    "actual_evaluations",
    "absolute_best",
    "stop_fitness",
    "success",
    "value",
    "wall_seconds",
]

ADAPTERS = {
    ("cmaes", "CMA-ES", "parallel-population"): (
        "cmaes-benchmark-adapter",
        ["--mode", "population"],
    ),
    ("cmaes", "BIPOP-CMA-ES", "parallel-bipop"): (
        "cmaes-benchmark-adapter",
        ["--mode", "bipop"],
    ),
    ("genetic_algorithms", "L-SHADE", "native-serial"): (
        "genetic-algorithms-benchmark-adapter",
        [],
    ),
    ("math-optimisation", "DE/best/1/bin", "parallel-population"): (
        "math-optimisation-benchmark-adapter",
        [],
    ),
    ("argmin", "PSO", "parallel-population"): (
        "argmin-benchmark-adapter",
        [],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the alternative with the best mean Tandem optimum in the "
            "100-run comparison, then run independent long retries in parallel."
        )
    )
    parser.add_argument("--runs", type=int, default=1_000)
    parser.add_argument("--retry-workers", type=int, default=24)
    parser.add_argument("--evaluations", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.runs <= 0 or args.retry_workers <= 0 or args.evaluations <= 0:
        parser.error("runs, retry-workers, and evaluations must be positive")
    return args


def select_adapter() -> tuple[tuple[str, str, str], dict[str, float]]:
    values: dict[tuple[str, str, str], list[float]] = {
        adapter: [] for adapter in ADAPTERS
    }
    for path in sorted(RAW.glob("*.tsv")):
        if path.name.startswith("tandem_stress_") or path.name == "all_results.tsv":
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                key = (row["library"], row["algorithm"], row["parallel_mode"])
                if row["problem"] == "Tandem" and key in values:
                    values[key].append(float(row["value"]))
    incomplete = ["/".join(key) for key, samples in values.items() if len(samples) != 100]
    if incomplete:
        raise SystemExit(
            "the 100-run comparison must finish before automatic selection; "
            f"incomplete: {', '.join(incomplete)}"
        )
    means = {key: statistics.fmean(samples) for key, samples in values.items()}
    selected = min(means, key=means.__getitem__)
    printable = {"/".join(key): value for key, value in means.items()}
    return selected, printable


def valid_shard(path: Path, evaluations: int) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        return (
            len(rows) == 1
            and rows[0]["problem"] == "Tandem"
            and int(rows[0]["configured_evaluations"]) == evaluations
        )
    except (FileNotFoundError, KeyError, ValueError):
        return False


def run_one(
    run_index: int,
    args: argparse.Namespace,
    executable: Path,
    adapter_args: list[str],
    shard_dir: Path,
) -> tuple[int, Path]:
    shard = shard_dir / f"{run_index:04d}.tsv"
    if valid_shard(shard, args.evaluations):
        return run_index, shard
    command = [
        str(executable),
        *adapter_args,
        "--runs",
        "1",
        "--workers",
        "1",
        "--evaluations",
        str(args.evaluations),
        "--retries",
        "1",
        "--evaluations-per-retry",
        str(args.evaluations),
        "--seed",
        str(args.seed + run_index),
        "--problem",
        "tandem",
        "--output",
        str(shard),
    ]
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = "1"
    result = subprocess.run(
        command,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not valid_shard(shard, args.evaluations):
        raise RuntimeError(
            f"stress retry {run_index} failed with status {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return run_index, shard


def combine(shards: list[Path], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for run_index, shard in enumerate(shards):
            with shard.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source, delimiter="\t"))
            row = rows[0]
            row["run"] = str(run_index)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    selected, candidate_means = select_adapter()
    binary, adapter_args = ADAPTERS[selected]
    slug = "_".join(part.lower().replace("/", "_").replace("-", "_") for part in selected)
    subprocess.run(
        [str(ROOT / "prepare_external.sh"), str(EXTERNAL_ROOT)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--workspace",
            "--manifest-path",
            str(EXTERNAL_ROOT / "Cargo.toml"),
        ],
        check=True,
    )
    executable = EXTERNAL_ROOT / "target" / "release" / binary
    shard_dir = EXTERNAL_ROOT / "tandem-stress" / slug
    shard_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = 0
    shards: list[Path | None] = [None] * args.runs
    with ThreadPoolExecutor(max_workers=args.retry_workers) as pool:
        futures = {
            pool.submit(
                run_one,
                run_index,
                args,
                executable,
                adapter_args,
                shard_dir,
            ): run_index
            for run_index in range(args.runs)
        }
        for future in as_completed(futures):
            run_index, shard = future.result()
            shards[run_index] = shard
            completed += 1
            if completed % 10 == 0 or completed == args.runs:
                elapsed = time.monotonic() - started
                print(
                    f"tandem stress completed={completed}/{args.runs} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    elapsed = time.monotonic() - started
    complete_shards = [path for path in shards if path is not None]
    if len(complete_shards) != args.runs:
        raise SystemExit("internal error: missing completed stress shards")
    output = RAW / f"tandem_stress_{slug}.tsv"
    combine(complete_shards, output)
    metadata = {
        "selection_rule": "lowest mean final Tandem optimum in the 100-run comparison",
        "candidate_mean_optima": candidate_means,
        "selected": {
            "library": selected[0],
            "algorithm": selected[1],
            "parallel_mode_in_selection_run": selected[2],
        },
        "stress_parallelism": "independent optimizer processes",
        "retry_workers": args.retry_workers,
        "optimizer_threads_per_retry": 1,
        "runs": args.runs,
        "evaluations_per_retry": args.evaluations,
        "configured_total_evaluations": args.runs * args.evaluations,
        "root_seed": args.seed,
        "invocation_wall_seconds": elapsed,
        "raw_output": str(output.relative_to(ROOT)),
    }
    (ROOT / "tandem_stress_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(ROOT / "render_report.py")], check=True)


if __name__ == "__main__":
    main()
