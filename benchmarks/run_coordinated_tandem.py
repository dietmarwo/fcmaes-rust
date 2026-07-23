#!/usr/bin/env python3
"""Run the original coordinated GTOP configuration on Tandem, resumably."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC_ROOT = ROOT.parent
WORK_ROOT = Path(
    os.environ.get(
        "FCMAES_BENCHMARK_WORKSPACE",
        PUBLIC_ROOT.parent / "fcmaes-benchmark-work",
    )
)
HEADER = [
    "problem",
    "run",
    "seed",
    "success",
    "value",
    "evaluations",
    "retries",
    "wall_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.runs <= 0 or args.workers <= 0:
        parser.error("runs and workers must be positive")
    return args


def read_shard(path: Path, seed: int) -> dict[str, str] | None:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if (
            len(rows) == 1
            and rows[0]["problem"] == "tandem"
            and int(rows[0]["seed"]) == seed
        ):
            return rows[0]
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return None


def main() -> None:
    args = parse_args()
    binary = PUBLIC_ROOT / "target" / "release" / "benchmark-gtop"
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(PUBLIC_ROOT / "Cargo.toml"),
            "-p",
            "fcmaes-examples",
            "--bin",
            "benchmark-gtop",
        ],
        check=True,
    )
    shard_dir = WORK_ROOT / "coordinated-tandem"
    shard_dir.mkdir(parents=True, exist_ok=True)
    invocation_started = time.monotonic()
    fresh_runs = 0
    rows: list[dict[str, str]] = []
    for run_index in range(args.runs):
        seed = args.seed + run_index
        shard = shard_dir / f"{run_index:03d}.tsv"
        row = read_shard(shard, seed)
        if row is None:
            command = [
                str(binary),
                "--problem",
                "tandem",
                "--runs",
                "1",
                "--workers",
                str(args.workers),
                "--evaluations",
                "1500",
                "--seed",
                str(seed),
                "--max-eval-fac",
                "50",
                "--check-interval",
                "100",
                "--raw-output",
                str(shard),
            ]
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            row = read_shard(shard, seed)
            if result.returncode != 0 or row is None:
                raise RuntimeError(
                    f"Tandem run {run_index + 1} failed with status "
                    f"{result.returncode}: {result.stderr.strip()}"
                )
            fresh_runs += 1
        row["run"] = str(run_index + 1)
        rows.append(row)
        elapsed = time.monotonic() - invocation_started
        print(
            f"coordinated Tandem completed={run_index + 1}/{args.runs} "
            f"seed={seed} success={row['success']} value={float(row['value']):.9f} "
            f"evaluations={row['evaluations']} wall={float(row['wall_seconds']):.3f}s "
            f"invocation_elapsed={elapsed:.1f}s",
            flush=True,
        )

    output = ROOT / "benchmark_gtop_tandem_100_raw.tsv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "problem": "Tandem",
        "runs": args.runs,
        "workers": args.workers,
        "root_seed": args.seed,
        "initial_evaluations_per_retry": 1_500,
        "max_evaluation_factor": 50.0,
        "diversity_checkpoint_interval": 100,
        "optimizer_sequence": "40% Differential Evolution, 60% active CMA-ES",
        "retry_cap": 20_000,
        "exact_configured_evaluation_ceiling_per_experiment": 765_000_000,
        "fresh_runs_in_invocation": fresh_runs,
        "invocation_wall_seconds": time.monotonic() - invocation_started,
        "raw_output": output.name,
    }
    (ROOT / "benchmark_gtop_tandem_100_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
