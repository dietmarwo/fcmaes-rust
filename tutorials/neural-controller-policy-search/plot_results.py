#!/usr/bin/env python3
"""Render deterministic SVG figures from the policy-search CSV outputs."""

from __future__ import annotations

import argparse
import csv
import filecmp
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "publication"
OUTPUT = ROOT / "images" / "publication"
FIGURES = ("quality.svg", "scaling.svg", "convergence.svg", "trajectory.svg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.hashsalt": "fcmaes-neural-controller-policy-search",
    }
)

COLORS = {
    "pgpe": "#0072B2",
    "crfmnes": "#009E73",
    "cmaes": "#D55E00",
    "biteopt": "#CC79A7",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_results() -> None:
    """Reject incomplete or internally inconsistent publication data."""
    run_rows = rows(RESULTS / "runs.csv")
    expected = {
        ("quality", "fixed"): 20,
        ("noise", "rotating"): 20,
        ("scaling", "fixed"): 60,
    }
    actual: dict[tuple[str, str], int] = defaultdict(int)
    for row in run_rows:
        actual[(row["experiment"], row["scenario_mode"])] += 1
        if int(row["popsize"]) != 64:
            raise ValueError("publication runs must use population 64")
        if int(row["evaluations"]) != 20_480:
            raise ValueError("publication runs must use 20,480 candidate evaluations")
        if int(row["optimizer_rollouts"]) != 81_920:
            raise ValueError("publication runs must use 81,920 optimizer rollouts")
        if (
            int(row["train_scenarios"]) != 4
            or int(row["validation_scenarios"]) != 128
            or int(row["horizon"]) != 300
        ):
            raise ValueError("publication scenario or horizon protocol changed")
    if dict(actual) != expected:
        raise ValueError(f"unexpected publication run groups: {dict(actual)}")

    invariant_columns = (
        "train_best",
        "validation_score",
        "mean_loss",
        "cvar_loss",
        "success_rate",
        "mean_steps",
        "rms_force",
    )
    scaling: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    for row in run_rows:
        if row["experiment"] == "scaling":
            scaling[(row["algorithm"], row["seed"])].add(
                tuple(row[column] for column in invariant_columns)
            )
    if len(scaling) != 20 or any(len(values) != 1 for values in scaling.values()):
        raise ValueError("changing worker count changed scaling-run results")

    if len(rows(RESULTS / "baselines.csv")) != 3:
        raise ValueError("expected three baseline controllers")
    if len(rows(RESULTS / "best_policy.csv")) != 118:
        raise ValueError("expected a 118-parameter selected policy")
    frozen = rows(RESULTS / "frozen_final_test.csv")
    if len(frozen) != 1 or frozen[0]["scenarios"] != "1024":
        raise ValueError("expected one 1,024-scenario frozen final test")
    if not rows(RESULTS / "best_trajectory.csv"):
        raise ValueError("best-policy replay is empty")


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Date": None,
            "Creator": "neural-controller-policy-search/plot_results.py",
        },
    )
    plt.close(figure)


def grouped_summary(run_rows: list[dict[str, str]], experiment: str):
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in run_rows:
        if row["experiment"] == experiment:
            groups[row["algorithm"]].append(row)
    return groups


def plot_quality(run_rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for row_index, experiment in enumerate(("quality", "noise")):
        groups = grouped_summary(run_rows, experiment)
        algorithms = [name for name in COLORS if name in groups]
        scores = [
            [float(row["validation_score"]) for row in groups[name]]
            for name in algorithms
        ]
        success = [
            [100.0 * float(row["success_rate"]) for row in groups[name]]
            for name in algorithms
        ]
        positions = list(range(len(algorithms)))
        for column_index, values in enumerate((scores, success)):
            axis = axes[row_index][column_index]
            bars = axis.boxplot(
                values, positions=positions, widths=0.6, patch_artist=True
            )
            for patch, name in zip(bars["boxes"], algorithms):
                patch.set_facecolor(COLORS[name])
                patch.set_alpha(0.75)
            axis.set_xticks(positions, algorithms, rotation=20)
            axis.grid(axis="y", alpha=0.25)
            if column_index == 0:
                axis.set_ylabel("Validation score (lower is better)")
            else:
                axis.set_ylabel("Validation success rate (%)")
            if row_index == 0:
                axis.set_title(
                    "Fixed training scenarios"
                    if column_index == 0
                    else "Fixed: holdout success"
                )
            else:
                axis.set_title(
                    "Rotating common scenarios"
                    if column_index == 0
                    else "Rotating: holdout success"
                )
    fig.suptitle("Fixed-topology neural cart-pole policy search")
    fig.tight_layout()
    save_figure(fig, output / "quality.svg")


def plot_scaling(run_rows: list[dict[str, str]], output: Path) -> None:
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in run_rows:
        if row["experiment"] == "scaling":
            groups[(row["algorithm"], int(row["workers"]))].append(
                float(row["wall_seconds"])
            )
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    for algorithm, color in COLORS.items():
        points = sorted(
            (workers, values)
            for (name, workers), values in groups.items()
            if name == algorithm
        )
        if not points:
            continue
        worker_values = [point[0] for point in points]
        means = [mean(point[1]) for point in points]
        errors = [stdev(point[1]) if len(point[1]) > 1 else 0.0 for point in points]
        axis.errorbar(
            worker_values,
            means,
            yerr=errors,
            marker="o",
            capsize=3,
            label=algorithm,
            color=color,
        )
    axis.set_xlabel("Evaluation workers")
    axis.set_ylabel("Wall time (seconds)")
    axis.set_title("Candidate-evaluation parallel scaling")
    axis.set_xticks(sorted({workers for _, workers in groups}))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    save_figure(fig, output / "scaling.svg")


def plot_convergence(convergence_rows: list[dict[str, str]], output: Path) -> None:
    groups: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in convergence_rows:
        if row["experiment"] != "quality":
            continue
        groups[row["algorithm"]][int(row["evaluations"])].append(
            float(row["monitor_score"])
        )
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    for algorithm, by_evaluation in groups.items():
        points = sorted(by_evaluation.items())
        axis.plot(
            [point[0] for point in points],
            [mean(point[1]) for point in points],
            label=algorithm,
            color=COLORS[algorithm],
        )
    axis.set_xlabel("Candidate evaluations")
    axis.set_ylabel("Fixed monitor score (lower is better)")
    axis.set_title("Validation-monitor convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    save_figure(fig, output / "convergence.svg")


def plot_trajectory(trajectory_rows: list[dict[str, str]], output: Path) -> None:
    time = [float(row["time"]) for row in trajectory_rows]
    angle = [float(row["angle"]) * 180.0 / 3.141592653589793 for row in trajectory_rows]
    position = [float(row["position"]) for row in trajectory_rows]
    force = [float(row["force"]) for row in trajectory_rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    axes[0].plot(time, angle, color="#0072B2")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Pole angle (deg)")
    axes[1].plot(time, position, color="#009E73")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Cart position (m)")
    axes[2].plot(time, force, color="#D55E00")
    axes[2].set_ylabel("Force (N)")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Best validated controller: representative holdout rollout")
    fig.tight_layout()
    save_figure(fig, output / "trajectory.svg")


def render(output: Path) -> list[Path]:
    validate_results()
    run_rows = rows(RESULTS / "runs.csv")
    plot_quality(run_rows, output)
    plot_scaling(run_rows, output)
    plot_convergence(rows(RESULTS / "convergence.csv"), output)
    plot_trajectory(rows(RESULTS / "best_trajectory.csv"), output)
    return [output / name for name in FIGURES]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        for path in render(OUTPUT):
            print(f"wrote {path.relative_to(ROOT)}")
        return

    stale: list[Path] = []
    with tempfile.TemporaryDirectory() as temporary:
        generated = render(Path(temporary))
        for path in generated:
            checked_in = OUTPUT / path.name
            if not checked_in.is_file() or not filecmp.cmp(
                path, checked_in, shallow=False
            ):
                stale.append(checked_in)
    if stale:
        for path in stale:
            print(f"missing or stale tutorial figure: {path}")
        raise SystemExit(1)
    print("publication figures are current")


if __name__ == "__main__":
    main()
