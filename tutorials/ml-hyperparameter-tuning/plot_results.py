#!/usr/bin/env python3
"""Render deterministic validation and budget figures for the HPO tutorial."""

from __future__ import annotations

import argparse
import csv
import filecmp
import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "quick" / "budget-sweep" / "budget_summary.csv"
LATENCY = ROOT / "results" / "quick" / "benchmark" / "latency.csv"
SCALING = ROOT / "results" / "quick" / "benchmark" / "parallel_scaling.csv"
OUTPUT = ROOT / "images" / "quick-budget-sweep"
COLORS = {
    "fcmaes-biteopt": "#0072B2",
    "random": "#D55E00",
    "lhs": "#009E73",
}
LABELS = {
    "fcmaes-biteopt": "fcmaes BiteOpt retry",
    "random": "uniform random",
    "lhs": "Latin hypercube",
}


def load_rows() -> list[dict[str, str]]:
    with SUMMARY.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def configure() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "svg.hashsalt": "fcmaes-rust-hpo-v1",
        }
    )


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "ml-hyperparameter-tuning/plot_results.py"},
    )
    plt.close(figure)


def method_comparison(rows: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))
    for method in COLORS:
        equal_calls = [
            row
            for row in rows
            if row["comparison"] == "equal-calls" and row["method"] == method
        ]
        calibrated_wall = [
            row
            for row in rows
            if row["comparison"] == "calibrated-wall" and row["method"] == method
        ]
        budgets = [int(row["actual_evaluations"]) for row in equal_calls]
        selection = [float(row["selection_log_loss"]) for row in equal_calls]
        wall = [float(row["wall_seconds"]) for row in equal_calls]
        axes[0].plot(
            budgets,
            selection,
            marker="o",
            linewidth=1.7,
            color=COLORS[method],
            label=LABELS[method],
        )
        axes[1].plot(
            [float(row["wall_seconds"]) for row in calibrated_wall],
            [float(row["selection_log_loss"]) for row in calibrated_wall],
            marker="o",
            linewidth=1.7,
            color=COLORS[method],
            label=LABELS[method],
        )
        axes[2].plot(
            budgets,
            wall,
            marker="o",
            linewidth=1.7,
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0].set(
        xlabel="Candidate objective calls",
        ylabel="Disjoint selection log-loss",
        title="Selection quality at equal call budgets",
        xscale="log",
    )
    axes[1].set(
        xlabel="Achieved wall time [s]",
        ylabel="Disjoint selection log-loss",
        title="One-pilot calibrated wall budgets",
    )
    axes[2].set(
        xlabel="Candidate objective calls",
        ylabel="Wall time [s]",
        title="Cost differs as configurations change",
        xscale="log",
    )
    axes[0].legend()
    figure.suptitle("Smoke-budget method comparison (functional evidence only)", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def validation_optimism(rows: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for method in COLORS:
        selected = [
            row
            for row in rows
            if row["comparison"] == "equal-calls" and row["method"] == method
        ]
        budgets = [int(row["actual_evaluations"]) for row in selected]
        tuning = [float(row["tuning_log_loss"]) for row in selected]
        selection = [float(row["selection_log_loss"]) for row in selected]
        gap = [validation - train for train, validation in zip(tuning, selection, strict=True)]
        axes[0].plot(
            budgets,
            tuning,
            marker="o",
            linestyle="--",
            linewidth=1.2,
            color=COLORS[method],
            alpha=0.65,
        )
        axes[0].plot(
            budgets,
            selection,
            marker="s",
            linewidth=1.7,
            color=COLORS[method],
            label=LABELS[method],
        )
        axes[1].plot(
            budgets,
            gap,
            marker="o",
            linewidth=1.7,
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0].set(
        xlabel="Candidate objective calls",
        ylabel="Log-loss",
        title="Dashed: fixed-fold tuning; solid: selection",
        xscale="log",
    )
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set(
        xlabel="Candidate objective calls",
        ylabel="Selection − tuning log-loss",
        title="Validation optimism",
        xscale="log",
    )
    axes[0].legend()
    figure.suptitle("Why HPO needs a disjoint selection stage", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def latency_validation(output: Path) -> None:
    with LATENCY.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    work = [float(row["structural_cost"]) for row in rows]
    latency = [float(row["microseconds_per_row"]) for row in rows]
    model_bytes = [float(row["model_bytes"]) for row in rows]
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    artist = axis.scatter(
        work,
        latency,
        c=model_bytes,
        cmap="viridis",
        s=55,
        edgecolors="#263238",
        linewidths=0.4,
    )
    figure.colorbar(artist, ax=axis, label="Serialized model size [bytes]")
    axis.set(
        xlabel="Tree-depth inference-work proxy",
        ylabel="Isolated prediction latency [µs/row]",
        title="Validate the deterministic proxy outside optimization",
    )
    figure.tight_layout()
    save(figure, output)


def parallel_scaling(output: Path) -> None:
    with SCALING.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    workers = [int(row["workers"]) for row in rows]
    throughput = [float(row["candidates_per_second"]) for row in rows]
    rss = [
        float(row["peak_rss_kib"]) / 1024.0
        if row["peak_rss_kib"]
        else float("nan")
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    axis.plot(workers, throughput, marker="o", linewidth=1.8, color="#0072B2")
    axis.set(
        xlabel="fcmaes candidate workers",
        ylabel="Candidate evaluations per second",
        title="Outer parallelism scaling on the smoke workload",
        xticks=workers,
    )
    twin = axis.twinx()
    twin.plot(workers, rss, marker="s", linewidth=1.4, color="#D55E00")
    twin.set_ylabel("Process peak RSS [MiB]", color="#D55E00")
    figure.tight_layout()
    save(figure, output)


def encoding(output: Path) -> None:
    coordinates = [index / 200.0 for index in range(201)]
    trees = [round(8.0 * math.exp(value * math.log(256.0 / 8.0))) for value in coordinates]
    leaves = [round(math.exp(value * math.log(64.0))) for value in coordinates]
    depth = [round(2.0 + value * 22.0) for value in coordinates]
    normalized_trees = [(value - 8.0) / (256.0 - 8.0) for value in trees]
    normalized_depth = [(value - 2.0) / 22.0 for value in depth]

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    axes[0].plot(coordinates, trees, linewidth=1.8, label="trees: 8–256")
    axes[0].plot(coordinates, leaves, linewidth=1.8, label="minimum leaf: 1–64")
    axes[0].set(
        xlabel="Normalized optimizer coordinate",
        ylabel="Decoded integer",
        yscale="log",
        title="Logarithmic integer dimensions",
    )
    axes[0].legend()
    axes[1].plot(
        coordinates,
        normalized_trees,
        linewidth=1.8,
        label="logarithmic trees",
    )
    axes[1].plot(
        coordinates,
        normalized_depth,
        linewidth=1.8,
        label="linear maximum depth",
    )
    axes[1].plot(
        coordinates,
        coordinates,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="continuous identity",
    )
    axes[1].set(
        xlabel="Normalized optimizer coordinate",
        ylabel="Fraction through decoded domain",
        title="Encoding changes search resolution",
    )
    axes[1].legend()
    figure.suptitle("Mixed-space decoding is part of the objective", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def render(destination: Path) -> list[Path]:
    configure()
    rows = load_rows()
    generated = [
        destination / "method-comparison.svg",
        destination / "validation-optimism.svg",
        destination / "latency-validation.svg",
        destination / "parallel-scaling.svg",
        destination / "encoding.svg",
    ]
    method_comparison(rows, generated[0])
    validation_optimism(rows, generated[1])
    latency_validation(generated[2])
    parallel_scaling(generated[3])
    encoding(generated[4])
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        for path in render(OUTPUT):
            print(path)
        return 0
    with tempfile.TemporaryDirectory() as temporary:
        generated = render(Path(temporary))
        stale = [
            OUTPUT / path.name
            for path in generated
            if not (OUTPUT / path.name).is_file()
            or not filecmp.cmp(path, OUTPUT / path.name, shallow=False)
        ]
    if stale:
        print("missing or stale HPO tutorial figures:")
        for path in stale:
            print(path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
