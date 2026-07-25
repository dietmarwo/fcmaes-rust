#!/usr/bin/env python3
"""Render deterministic HPO figures and rebuild publication summaries."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import math
import statistics
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "quick" / "budget-sweep" / "budget_summary.csv"
DESCRIPTOR_STUDY = ROOT / "results" / "publication" / "descriptor-study"
DESCRIPTOR_SUMMARY = DESCRIPTOR_STUDY / "descriptor-summary.csv"
QD_PUBLICATION = ROOT / "results" / "publication"
QD_SUMMARY = QD_PUBLICATION / "qd-summary.csv"
QD_VALIDATION_SUMMARY = QD_PUBLICATION / "qd-validation-summary.csv"
QD_RETENTION_BY_GRID = QD_PUBLICATION / "qd-retention-by-grid.csv"
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


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_scale * right_scale)


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def occupied_cells(
    rows: list[dict[str, str]],
    x_name: str,
    y_name: str,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    side: int = 20,
) -> int:
    occupied: set[tuple[int, int]] = set()
    for row in rows:
        x = float(row[x_name])
        y = float(row[y_name])
        if not (x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1]):
            continue
        grid_x = min(side - 1, int((x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) * side))
        grid_y = min(side - 1, int((y - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) * side))
        occupied.add((grid_x, grid_y))
    return len(occupied)


def descriptor_summary_text() -> str:
    rows: list[dict[str, str]] = []
    for method in ("random", "lhs"):
        with (DESCRIPTOR_STUDY / method / "candidates.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            rows.extend(csv.DictReader(source))
    feasible = [row for row in rows if row["feasible"] == "1"]
    columns = {
        name: [float(row[name]) for row in feasible]
        for name in (
            "predicted_positive_rate",
            "error_ratio",
            "precision",
            "sharpness",
            "ece",
        )
    }
    ppr_bounds = (
        min(columns["predicted_positive_rate"]),
        max(columns["predicted_positive_rate"]),
    )
    error_bounds = (min(columns["error_ratio"]), max(columns["error_ratio"]))
    header = [
        "candidate_calls",
        "feasible_candidates",
        "precision_min",
        "precision_max",
        "sharpness_min",
        "sharpness_max",
        "error_ratio_min",
        "error_ratio_max",
        "spearman_ppr_error_ratio",
        "spearman_precision_sharpness",
        "spearman_ece_sharpness",
        "occupied_ppr_error_original_bounds",
        "occupied_ppr_error_observed_bounds",
        "occupied_precision_sharpness_frozen_bounds",
    ]
    values = [
        len(rows),
        len(feasible),
        min(columns["precision"]),
        max(columns["precision"]),
        min(columns["sharpness"]),
        max(columns["sharpness"]),
        error_bounds[0],
        error_bounds[1],
        spearman(columns["predicted_positive_rate"], columns["error_ratio"]),
        spearman(columns["precision"], columns["sharpness"]),
        spearman(columns["ece"], columns["sharpness"]),
        occupied_cells(
            feasible,
            "predicted_positive_rate",
            "error_ratio",
            (0.0, 0.5),
            (-3.0, 3.0),
        ),
        occupied_cells(
            feasible,
            "predicted_positive_rate",
            "error_ratio",
            ppr_bounds,
            error_bounds,
        ),
        occupied_cells(
            feasible,
            "precision",
            "sharpness",
            (0.24, 0.52),
            (0.10, 0.45),
        ),
    ]
    return ",".join(header) + "\n" + ",".join(str(value) for value in values) + "\n"


def publication_qd_rows() -> list[dict[str, object]]:
    rows = []
    for seed in (42, 43, 44):
        path = QD_PUBLICATION / f"qd-seed-{seed}" / "run.json"
        with path.open(encoding="utf-8") as source:
            manifest = json.load(source)
        qd = manifest["qd"]
        occupied = int(qd["occupied"])
        capacity = int(qd["capacity"])
        retained = int(qd["retained_niches"])
        rows.append(
            {
                "seed": seed,
                "evaluations": int(manifest["actual_evaluations"]),
                "elapsed_seconds": float(manifest["elapsed_seconds"]),
                "occupied": occupied,
                "capacity": capacity,
                "coverage": occupied / capacity,
                "distinct_configurations": int(qd["distinct_configurations"]),
                "retained_niches": retained,
                "retention": retained / occupied,
                "invalid_evaluations": int(qd["invalid_evaluations"]),
                "clipped_descriptors": int(qd["clipped_descriptors"]),
                "decision": manifest["qd_decision"],
            }
        )
    return rows


def publication_qd_summary_text() -> str:
    rows = publication_qd_rows()
    header = list(rows[0])
    output = [",".join(header)]
    output.extend(",".join(str(row[name]) for name in header) for row in rows)
    return "\n".join(output) + "\n"


def qd_archive_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in (42, 43, 44):
        path = QD_PUBLICATION / f"qd-seed-{seed}" / "qd_archive.csv"
        with path.open(newline="", encoding="utf-8") as source:
            rows.extend(csv.DictReader(source))
    return rows


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def qd_validation_summary_text() -> str:
    rows = qd_archive_rows()
    output = ["axis,valid_elites,cell_width,median_abs_shift,p90_abs_shift"]
    for axis, lower, upper in (
        ("precision", 0.24, 0.52),
        ("sharpness", 0.10, 0.45),
    ):
        shifts = []
        for row in rows:
            training = float(row[f"descriptor_{axis}_train"])
            validation = float(row[f"descriptor_{axis}_validation"])
            if math.isfinite(training) and math.isfinite(validation):
                shifts.append(abs(validation - training))
        output.append(
            ",".join(
                str(value)
                for value in (
                    axis,
                    len(shifts),
                    (upper - lower) / 20,
                    statistics.median(shifts),
                    quantile(shifts, 0.9),
                )
            )
        )
    return "\n".join(output) + "\n"


def descriptor_niche(row: dict[str, str], side: int, suffix: str) -> int | None:
    coordinates = []
    for axis, lower, upper in (
        ("precision", 0.24, 0.52),
        ("sharpness", 0.10, 0.45),
    ):
        value = float(row[f"descriptor_{axis}_{suffix}"])
        if not math.isfinite(value) or not lower <= value <= upper:
            return None
        coordinates.append(min(side - 1, int((value - lower) / (upper - lower) * side)))
    return coordinates[1] * side + coordinates[0]


def qd_retention_by_grid_text() -> str:
    rows = qd_archive_rows()
    output = ["side,retained_niches,occupied_elites,retention"]
    for side in (20, 10, 5, 4):
        retained = sum(
            row["selection_feasible"] == "1"
            and descriptor_niche(row, side, "train") is not None
            and descriptor_niche(row, side, "train")
            == descriptor_niche(row, side, "validation")
            for row in rows
        )
        output.append(f"{side},{retained},{len(rows)},{retained / len(rows)}")
    return "\n".join(output) + "\n"


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
    summaries = {
        DESCRIPTOR_SUMMARY: descriptor_summary_text(),
        QD_SUMMARY: publication_qd_summary_text(),
        QD_VALIDATION_SUMMARY: qd_validation_summary_text(),
        QD_RETENTION_BY_GRID: qd_retention_by_grid_text(),
    }
    if arguments.write:
        for path, content in summaries.items():
            path.write_text(content, encoding="utf-8")
            print(path)
        for path in render(OUTPUT):
            print(path)
        return 0
    stale_summaries = [
        path
        for path, content in summaries.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if stale_summaries:
        print("missing or stale HPO summaries:")
        for path in stale_summaries:
            print(path)
        return 1
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
