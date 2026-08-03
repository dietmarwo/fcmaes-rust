#!/usr/bin/env python3
"""Render deterministic SVGs from a CMA-ES comparison bundle."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
COLORS = {"fcmaes": "#2563eb", "cmaes": "#e76f51"}
plt.rcParams["svg.hashsalt"] = "fcmaes-cmaes-implementation-v1"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def architecture(path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    descriptions = [
        ("A", "one CMA-ES", "serial objective", "1 active core"),
        ("B", "one CMA-ES", "parallel population", "at most λ useful cores"),
        ("C", "N CMA-ES runs", "serial objectives", "N independent workers"),
    ]
    for axis, (arm, optimizer, evaluation, utilization) in zip(axes, descriptions):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.add_patch(
            FancyBboxPatch(
                (0.08, 0.18),
                0.84,
                0.64,
                boxstyle="round,pad=0.03",
                facecolor="#f8fafc",
                edgecolor="#334155",
                linewidth=1.5,
            )
        )
        axis.text(0.5, 0.72, f"Arm {arm}", ha="center", weight="bold", fontsize=13)
        axis.text(0.5, 0.53, optimizer, ha="center", fontsize=11)
        axis.text(0.5, 0.39, evaluation, ha="center", fontsize=10)
        axis.text(0.5, 0.25, utilization, ha="center", fontsize=9, color="#475569")
    fig.suptitle("Three controlled active CMA-ES diagnostic arms", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def paired_rows(
    data: list[dict[str, str]], deadline: int
) -> list[tuple[dict[str, str], dict[str, str]]]:
    grouped: dict[tuple[str, int, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in data:
        if int(row["deadline_ms"]) != deadline:
            continue
        key = (
            row["problem"],
            int(row["injected_cost_ns"]),
            row["arm"],
            int(row["seed"]),
        )
        grouped[key][row["library"]] = row
    return [
        (libraries["fcmaes"], libraries["cmaes"])
        for libraries in grouped.values()
        if set(libraries) == {"fcmaes", "cmaes"}
    ]


def smoke_throughput(data: list[dict[str, str]], path: Path) -> None:
    zero = [row for row in data if int(row["injected_cost_ns"]) == 0]
    deadline = max(int(row["deadline_ms"]) for row in zero)
    zero = [row for row in zero if int(row["deadline_ms"]) == deadline]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in zero:
        wall = float(row["wall_seconds"])
        grouped[(row["arm"], row["library"])].append(float(row["evaluations"]) / wall)
    arms = [arm for arm in "abc" if any(key[0] == arm for key in grouped)]
    x = list(range(len(arms)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    for offset, library in [(-width / 2, "fcmaes"), (width / 2, "cmaes")]:
        values = [statistics.median(grouped[(arm, library)]) / 1e6 for arm in arms]
        axis.bar(
            [position + offset for position in x],
            values,
            width,
            label="fcmaes-core" if library == "fcmaes" else "cmaes 0.2.2",
            color=COLORS[library],
        )
    axis.set_xticks(x, [f"Arm {arm.upper()}" for arm in arms])
    axis.set_ylabel("median objective calls / second (millions)")
    axis.set_title(f"Smoke harness throughput, zero injected cost, {deadline} ms")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def publication_throughput(data: list[dict[str, str]], path: Path) -> None:
    deadline = max(int(row["deadline_ms"]) for row in data)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for fcmaes, cmaes in paired_rows(data, deadline):
        fc_rate = float(fcmaes["evaluations"]) / float(fcmaes["wall_seconds"])
        cm_rate = float(cmaes["evaluations"]) / float(cmaes["wall_seconds"])
        if fc_rate > 0.0 and cm_rate > 0.0:
            grouped[(fcmaes["arm"], int(fcmaes["injected_cost_ns"]))].append(
                fc_rate / cm_rate
            )

    arms = [arm for arm in "abc" if any(key[0] == arm for key in grouped)]
    costs = sorted({key[1] for key in grouped})
    x = list(range(len(arms)))
    width = 0.76 / max(1, len(costs))
    palette = ["#2563eb", "#2a9d8f", "#e76f51"]
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    for index, cost in enumerate(costs):
        offset = (index - (len(costs) - 1) / 2) * width
        values = [statistics.median(grouped[(arm, cost)]) for arm in arms]
        label = "0 ns" if cost == 0 else f"{cost / 1_000:g} µs"
        axis.bar(
            [position + offset for position in x],
            values,
            width,
            color=palette[index % len(palette)],
            label=label,
        )
    axis.axhline(1.0, color="#334155", linewidth=1.2, linestyle="--")
    axis.set_yscale("log")
    axis.set_xticks(x, [f"Arm {arm.upper()}" for arm in arms])
    axis.set_ylabel("median paired eval/s ratio (fcmaes-core / cmaes)")
    axis.set_title(
        f"Diagnostic active throughput, {deadline / 1_000:g} s endpoint"
    )
    axis.grid(axis="y", alpha=0.25, which="both")
    axis.legend(title="Minimum objective cost")
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def publication_quality(data: list[dict[str, str]], path: Path) -> None:
    deadline = max(int(row["deadline_ms"]) for row in data)
    outcomes: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    for fcmaes, cmaes in paired_rows(data, deadline):
        fc_best = float(fcmaes["best"])
        cm_best = float(cmaes["best"])
        scale = max(1.0, abs(fc_best), abs(cm_best))
        key = (fcmaes["arm"], int(fcmaes["injected_cost_ns"]))
        if math.isclose(fc_best, cm_best, rel_tol=1e-10, abs_tol=1e-10 * scale):
            outcomes[key][2] += 1
        elif fc_best < cm_best:
            outcomes[key][0] += 1
        else:
            outcomes[key][1] += 1

    keys = sorted(outcomes, key=lambda key: (key[0], key[1]))
    labels = [
        f"{arm.upper()} / " + ("0 ns" if cost == 0 else f"{cost / 1_000:g} µs")
        for arm, cost in keys
    ]
    totals = [sum(outcomes[key]) for key in keys]
    values = [
        [100.0 * outcomes[key][part] / totals[index] for index, key in enumerate(keys)]
        for part in range(3)
    ]
    colors = [COLORS["fcmaes"], COLORS["cmaes"], "#94a3b8"]
    names = ["fcmaes-core wins", "cmaes wins", "tie"]
    fig, axis = plt.subplots(figsize=(10.2, 5.0))
    bottom = [0.0] * len(keys)
    for name, color, segment in zip(names, colors, values):
        axis.bar(labels, segment, bottom=bottom, color=color, label=name)
        bottom = [base + value for base, value in zip(bottom, segment)]
    for index, total in enumerate(totals):
        axis.text(index, 101.5, f"n={total}", ha="center", va="bottom", fontsize=8)
    axis.set_ylim(0, 108)
    axis.set_ylabel("paired outcomes (%)")
    axis.set_title(
        f"Diagnostic final-objective outcomes, {deadline / 1_000:g} s endpoint",
        pad=42,
    )
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def render(result_dir: Path, destination: Path, profile: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    data = rows(result_dir / "paired.csv")
    architecture(destination / "architecture.svg")
    if profile == "smoke":
        smoke_throughput(data, destination / "smoke-throughput.svg")
    else:
        publication_throughput(data, destination / "publication-throughput.svg")
        publication_quality(data, destination / "publication-quality.svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results" / "harness-smoke"
    )
    parser.add_argument("--profile", choices=["smoke", "publication"], default="smoke")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "images"
    if not args.check:
        render(args.results, target, args.profile)
        return
    with tempfile.TemporaryDirectory() as temporary:
        rendered = Path(temporary)
        render(args.results, rendered, args.profile)
        stale = []
        for generated in sorted(rendered.glob("*.svg")):
            checked = target / generated.name
            if not checked.exists() or generated.read_bytes() != checked.read_bytes():
                stale.append(checked)
        if stale:
            for path in stale:
                print(f"missing or stale: {path}")
            raise SystemExit(1)
    print("cmaes-implementation figures are current")


if __name__ == "__main__":
    main()
