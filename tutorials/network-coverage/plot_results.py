#!/usr/bin/env python3
"""Render deterministic network-coverage diagrams from publication artifacts."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "publication"
IMAGES = ROOT / "images"
FIGURES = [
    "architecture.svg",
    "synthetic-network.svg",
    "coverage-kernel.svg",
    "decoder-plateau.svg",
    "oracle-certificates.svg",
    "throughput.svg",
    "so-comparison.svg",
    "pareto-comparison.svg",
    "budget-sensitivity.svg",
    "weight-sensitivity.svg",
]
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#78909C"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
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
            "svg.hashsalt": "fcmaes-network-coverage-v1",
        }
    )


def validate_artifacts() -> None:
    manifests = [
        json.loads((RESULTS / part / "run.json").read_text(encoding="utf-8"))
        for part in ["validation", "throughput", "so", "mo"]
    ]
    if any(item["schema_version"] != 1 for item in manifests):
        raise ValueError("publication manifests do not share schema version 1")
    if manifests[1]["selected_publication_instance"] != "reference-4k":
        raise ValueError("figures expect the recorded 4k throughput-gate decision")
    oracle = rows(RESULTS / "validation" / "classic_oracles.csv")
    if not all(row["verified"] == "1" for row in oracle):
        raise ValueError("an oracle artifact contains an unverified cover")
    for row in oracle:
        if float(row["ratio"]) > 2.0 + 1.0e-10:
            raise ValueError("a certificate exceeds its own factor-two contract")
        if row["exact_status"] not in {"optimal", "not-attempted"}:
            raise ValueError("exact-cover scope is not explicit")
    scalar = rows(RESULTS / "so" / "arms.csv")
    if not all(row["verified"] == "1" for row in scalar):
        raise ValueError("a scalar publication incumbent failed replay")
    optimized = [row for row in scalar if row["arm"].startswith("de-")]
    if not all(row["retained_source"] in {"seed", "optimizer"} for row in optimized):
        raise ValueError("scalar optimizer rows do not expose retained origin")
    pareto = rows(RESULTS / "mo" / "pareto.csv")
    if not pareto or min(float(row["roi"]) for row in pareto) < -1.0e-12:
        raise ValueError("invalid MODE front")
    if any(row["cost"].startswith("-") for row in pareto):
        raise ValueError("MODE artifacts contain signed zero or a negative cost")
    sensitivity = rows(RESULTS / "mo" / "budget_sensitivity.csv")
    if int(sensitivity[-1]["requested_evaluations"]) != 200_000:
        raise ValueError("high-budget MODE sensitivity campaign is missing")
    if int(sensitivity[-1]["mode_generated_not_dominated_by_greedy"]) != 0:
        raise ValueError("high-budget MODE generated a point surviving greedy")


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "network-coverage/plot_results.py"},
    )
    plt.close(figure)
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n",
        encoding="utf-8",
    )


def box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    color: str,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.06",
            facecolor=color,
            edgecolor="#455A64",
            linewidth=1.1,
        )
    )
    axis.text(x + width / 2, y + height / 2, label, ha="center", va="center")


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#455A64"},
    )


def architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.2, 4.0))
    axis.set(xlim=(0, 11.2), ylim=(0, 4.0))
    axis.axis("off")
    box(axis, 0.15, 1.48, 1.55, 0.88, "synthetic graph\nor local edge list", "#E3F2FD")
    box(axis, 2.08, 1.48, 1.55, 0.88, "[0,2) controls\nbinary integer bins", "#FFF3E0")
    box(axis, 4.02, 1.48, 1.72, 0.88, "ordinary edges +\nnative group counts", "#E8F5E9")
    box(axis, 6.13, 2.12, 1.78, 0.78, "matching / primal-dual\nexact tiny ILP", "#FFEBEE")
    box(axis, 6.13, 0.84, 1.78, 0.78, "marginal-greedy\nprefix frontier", "#E0F2F1")
    box(axis, 8.36, 2.12, 2.55, 0.78, "seeded DE retry\nclassic covers", "#F3E5F5")
    box(axis, 8.36, 0.84, 2.55, 0.78, "integer-aware MODE\ncost / coverage", "#EDE7F6")
    for start, end in [
        ((1.70, 1.92), (2.08, 1.92)),
        ((3.63, 1.92), (4.02, 1.92)),
        ((5.74, 1.92), (6.13, 2.50)),
        ((5.74, 1.92), (6.13, 1.23)),
        ((7.91, 2.50), (8.36, 2.50)),
        ((7.91, 1.23), (8.36, 1.23)),
    ]:
        arrow(axis, start, end)
    axis.text(
        5.6,
        3.55,
        "One replayable kernel; distinct certificates and optimization questions",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        5.6,
        0.25,
        "all published masks are decoded and scored again before writing artifacts",
        ha="center",
        color="#37474F",
    )
    save(figure, output)


def fixture_rows(name: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    base = ROOT / "instances" / name
    return rows(base / "nodes.csv"), rows(base / "edges.csv")


def synthetic_network(output: Path) -> None:
    nodes, edges = fixture_rows("small")
    count = len(nodes)
    blocks = np.array([int(row["block"]) for row in nodes])
    angles = np.zeros(count)
    radii = np.zeros(count)
    for block in sorted(set(blocks)):
        members = np.where(blocks == block)[0]
        center_angle = 2 * math.pi * block / (blocks.max() + 1)
        for position, node in enumerate(members):
            angles[node] = center_angle + 0.24 * math.sin(position * 2.4)
            radii[node] = 1.7 + 0.055 * position
    xy = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    segments = [
        (xy[int(edge["source"])], xy[int(edge["target"])]) for edge in edges
    ]
    figure, axis = plt.subplots(figsize=(8.4, 6.0))
    axis.add_collection(
        LineCollection(segments, colors="#B0BEC5", linewidths=0.45, alpha=0.42)
    )
    scatter = axis.scatter(
        xy[:, 0],
        xy[:, 1],
        c=blocks,
        cmap="tab10",
        s=32,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    axis.set(
        title="Checked-in synthetic small fixture: 60 nodes, 240 ordinary edges, 12 groups",
        aspect="equal",
    )
    axis.axis("off")
    axis.legend(*scatter.legend_elements(), title="block", loc="lower right", ncol=2)
    axis.text(
        0.02,
        0.02,
        "layout is illustrative; optimization uses no coordinates",
        transform=axis.transAxes,
        color="#455A64",
    )
    save(figure, output)


def coverage_kernel(output: Path) -> None:
    selected = {0, 3}
    nodes = np.array([[0, 0], [1.3, 0.9], [2.6, 0], [1.3, -1.1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axis = axes[0]
    for u, v in edges:
        color = GREEN if u in selected or v in selected else "#B0BEC5"
        axis.plot(*zip(nodes[u], nodes[v]), color=color, lw=2.2)
    axis.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=[ORANGE if i in selected else "white" for i in range(4)],
        edgecolors="#37474F",
        s=115,
        zorder=3,
    )
    for i, (x, y) in enumerate(nodes):
        axis.text(x, y, str(i), ha="center", va="center", zorder=4)
    axis.set(title="Ordinary edge: covered if either endpoint is selected", aspect="equal")
    axis.axis("off")
    axis = axes[1]
    s = np.arange(2, 301)
    axis.plot(s, s ** -0.5, color=PURPLE, lw=2)
    axis.set(
        title="Native group pair weight",
        xlabel="group size s",
        ylabel=r"$g(s)=s^{-1/2}$",
    )
    axis.text(
        0.48,
        0.70,
        r"$g(s)\,[\binom{s}{2}-\binom{s-k}{2}]$"
        "\nexact weighted clique score\nwithout clique storage",
        transform=axis.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#B0BEC5"},
        ha="center",
    )
    save(figure, output)


def decoder_plateau(output: Path) -> None:
    x = np.linspace(0, 1.999999999999, 1001)
    y = (x >= 1.0).astype(float)
    figure, axis = plt.subplots(figsize=(8.3, 3.9))
    axis.step(x, y, where="post", color=BLUE, lw=2.4)
    axis.axvline(1.0, color=ORANGE, linestyle="--", lw=1.4)
    axis.set(
        title="One coordinate has two reachable integer bins and one transition",
        xlabel="optimizer coordinate",
        ylabel="physical node state",
        yticks=[0, 1],
        yticklabels=["not selected", "selected"],
        xlim=(0, 2),
        ylim=(-0.18, 1.18),
    )
    axis.text(0.5, 0.15, "bin 0", ha="center", color=BLUE)
    axis.text(1.5, 0.85, "bin 1", ha="center", color=BLUE)
    save(figure, output)


def oracle_certificates(output: Path) -> None:
    data = [
        row
        for row in rows(RESULTS / "validation" / "classic_oracles.csv")
        if row["instance"] == "reference-4k"
    ]
    labels = ["cardinality\nmatching", "weighted\nprimal-dual"]
    ratios = [float(row["ratio"]) for row in data]
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    bars = axis.bar(labels, ratios, color=[BLUE, GREEN], width=0.58)
    axis.axhline(2.0, color=ORANGE, linestyle="--", label="certified factor-2 ceiling")
    axis.set(
        title="Independent classic-cover certificate ratios on reference-4k",
        ylabel="verified cover value / certified lower bound",
        ylim=(0, 2.15),
    )
    for bar, value in zip(bars, ratios):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.3f}", ha="center")
    axis.legend()
    save(figure, output)


def throughput(output: Path) -> None:
    data = rows(RESULTS / "throughput" / "throughput.csv")
    labels = [
        f"{row['instance'].replace('reference-', '')}\nworkers={row['workers']}"
        for row in data
    ]
    values = [float(row["candidates_per_second"]) for row in data]
    figure, axis = plt.subplots(figsize=(8.8, 4.3))
    bars = axis.bar(labels, values, color=[GREY, BLUE, GREY, GREEN])
    axis.axhline(20_000, color=ORANGE, linestyle="--", label="4k scale gate")
    axis.set(
        title="Coverage-kernel throughput selected the 4,000-node publication fixture",
        ylabel="candidate evaluations / s",
        yscale="log",
    )
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value * 1.09, f"{value:,.0f}", ha="center")
    axis.legend()
    save(figure, output)


def so_comparison(output: Path) -> None:
    data = rows(RESULTS / "so" / "arms.csv")
    labels = [row["arm"].replace("-", "\n", 1) for row in data]
    ratios = [float(row["ratio_to_bound"]) for row in data]
    figure, axis = plt.subplots(figsize=(9.0, 4.3))
    colors = [BLUE, GREEN, PURPLE, ORANGE]
    bars = axis.bar(labels, ratios, color=colors)
    for bar, row in zip(bars, data):
        if row["retained_source"] == "seed":
            bar.set_hatch("//")
    axis.axhline(1.0, color="#37474F", lw=1.0)
    axis.axhline(2.0, color="#37474F", linestyle="--", lw=1.0)
    axis.set(
        title="Finite DE retained the certified seeds; hatched bars are fallbacks",
        ylabel="cover value / corresponding lower bound",
        ylim=(0, 2.1),
    )
    for bar, value in zip(bars, ratios):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.3f}", ha="center")
    save(figure, output)


def pareto_comparison(output: Path) -> None:
    mode = rows(RESULTS / "mo" / "pareto.csv")
    greedy = rows(RESULTS / "mo" / "greedy_front.csv")
    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    axis.plot(
        [float(row["cost"]) for row in greedy],
        [float(row["roi"]) for row in greedy],
        color=GREEN,
        lw=2.0,
        label="marginal-greedy prefix",
    )
    generated = [row for row in mode if row["source"] == "mode-generated"]
    retained = [row for row in mode if row["source"] != "mode-generated"]
    axis.scatter(
        [float(row["cost"]) for row in generated],
        [float(row["roi"]) for row in generated],
        s=19,
        alpha=0.65,
        color=PURPLE,
        label="MODE-generated population front",
    )
    axis.scatter(
        [float(row["cost"]) for row in retained],
        [float(row["roi"]) for row in retained],
        s=42,
        marker="x",
        color=ORANGE,
        label="MODE-retained supplied initial point",
    )
    axis.set(
        title="Specialist marginal greedy dominates the finite stochastic front",
        xlabel="selected-node cost",
        ylabel="coverage ROI",
        xlim=(0, None),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right")
    save(figure, output)


def budget_sensitivity(output: Path) -> None:
    data = rows(RESULTS / "mo" / "budget_sensitivity.csv")
    labels = [
        f"{int(row['requested_evaluations']):,}\n({row['campaign']})" for row in data
    ]
    generated = [int(row["mode_generated_points"]) for row in data]
    surviving = [
        int(row["mode_generated_not_dominated_by_greedy"]) for row in data
    ]
    figure, axis = plt.subplots(figsize=(7.8, 4.3))
    x = np.arange(len(data))
    bars = axis.bar(x, generated, color=PURPLE, width=0.56, label="MODE-generated front")
    axis.scatter(
        x,
        surviving,
        s=90,
        marker="x",
        linewidths=2,
        color=ORANGE,
        label="generated points not dominated by greedy",
        zorder=3,
    )
    for bar, value in zip(bars, generated):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            str(value),
            ha="center",
        )
    axis.set(
        title="A 24× MODE budget still contributes no point beyond marginal greedy",
        xlabel="Requested candidate evaluations",
        ylabel="Population-front points",
        xticks=x,
        xticklabels=labels,
        ylim=(0, max(generated + [1]) * 1.16),
    )
    axis.legend()
    save(figure, output)


def weight_sensitivity(output: Path) -> None:
    data = rows(RESULTS / "validation" / "group_weight_sensitivity.csv")
    figure, axis = plt.subplots(figsize=(7.8, 4.3))
    for exponent, color in [("0.0", BLUE), ("0.5", GREEN), ("1.0", ORANGE)]:
        subset = [row for row in data if row["exponent"] == exponent]
        axis.plot(
            [int(row["selected"]) for row in subset],
            [float(row["roi"]) for row in subset],
            marker="o",
            color=color,
            label=f"exponent {exponent}",
        )
    axis.set(
        title="Group-size weighting changes ROI but not the selected tiny masks",
        xlabel="selected nodes in fixed greedy prefixes",
        ylabel="replayed ROI",
        ylim=(0, 1),
    )
    axis.legend()
    save(figure, output)


def render(directory: Path) -> None:
    configure()
    validate_artifacts()
    functions = [
        architecture,
        synthetic_network,
        coverage_kernel,
        decoder_plateau,
        oracle_certificates,
        throughput,
        so_comparison,
        pareto_comparison,
        budget_sensitivity,
        weight_sensitivity,
    ]
    for function, name in zip(functions, FIGURES):
        function(directory / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.check:
        render(IMAGES)
        for name in FIGURES:
            print(IMAGES / name)
        return
    with tempfile.TemporaryDirectory() as temporary:
        generated = Path(temporary)
        render(generated)
        stale = [
            str(IMAGES / name)
            for name in FIGURES
            if not (IMAGES / name).exists()
            or not filecmp.cmp(generated / name, IMAGES / name, shallow=False)
        ]
    if stale:
        raise SystemExit("missing or stale network-coverage figures:\n" + "\n".join(stale))
    print("network-coverage figures are current")


if __name__ == "__main__":
    main()
