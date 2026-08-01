#!/usr/bin/env python3
"""Render deterministic oscillator-topology diagrams from checked artifacts."""

from __future__ import annotations

import argparse
import csv
import filecmp
import itertools
import json
import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "publication"
IMAGES = ROOT / "images"
FIGURES = [
    "architecture.svg",
    "grammar-space.svg",
    "runtime-network.svg",
    "reference-motifs.svg",
    "campaign-results.svg",
    "best-traces.svg",
]
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#607D8B"


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
            "svg.hashsalt": "fcmaes-oscillator-topology-v1",
        }
    )


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def validate_artifacts() -> None:
    arms = ["random", "evolutionary", "agent"]
    manifests = {
        part: json.loads((RESULTS / part / "run.json").read_text(encoding="utf-8"))
        for part in arms
    }
    if any(manifest["schema_version"] != 2 for manifest in manifests.values()):
        raise ValueError("publication manifests do not share schema version 2")
    if any(manifest["status"] != "complete" for manifest in manifests.values()):
        raise ValueError("all three publication arms must be complete")
    if any(manifest["accepted_candidates"] != 200 for manifest in manifests.values()):
        raise ValueError("publication arms must contain 200 accepted candidates")
    protocols = [manifest["protocol"] for manifest in manifests.values()]
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("publication arms do not share one numerical protocol")
    if protocols[0]["inner_retries"] != 16 or protocols[0]["inner_evaluations"] != 12_000:
        raise ValueError("publication protocol is not the documented 16 x 12,000 budget")
    agent = manifests["agent"]
    if agent["proposal_policy"] != "external-feedback-candidate-menu-agent-circuit3-v4":
        raise ValueError("publication agent does not use the v4 unseen-menu policy")
    if agent["duplicates_or_invalid"] or agent["transport_failures"]:
        raise ValueError("publication agent contains proposal or transport failures")
    if agent["exact_rediscoveries"]["repressilator"] != 188:
        raise ValueError("publication agent repressilator rediscovery changed")
    for arm in arms:
        if len(rows(RESULTS / arm / "candidates.csv")) != 200:
            raise ValueError(f"{arm} does not have 200 publication candidates")


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={
            "Date": None,
            "Creator": "oscillator-topology-search/plot_results.py",
        },
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
    figure, axis = plt.subplots(figsize=(11.6, 4.5))
    axis.set(xlim=(0, 11.6), ylim=(0, 4.5))
    axis.axis("off")
    box(axis, 0.2, 1.75, 1.55, 0.9, "outer proposer\nsigned topology", "#E3F2FD")
    box(axis, 2.15, 1.75, 1.55, 0.9, "grammar gate\n+ dedup cache", "#FFF3E0")
    box(axis, 4.1, 1.75, 1.65, 0.9, "variable 10–18 D\nkinetic bounds", "#F3E5F5")
    box(axis, 6.15, 1.75, 1.55, 0.9, "BiteOpt\nfixed eval budget", "#EDE7F6")
    box(axis, 8.1, 2.45, 1.55, 0.8, "ReBop SSA\ntraining CRNs", "#E8F5E9")
    box(axis, 8.1, 1.05, 1.55, 0.8, "ReBop SSA\ndisjoint holdout", "#E0F2F1")
    box(axis, 10.05, 1.75, 1.35, 0.9, "archive\nscore + motifs", "#FFEBEE")
    for start, end in [
        ((1.75, 2.2), (2.15, 2.2)),
        ((3.7, 2.2), (4.1, 2.2)),
        ((5.75, 2.2), (6.15, 2.2)),
        ((7.7, 2.2), (8.1, 2.85)),
        ((7.7, 2.2), (8.1, 1.45)),
        ((9.65, 2.85), (10.05, 2.35)),
        ((9.65, 1.45), (10.05, 2.05)),
    ]:
        arrow(axis, start, end)
    axis.add_patch(
        FancyArrowPatch(
            (10.72, 2.68),
            (0.95, 2.68),
            connectionstyle="arc3,rad=0.27",
            arrowstyle="->",
            mutation_scale=12,
            color=GREY,
            linewidth=1.2,
        )
    )
    axis.text(
        5.8,
        4.1,
        "Split brain: discrete proposals outside, numerical evidence inside",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        5.8,
        0.32,
        "held-out reference encodings never enter proposer observations or archives",
        ha="center",
        color="#37474F",
    )
    save(figure, output)


def valid_topology(edges: tuple[int, ...]) -> bool:
    active = sum(value != 0 for value in edges)
    if not 2 <= active <= 6:
        return False
    edge_index = [
        (0, 0),
        (1, 1),
        (2, 2),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    ]
    return all(
        any(value and (source == gene or target == gene)
            for value, (source, target) in zip(edges, edge_index))
        for gene in range(3)
    )


def grammar_space(output: Path) -> None:
    counts = {active: 0 for active in range(2, 7)}
    valid = 0
    for edges in itertools.product(range(3), repeat=9):
        if valid_topology(edges):
            counts[sum(value != 0 for value in edges)] += 1
            valid += 1
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    axes[0].bar(list(counts), list(counts.values()), color=BLUE, width=0.72)
    axes[0].set(
        title=f"Grammar keeps {valid:,} of $3^9$ signed vectors",
        xlabel="active edges",
        ylabel="valid canonical topologies",
        xticks=list(counts),
    )
    axis = axes[1]
    axis.set(xlim=(-0.5, 9.5), ylim=(-1.2, 2.3))
    axis.axis("off")
    labels = ["A→A", "B→B", "C→C", "A→B", "A→C", "B→A", "B→C", "C→A", "C→B"]
    example = [0, 0, 0, 2, 0, 0, 2, 2, 0]
    colors = ["#ECEFF1", "#E8F5E9", "#FFEBEE"]
    for index, (label, value) in enumerate(zip(labels, example)):
        box(axis, index, 0.25, 0.72, 0.78, f"{label}\n{value}", colors[value])
    axis.text(
        4.5,
        1.75,
        "canonical slot order — 0 absent, 1 activation, 2 inhibition",
        ha="center",
        weight="bold",
    )
    axis.text(
        4.5,
        -0.62,
        "example 000200220: the three-inhibition repressilator",
        ha="center",
        color="#37474F",
    )
    figure.tight_layout()
    save(figure, output)


def runtime_network(output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    x = np.linspace(0, 100, 400)
    hill = 3.0
    strength = 50.0
    k = 20.0
    activation = strength * x**hill / (k**hill + x**hill)
    inhibition = strength * k**hill / (k**hill + x**hill)
    axes[0].plot(x, activation, label="activation", color=GREEN, lw=2)
    axes[0].plot(x, inhibition, label="inhibition", color=ORANGE, lw=2)
    axes[0].axvline(k, color=GREY, ls="--", lw=1, label="K = 20")
    axes[0].set(
        title="Runtime Hill contributions",
        xlabel="source copy count",
        ylabel="production contribution",
    )
    axes[0].legend()
    axis = axes[1]
    axis.set(xlim=(0, 5.4), ylim=(0, 4.0))
    axis.axis("off")
    for gene, y, color in [("A", 3.0, "#E3F2FD"), ("B", 2.0, "#E8F5E9"), ("C", 1.0, "#FFF3E0")]:
        box(axis, 2.0, y - 0.28, 1.2, 0.56, f"X{gene}", color)
        arrow(axis, (0.35, y), (2.0, y))
        arrow(axis, (3.2, y), (4.9, y))
        axis.text(0.1, y, "∅", va="center", fontsize=13)
        axis.text(5.08, y, "∅", va="center", fontsize=13)
    axis.text(1.2, 3.55, "basal + Σ Hill(edge)", ha="center", color=GREEN)
    axis.text(4.1, 3.55, "δᵢ Xᵢ", ha="center", color=ORANGE)
    axis.text(
        2.7,
        0.25,
        "always 3 species and 6 reactions; topology changes propensities",
        ha="center",
        weight="bold",
    )
    figure.tight_layout()
    save(figure, output)


def draw_motif(axis: plt.Axes, edges: list[tuple[int, int, int]], title: str) -> None:
    points = np.array([[0, 1.0], [-0.9, -0.65], [0.9, -0.65]])
    axis.scatter(points[:, 0], points[:, 1], s=500, c=["#E3F2FD", "#E8F5E9", "#FFF3E0"],
                 edgecolors="#455A64", zorder=4)
    for index, label in enumerate("ABC"):
        axis.text(*points[index], label, ha="center", va="center", weight="bold", zorder=5)
    for source, target, kind in edges:
        start = points[source]
        end = points[target]
        delta = end - start
        norm = np.linalg.norm(delta)
        start = start + 0.22 * delta / norm
        end = end - 0.22 * delta / norm
        patch = FancyArrowPatch(
            start,
            end,
            connectionstyle="arc3,rad=0.12",
            arrowstyle="-|>" if kind == 1 else "-[",
            mutation_scale=12,
            linewidth=1.8,
            color=GREEN if kind == 1 else ORANGE,
        )
        axis.add_patch(patch)
    axis.set(title=title, xlim=(-1.45, 1.45), ylim=(-1.15, 1.45), aspect="equal")
    axis.axis("off")


def reference_motifs(output: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 3.25))
    draw_motif(axes[0], [(0, 1, 2), (1, 2, 2), (2, 0, 2)], "repressilator")
    draw_motif(axes[1], [(0, 1, 1), (1, 2, 1), (2, 0, 2)], "Goodwin-like cycle")
    draw_motif(axes[2], [(0, 1, 1), (1, 2, 1), (2, 0, 1)], "positive cycle")
    draw_motif(axes[3], [(0, 1, 2), (1, 0, 2), (0, 2, 1)], "toggle control")
    figure.suptitle("Held-out structural references (not outer-search seeds)", weight="bold")
    figure.tight_layout()
    save(figure, output)


def campaign_results(output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.25))
    arms = ["random", "evolutionary", "agent"]
    colors = {"random": BLUE, "evolutionary": GREEN, "agent": ORANGE}
    for arm in arms:
        data = rows(RESULTS / arm / "convergence.csv")
        axes[0].step(
            [int(row["accepted"]) for row in data],
            [float(row["best_validation_score"]) for row in data],
            where="post",
            label=arm,
            color=colors[arm],
            lw=2,
        )
    axes[0].set(
        title="Best holdout score versus accepted topology",
        xlabel="accepted topologies",
        ylabel="best validation score (lower is better)",
    )
    axes[0].legend()
    best = []
    medians = []
    for arm in arms:
        manifest = json.loads((RESULTS / arm / "run.json").read_text(encoding="utf-8"))
        best.append(manifest["best_validation_score"])
        scores = sorted(
            float(row["validation_score"])
            for row in rows(RESULTS / arm / "candidates.csv")
        )
        medians.append((scores[99] + scores[100]) / 2)
    positions = np.arange(len(arms))
    axes[1].bar(positions - 0.18, best, color=[colors[arm] for arm in arms], width=0.34, label="best")
    axes[1].bar(positions + 0.18, medians, color="#B0BEC5", width=0.34, label="median")
    axes[1].set(
        title="Matched 200-candidate outcome",
        ylabel="validation score",
        xticks=positions,
        xticklabels=["random", "evolutionary", "Gemma 4\n31B Q8"],
        ylim=(0, max(medians) * 1.18),
    )
    axes[1].legend()
    axes[1].text(
        0.02,
        0.98,
        "Agent rediscovered the exact repressilator at proposal 188.",
        transform=axes[1].transAxes,
        va="top",
        color="#37474F",
    )
    figure.tight_layout()
    save(figure, output)


def best_traces(output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    for axis, arm in zip(axes, ["random", "evolutionary", "agent"]):
        data = rows(RESULTS / arm / "best_trace.csv")
        time = [float(row["time"]) for row in data]
        for gene, color in zip("ABC", [BLUE, GREEN, ORANGE]):
            axis.plot(time, [float(row[gene]) for row in data], color=color, lw=1.2, label=gene)
        manifest = json.loads((RESULTS / arm / "run.json").read_text(encoding="utf-8"))
        label = "Gemma agent" if arm == "agent" else arm
        axis.set(
            title=f"{label}: {manifest['best_topology']}  holdout score {manifest['best_validation_score']:.3f}",
            ylabel="copies",
        )
        axis.legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("model time")
    figure.suptitle("Disjoint-seed replay of each publication incumbent", weight="bold")
    figure.tight_layout()
    save(figure, output)


def render(root: Path) -> None:
    configure()
    validate_artifacts()
    functions = [
        architecture,
        grammar_space,
        runtime_network,
        reference_motifs,
        campaign_results,
        best_traces,
    ]
    for function, name in zip(functions, FIGURES):
        function(root / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        render(IMAGES)
        for name in FIGURES:
            print(IMAGES / name)
        return 0
    with tempfile.TemporaryDirectory() as temporary:
        generated = Path(temporary)
        render(generated)
        stale = [
            IMAGES / name
            for name in FIGURES
            if not (IMAGES / name).is_file()
            or not filecmp.cmp(generated / name, IMAGES / name, shallow=False)
        ]
    if stale:
        print("missing or stale oscillator-topology-search figures:")
        for path in stale:
            print(path)
        return 1
    print("oscillator-topology-search figures are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
