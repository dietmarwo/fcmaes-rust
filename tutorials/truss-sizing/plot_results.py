#!/usr/bin/env python3
"""Render deterministic truss-sizing diagrams from Rust artifacts."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "publication"
IMAGES = ROOT / "images"
FIGURES = [
    "architecture.svg",
    "ground-structure.svg",
    "triangular-oracle.svg",
    "failure-contract.svg",
    "so-comparison.svg",
    "selected-truss.svg",
    "descriptor-pilot.svg",
    "mo-pareto.svg",
    "condition-sensitivity.svg",
]
COLORS = {
    "seed": "#999999",
    "cma-seeded": "#0072B2",
    "de-seeded": "#009E73",
    "bite-seeded": "#D55E00",
    "bite-unseeded": "#CC79A7",
}


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
            "svg.hashsalt": "fcmaes-truss-sizing-v1",
        }
    )


def validate_artifacts() -> None:
    manifests = [
        json.loads((RESULTS / part / "run.json").read_text(encoding="utf-8"))
        for part in ["validation", "so", "pilot", "qd", "mo"]
    ]
    if any(item["schema_version"] != 1 for item in manifests):
        raise ValueError("publication manifests do not share schema version 1")
    oracle = rows(RESULTS / "validation" / "oracle.csv")
    if max(float(row["absolute_error"]) for row in oracle[:3]) > 1.0e-6:
        raise ValueError("triangular force oracle exceeds tolerance")
    if float(oracle[-1]["absolute_error"]) > 1.0e-10:
        raise ValueError("virtual-work displacement oracle exceeds tolerance")
    so_rows = rows(RESULTS / "so" / "arms.csv")
    if so_rows[0]["optimizer"] != "seed":
        raise ValueError("scalar artifacts do not expose the seed baseline")
    if any(row["metrics_available"] not in {"0", "1"} for row in so_rows):
        raise ValueError("scalar artifacts do not distinguish unavailable mechanics")
    if not all(row["feasible"] == "1" for row in so_rows[1:4]):
        raise ValueError("a seeded scalar arm did not retain a feasible design")
    pilot = manifests[2]
    qd = manifests[3]
    if pilot["qd"]["decision"] != "rejected":
        raise ValueError("figures expect the measured rejected descriptor gate")
    if pilot["qd"]["pilot_protocol_revision"] != 2:
        raise ValueError("figures expect descriptor-pilot protocol revision 2")
    components = {
        item["name"]: item for item in pilot["qd"]["generator_mixture"]["components"]
    }
    if components["broad-uniform"]["feasible"] == 0:
        raise ValueError("broad pilot component produced no feasible diagnostic evidence")
    if qd.get("status") != "skipped" or qd["actual_evaluations"] is not None:
        raise ValueError("rejected QD gate must produce a schema-compliant skip")
    if not all(row["feasible"] == "1" for row in rows(RESULTS / "mo" / "pareto.csv")):
        raise ValueError("MODE publication contains an infeasible point")


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "truss-sizing/plot_results.py"},
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
    text: str,
    color: str,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#455A64",
            linewidth=1.15,
        )
    )
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center")


def arrow(
    axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": "#455A64", "lw": 1.35},
    )


def architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.2, 3.9))
    axis.set(xlim=(0, 11.2), ylim=(0, 3.9))
    axis.axis("off")
    box(axis, 0.15, 1.45, 1.55, 0.9, "171 normalized\ncontrols", "#E3F2FD")
    box(axis, 2.05, 1.45, 1.75, 0.9, "exact-k topology\nsections · offsets", "#FFF3E0")
    box(axis, 4.15, 1.45, 1.70, 0.9, "connectivity +\nspectral rcond gate", "#FFEBEE")
    box(axis, 6.20, 1.45, 1.70, 0.9, "FEM solves +\nphysical residuals", "#E8F5E9")
    box(axis, 8.25, 2.05, 2.65, 0.75, "retry · CMA-ES · DE · BiteOpt\nmass minimization", "#F3E5F5")
    box(axis, 8.25, 0.90, 2.65, 0.75, "pilot gate · QD skip · MODE\nrobust trade-offs", "#E0F2F1")
    for start, end in [
        ((1.70, 1.90), (2.05, 1.90)),
        ((3.80, 1.90), (4.15, 1.90)),
        ((5.85, 1.90), (6.20, 1.90)),
        ((7.90, 1.90), (8.25, 2.42)),
        ((7.90, 1.90), (8.25, 1.27)),
    ]:
        arrow(axis, start, end)
    axis.text(
        5.6,
        3.48,
        "Discrete topology and catalogue sizing stay inside one deterministic objective",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        5.6,
        0.35,
        "failed mechanics return typed constraints · no fabricated stress or displacement",
        ha="center",
        color="#37474F",
    )
    save(figure, output)


def reference_nodes() -> np.ndarray:
    return np.array([(2.4 * column, 2.0 * row) for column in range(6) for row in range(3)])


def ground_structure(output: Path) -> None:
    nodes = reference_nodes()
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if np.linalg.norm(nodes[right] - nodes[left]) <= 5.0 + 1.0e-12:
                candidates.append((nodes[left], nodes[right]))
    figure, axis = plt.subplots(figsize=(10.6, 4.25))
    axis.add_collection(
        LineCollection(candidates, colors="#B0BEC5", linewidths=0.65, alpha=0.55)
    )
    axis.scatter(nodes[:, 0], nodes[:, 1], s=24, c="#37474F", zorder=3)
    axis.scatter(nodes[[0, 15], 0], nodes[[0, 15], 1], s=90, marker="^", c="#D55E00", zorder=4)
    axis.scatter(nodes[[8, 11], 0], nodes[[8, 11], 1], s=80, marker="v", c="#0072B2", zorder=4)
    for node in [8, 11]:
        axis.annotate(
            "",
            xy=(nodes[node, 0], nodes[node, 1] - 0.7),
            xytext=(nodes[node, 0], nodes[node, 1] + 0.05),
            arrowprops={"arrowstyle": "-|>", "color": "#0072B2", "lw": 1.6},
        )
    axis.set(
        title=f"Reference ground structure: 18 nodes and {len(candidates)} candidate members",
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
        xlim=(-0.7, 12.7),
        ylim=(-1.0, 4.8),
    )
    axis.text(0.0, -0.62, "pin", ha="center", color="#D55E00")
    axis.text(12.0, -0.62, "vertical roller", ha="center", color="#D55E00")
    save(figure, output)


def triangular_oracle(output: Path) -> None:
    data = rows(RESULTS / "validation" / "oracle.csv")
    forces = data[:3]
    labels = ["base tie", "left diagonal", "right diagonal"]
    analytic = np.array([float(row["analytic"]) / 1000.0 for row in forces])
    fem = np.array([float(row["fem"]) / 1000.0 for row in forces])
    x = np.arange(3)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))
    axes[0].bar(x - 0.18, analytic, 0.36, label="equilibrium", color="#0072B2")
    axes[0].bar(x + 0.18, fem, 0.36, label="FEM", color="#E69F00", alpha=0.85)
    axes[0].set(
        title="Member forces coincide",
        ylabel="Axial force (kN)",
        xticks=x,
        xticklabels=labels,
    )
    axes[0].legend()
    displacement = data[-1]
    values = [
        float(displacement["analytic"]) * 1000.0,
        float(displacement["fem"]) * 1000.0,
    ]
    axes[1].bar(["virtual work", "FEM"], values, color=["#009E73", "#CC79A7"])
    axes[1].set(
        title="Independent displacement oracle",
        ylabel="Apex vertical displacement (mm)",
    )
    axes[1].text(
        0.5,
        0.06,
        f"absolute error = {float(displacement['absolute_error']):.1e} m",
        transform=axes[1].transAxes,
        ha="center",
    )
    figure.tight_layout()
    save(figure, output)


def failure_contract(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 4.2))
    axis.set(xlim=(0, 10.5), ylim=(0, 4.2))
    axis.axis("off")
    box(axis, 0.20, 1.55, 1.70, 0.9, "decoded\ncandidate", "#E3F2FD")
    box(axis, 2.35, 2.65, 1.85, 0.75, "load node has\nsupport path?", "#FFF3E0")
    box(axis, 2.35, 1.00, 1.85, 0.75, "positive rank +\nrcond ≥ 10⁻¹⁰?", "#FFF3E0")
    box(axis, 4.75, 2.65, 1.70, 0.75, "disconnected\nconstraint only", "#FFEBEE")
    box(axis, 4.75, 1.00, 1.70, 0.75, "mechanism or\nconditioning", "#FFEBEE")
    box(axis, 7.00, 1.80, 1.45, 0.9, "Cholesky +\nload solves", "#E8F5E9")
    box(axis, 8.95, 1.80, 1.30, 0.9, "stress · buckling\nmovement", "#E0F2F1")
    arrow(axis, (1.90, 2.00), (2.35, 3.02))
    arrow(axis, (1.90, 2.00), (2.35, 1.37))
    arrow(axis, (4.20, 3.02), (4.75, 3.02))
    arrow(axis, (4.20, 1.37), (4.75, 1.37))
    arrow(axis, (4.20, 3.02), (7.00, 2.38))
    arrow(axis, (4.20, 1.37), (7.00, 2.12))
    arrow(axis, (8.45, 2.25), (8.95, 2.25))
    axis.text(
        5.25,
        3.92,
        "Unavailable mechanics remain unavailable",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        5.25,
        0.35,
        "optimizer sentinels transport failure; artifacts store missing physical fields",
        ha="center",
        color="#37474F",
    )
    save(figure, output)


def so_comparison(output: Path) -> None:
    data = rows(RESULTS / "so" / "arms.csv")
    labels = [row["optimizer"].replace("-seeded", "\nseeded").replace("-", "\n") for row in data]
    mass = [float(row["mass_kg"]) for row in data]
    delta = [float(row["delta_vs_seed"]) for row in data]
    feasible = [row["feasible"] == "1" for row in data]
    colors = [COLORS[row["optimizer"]] for row in data]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    bars = axes[0].bar(labels, mass, color=colors)
    for bar, valid in zip(bars, feasible):
        if not valid:
            bar.set_hatch("//")
            bar.set_alpha(0.55)
    axes[0].set(title="Retained structural mass", ylabel="Mass (kg)")
    axes[0].text(
        0.98,
        0.96,
        "hatched = infeasible",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
    )
    axes[1].bar(labels[1:], np.array(delta[1:]) / 1000.0, color=colors[1:])
    axes[1].axhline(0.0, color="#455A64", lw=0.9)
    axes[1].set(
        title="Change from explicit seed",
        ylabel="Penalized objective change (thousand kg)",
    )
    figure.tight_layout()
    save(figure, output)


def selected_truss(output: Path) -> None:
    data = rows(RESULTS / "so" / "best_members.csv")
    segments = [
        [(float(row["x_a_m"]), float(row["y_a_m"])), (float(row["x_b_m"]), float(row["y_b_m"]))]
        for row in data
    ]
    force = np.array([float(row["axial_force_n"]) for row in data])
    utilization = np.array([float(row["utilization"]) for row in data])
    scale = np.max(np.abs(force)).clip(min=1.0)
    colors = plt.get_cmap("coolwarm")((force / scale + 1.0) / 2.0)
    widths = 0.8 + 4.2 * utilization / max(float(np.max(utilization)), 1.0e-12)
    nodes = np.unique(np.array(segments).reshape(-1, 2), axis=0)
    figure, axis = plt.subplots(figsize=(10.8, 4.3))
    axis.add_collection(LineCollection(segments, colors=colors, linewidths=widths))
    axis.scatter(nodes[:, 0], nodes[:, 1], s=18, c="#263238", zorder=3)
    axis.set(
        title="Lowest-mass feasible scalar design (DE): red tension, blue compression",
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
    )
    axis.autoscale()
    axis.margins(0.08)
    axis.text(
        0.99,
        0.02,
        "line width ∝ governing utilization",
        transform=axis.transAxes,
        ha="right",
        color="#37474F",
    )
    save(figure, output)


def descriptor_pilot(output: Path) -> None:
    data = rows(RESULTS / "pilot" / "pilot.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    colors = ["#0072B2", "#009E73", "#D55E00"]
    for arm in range(3):
        arm_rows = [row for row in data if int(row["arm"]) == arm]
        for generator, marker in [
            ("structured-local", "o"),
            ("broad-uniform", "x"),
        ]:
            generated = [row for row in arm_rows if row["generator"] == generator]
            axes[0].scatter(
                [float(row["depth_to_span_train"]) for row in generated],
                [float(row["survival_train"]) for row in generated],
                s=26,
                alpha=0.75,
                color=colors[arm],
                marker=marker,
            )
            axes[1].scatter(
                [float(row["utilization_spread_train"]) for row in generated],
                [float(row["survival_train"]) for row in generated],
                s=26,
                alpha=0.75,
                color=colors[arm],
                marker=marker,
            )
    axes[0].set(
        title="D1: depth / span × removal survival",
        xlabel="Depth / span",
        ylabel="Removal survival",
    )
    axes[1].set(
        title="D2: utilization spread × survival",
        xlabel="Utilization standard deviation",
        ylabel="Removal survival",
    )
    arm_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            label=f"seed arm {index + 1}",
        )
        for index, color in enumerate(colors)
    ]
    generator_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#455A64",
            linestyle="none",
            label="structured-local",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#455A64",
            linestyle="none",
            label="broad-uniform",
        ),
    ]
    axes[0].legend(handles=arm_handles + generator_handles, ncol=2)
    figure.suptitle(
        "Broad sampling widens reach, but 95 of 99 feasible designs have zero survival",
        weight="bold",
    )
    figure.tight_layout()
    save(figure, output)


def mo_pareto(output: Path) -> None:
    data = rows(RESULTS / "mo" / "pareto.csv")
    mass = np.array([float(row["objective_mass_kg"]) for row in data])
    movement = np.array([float(row["objective_displacement_m"]) * 1000.0 for row in data])
    members = np.array([float(row["objective_active_count"]) for row in data])
    selected = np.array([row["selected"] == "1" for row in data])
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    points = axis.scatter(
        mass,
        movement,
        c=members,
        cmap="viridis",
        s=44,
        edgecolors="#263238",
        linewidths=0.35,
    )
    axis.scatter(
        mass[selected],
        movement[selected],
        facecolors="none",
        edgecolors="#D55E00",
        s=130,
        linewidths=1.8,
        label="reported extremes",
    )
    axis.set(
        title="Finite-budget MODE population front",
        xlabel="Structural mass (kg)",
        ylabel="Maximum displacement (mm)",
    )
    figure.colorbar(points, ax=axis, label="Active members")
    axis.legend()
    axis.text(
        0.02,
        0.03,
        "Every retained point lost ≥1 load path under a member removal;\nredundancy degradation is capped at 100.",
        transform=axis.transAxes,
        color="#37474F",
    )
    save(figure, output)


def condition_sensitivity(output: Path) -> None:
    data = rows(RESULTS / "validation" / "condition_sensitivity.csv")
    thresholds = np.array([float(row["rcond_threshold"]) for row in data])
    measured = float(data[0]["measured_rcond"])
    passed = np.array([row["passes"] == "1" for row in data])
    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    passed = passed[order]
    figure, axis = plt.subplots(figsize=(8.2, 3.9))
    axis.scatter(
        thresholds[passed],
        np.ones(np.sum(passed)),
        s=70,
        marker="o",
        color="#009E73",
        label="baseline passes",
    )
    axis.scatter(
        thresholds[~passed],
        np.ones(np.sum(~passed)),
        s=70,
        marker="x",
        color="#D55E00",
        label="baseline rejected",
    )
    axis.axvline(measured, color="#0072B2", lw=1.6, label=f"measured {measured:.2e}")
    axis.axvline(1.0e-10, color="#455A64", lw=1.2, ls="--", label="frozen gate 1e-10")
    axis.set(
        title="Conditioning policy is explicit and sensitivity-tested",
        xlabel="Required reciprocal condition",
        xscale="log",
        yticks=[],
        ylim=(0.88, 1.12),
    )
    axis.legend(ncol=2, loc="lower left")
    save(figure, output)


def render(directory: Path) -> None:
    configure()
    validate_artifacts()
    renderers = [
        architecture,
        ground_structure,
        triangular_oracle,
        failure_contract,
        so_comparison,
        selected_truss,
        descriptor_pilot,
        mo_pareto,
        condition_sensitivity,
    ]
    for filename, renderer in zip(FIGURES, renderers):
        renderer(directory / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        render(IMAGES)
        for filename in FIGURES:
            print(IMAGES / filename)
        return
    with tempfile.TemporaryDirectory() as temporary:
        generated = Path(temporary)
        render(generated)
        stale = [
            filename
            for filename in FIGURES
            if not filecmp.cmp(generated / filename, IMAGES / filename, shallow=False)
        ]
    if stale:
        raise SystemExit("missing or stale truss-sizing figures:\n" + "\n".join(stale))
    print("truss-sizing figures are current")


if __name__ == "__main__":
    main()
