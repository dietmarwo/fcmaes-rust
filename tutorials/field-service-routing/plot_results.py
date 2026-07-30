#!/usr/bin/env python3
"""Render deterministic field-service-routing figures from Rust artifacts."""

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
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "publication"
IMAGES = ROOT / "images"
FIGURES = [
    "architecture.svg",
    "decoder.svg",
    "staircase.svg",
    "scenario-stress.svg",
    "so-comparison.svg",
    "route-map.svg",
    "baseline-comparison.svg",
    "descriptor-gate.svg",
    "mo-pareto.svg",
]
COLORS = {
    "seed": "#999999",
    "cma": "#0072B2",
    "de": "#009E73",
    "bite": "#D55E00",
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
            "svg.hashsalt": "fcmaes-field-service-routing-v1",
        }
    )


def validate_artifacts() -> None:
    validator = json.loads(
        (RESULTS / "validation" / "summary.json").read_text(encoding="utf-8")
    )
    if validator["supplied_routes"] != 1000:
        raise ValueError("validator study does not contain 1,000 routes")
    if validator["max_absolute_discrepancy"] > 1.0e-9:
        raise ValueError("independent scorer discrepancy exceeds tolerance")
    if not validator.get("expected_bit_exact", False):
        raise ValueError("validator manifest does not explain exact agreement")
    pilot = json.loads((RESULTS / "pilot" / "run.json").read_text(encoding="utf-8"))
    qd = json.loads((RESULTS / "qd" / "run.json").read_text(encoding="utf-8"))
    if pilot["qd_decision"] != "rejected":
        raise ValueError("publication figures expect the measured rejected QD gate")
    if qd.get("status") != "skipped" or qd["actual_evaluations"] is not None:
        raise ValueError("rejected QD publication must be a schema-compliant skip")
    arms = rows(RESULTS / "so" / "arms.csv")
    if arms[0]["arm"] != "seed":
        raise ValueError("scalar publication does not expose its seed baseline")
    if not all(row["feasible"] == "true" for row in arms):
        raise ValueError("scalar publication contains an infeasible retained plan")
    if not any(
        row["search_found_feasible_improvement"] == "true" for row in arms[1:]
    ):
        raise ValueError("publication search did not improve the seed")
    if pilot["archive"]["capacity"] != 120:
        raise ValueError("pilot and publication archive capacities differ")
    if pilot["pairs"]["D1"]["holdout_feasible_fraction"] != 0.0:
        raise ValueError("publication robustness finding changed")
    if not all(row["feasible"] == "true" for row in rows(RESULTS / "mo" / "pareto.csv")):
        raise ValueError("MODE publication contains an infeasible point")


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "field-service-routing/plot_results.py"},
    )
    plt.close(figure)
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n",
        encoding="utf-8",
    )


def box(axis: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#455A64",
            linewidth=1.15,
        )
    )
    axis.text(x + w / 2, y + h / 2, text, ha="center", va="center")


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": "#455A64", "lw": 1.35},
    )


def architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.4, 3.7))
    axis.set(xlim=(0, 10.4), ylim=(0, 3.7))
    axis.axis("off")
    box(axis, 0.15, 1.35, 1.55, 0.9, "104 continuous\nrandom keys", "#E3F2FD")
    box(axis, 2.05, 1.35, 1.55, 0.9, "active mask +\navailable fleet", "#FFF3E0")
    box(axis, 3.95, 1.35, 1.55, 0.9, "exact-once\nroute decoder", "#E8F5E9")
    box(axis, 5.85, 1.35, 1.55, 0.9, "forward pass\ncost + residuals", "#F3E5F5")
    box(axis, 7.80, 1.95, 2.25, 0.75, "retry · MODE\ncandidate parallelism", "#FFEBEE")
    box(axis, 7.80, 0.88, 2.25, 0.75, "pilot gate · MAP-Elites\nrepertoire or skip", "#E0F2F1")
    for start, end in [
        ((1.70, 1.80), (2.05, 1.80)),
        ((3.60, 1.80), (3.95, 1.80)),
        ((5.50, 1.80), (5.85, 1.80)),
        ((7.40, 1.80), (7.80, 2.32)),
        ((7.40, 1.80), (7.80, 1.25)),
    ]:
        arrow(axis, start, end)
    axis.text(
        5.2,
        3.28,
        "Continuous global search outside, deterministic combinatorial decoding inside",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        5.2,
        0.38,
        "skills and exact-once service by construction · capacity, windows, and shifts as residuals",
        ha="center",
        color="#37474F",
    )
    save(figure, output)


def decoder(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.7, 4.1))
    axis.set(xlim=(0, 9.7), ylim=(0, 4.1))
    axis.axis("off")
    box(axis, 0.25, 2.45, 1.85, 0.82, "assignment key uᵢ\ncontinuous [0,1]", "#E3F2FD")
    box(axis, 0.25, 0.85, 1.85, 0.82, "priority key pᵢ\ncontinuous [0,1]", "#E3F2FD")
    box(axis, 2.75, 2.45, 2.15, 0.82, "equal-width bin over\ncompatible available vehicles", "#FFF3E0")
    box(axis, 2.75, 0.85, 2.15, 0.82, "total order by\n(pᵢ, task index)", "#FFF3E0")
    box(axis, 5.55, 1.65, 1.65, 0.95, "one vehicle +\none rank per task", "#E8F5E9")
    box(axis, 7.85, 1.65, 1.55, 0.95, "vehicle routes\nexactly once", "#F3E5F5")
    for start, end in [
        ((2.10, 2.86), (2.75, 2.86)),
        ((2.10, 1.26), (2.75, 1.26)),
        ((4.90, 2.86), (5.55, 2.20)),
        ((4.90, 1.26), (5.55, 2.00)),
        ((7.20, 2.12), (7.85, 2.12)),
    ]:
        arrow(axis, start, end)
    axis.text(
        4.85,
        3.75,
        "No repair: structural validity is an invariant of decoding",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        4.85,
        0.25,
        "inactive reserve slots are ignored; disruption masks never resize the vector",
        ha="center",
        color="#37474F",
    )
    save(figure, output)


def staircase(output: Path) -> None:
    data = rows(RESULTS / "staircase" / "staircase.csv")
    key = np.array([float(row["key"]) for row in data])
    objective = np.array([float(row["objective"]) for row in data])
    position = np.array([int(row["position"]) for row in data])
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    axes[0].step(key, objective, where="post", color="#0072B2", lw=1.9)
    axes[0].set(
        title="Objective is constant inside decoder cells",
        xlabel="Swept priority key",
        ylabel="Robust penalized objective",
    )
    axes[1].step(key, position, where="post", color="#D55E00", lw=1.9)
    axes[1].set(
        title=f"Decoded route states: {len(set(position))} (bound = 7)",
        xlabel="Swept priority key",
        ylabel="Task position in route",
        yticks=sorted(set(position)),
    )
    figure.tight_layout()
    save(figure, output)


def scenario_stress(output: Path) -> None:
    data = rows(RESULTS / "scenarios" / "robustness.csv")
    labels = [row["scenario"].replace("_", "\n") for row in data]
    costs = np.array([float(row["cost"]) for row in data])
    lateness = np.array([float(row["lateness_s"]) / 3600.0 for row in data])
    colors = ["#0072B2"] * 5 + ["#D55E00"] * 4
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 5.6), sharex=True)
    x = np.arange(len(data))
    axes[0].bar(x, costs, color=colors, alpha=0.85)
    axes[0].set(title="Training cases stay feasible; holdouts change failure kind", ylabel="Cost")
    axes[1].bar(x, lateness, color=colors, alpha=0.85)
    axes[1].set(ylabel="Aggregate lateness (h)", xlabel="Blue: training · orange: holdout")
    axes[1].set_xticks(x, labels)
    figure.tight_layout()
    save(figure, output)


def so_comparison(output: Path) -> None:
    data = rows(RESULTS / "so" / "arms.csv")
    arms = [row["arm"] for row in data]
    costs = [float(row["worst_cost"]) for row in data]
    optimized = data[1:]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.65))
    axes[0].bar(arms, costs, color=[COLORS[arm] for arm in arms])
    axes[0].set(title="Retained robust cost", ylabel="Worst training cost")
    axes[0].set_ylim(min(costs) - 2.0, max(costs) + 2.0)
    axes[1].bar(
        [row["arm"] for row in optimized],
        [float(row["delta_vs_seed"]) for row in optimized],
        color=[COLORS[row["arm"]] for row in optimized],
    )
    axes[1].axhline(0, color="#424242", lw=1.0)
    axes[1].set(title="Best feasible delta to seed", ylabel="Cost delta")
    axes[2].bar(
        [row["arm"] for row in optimized],
        [int(row["actual_evaluations"]) for row in optimized],
        color=[COLORS[row["arm"]] for row in optimized],
    )
    axes[2].axhline(10_000, color="#424242", ls="--", lw=1.1, label="requested")
    axes[2].set(title="Population completion", ylabel="Objective calls")
    axes[2].legend()
    figure.suptitle("BiteOpt improves the explicit construction baseline", weight="bold")
    figure.tight_layout()
    save(figure, output)


def route_map(output: Path) -> None:
    instance_rows = rows(ROOT / "instances" / "instance-00.csv")
    points = {
        int(row["id"]): (
            float(row["x_or_capacity"]),
            float(row["y_or_shift_start"]),
        )
        for row in instance_rows
        if row["kind"] == "task"
    }
    arm_rows = rows(RESULTS / "so" / "arms.csv")
    best = min(arm_rows[1:], key=lambda row: float(row["worst_cost"]))["arm"]
    route_row = next(
        row
        for row in rows(RESULTS / "so" / "routes.csv")
        if row["arm"] == best and row["scenario"] == "nominal"
    )
    parsed = []
    for route in route_row["routes"].split("|"):
        vehicle, tasks = route.split(":")
        parsed.append((int(vehicle), [int(task) for task in tasks.split("-")]))
    palette = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(6.7, 5.8))
    for vehicle, tasks in parsed:
        route_points = [(0.0, 0.0)] + [points[task] for task in tasks] + [(0.0, 0.0)]
        x, y = zip(*route_points)
        axis.plot(x, y, "-o", ms=3.5, lw=1.25, color=palette(vehicle), label=f"vehicle {vehicle}")
    axis.scatter([0], [0], marker="s", s=75, color="#111111", zorder=5, label="depot")
    axis.set(
        title=f"Nominal replay of the best robust seven-route plan ({best})",
        xlabel="East (km)",
        ylabel="North (km)",
        aspect="equal",
    )
    axis.legend(ncol=2, loc="upper left")
    figure.tight_layout()
    save(figure, output)


def baseline_comparison(output: Path) -> None:
    data = rows(RESULTS / "baseline" / "baseline_comparison.csv")
    names = [row["instance"].replace("fsr-", "") for row in data]
    gap = [float(row["gap_percent"]) for row in data]
    operations = [int(row["operations"]) for row in data]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    axes[0].bar(
        names,
        gap,
        color=["#009E73" if value < 0 else "#D55E00" for value in gap],
    )
    axes[0].axhline(0, color="#424242", lw=1.0)
    axes[0].set(
        title="Greedy + 2-opt gap to generated witness",
        xlabel="Frozen instance",
        ylabel="Cost gap (%) · negative is better",
    )
    axes[1].bar(names, operations, color="#0072B2")
    axes[1].set(
        title="Deterministic operation accounting",
        xlabel="Frozen instance",
        ylabel="Attempted 2-opt moves",
    )
    figure.tight_layout()
    save(figure, output)


def descriptor_gate(output: Path) -> None:
    data = rows(RESULTS / "pilot" / "pilot.csv")
    vehicles = np.array([float(row["vehicles_train"]) for row in data])
    imbalance = np.array([float(row["imbalance_cv_train"]) for row in data])
    distance = np.array([float(row["distance_km_train"]) for row in data])
    uniform = np.array([row["source"] == "uniform" for row in data])
    manifest = json.loads((RESULTS / "pilot" / "run.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))
    scatter = None
    for selected, marker, label in (
        (~uniform, "o", "local feasible"),
        (uniform, "^", "uniform feasible"),
    ):
        if selected.any():
            scatter = axes[0].scatter(
                vehicles[selected],
                imbalance[selected],
                c=distance[selected],
                cmap="viridis",
                s=22,
                alpha=0.75,
                marker=marker,
                label=label,
            )
    axes[0].set(
        title=f"D1 coverage = {100 * manifest['pairs']['D1']['coverage']:.1f}%",
        xlabel="Vehicles used",
        ylabel="Route-distance CV",
    )
    axes[0].legend()
    if scatter is not None:
        figure.colorbar(scatter, ax=axes[0], label="Nominal distance (km)")

    generator = manifest["generator"]
    sources = ["local", "uniform"]
    attempted = [generator[f"{source}_attempted"] for source in sources]
    feasible = [generator[f"{source}_feasible"] for source in sources]
    axes[1].bar(
        sources,
        [
            100 * kept / total
            for kept, total in zip(feasible, attempted, strict=True)
        ],
        color=["#0072B2", "#D55E00"],
    )
    axes[1].set(
        title="Training-feasible fraction by generator",
        ylabel="Percent of attempted candidates",
    )

    pairs = ["D1", "D2", "D3"]
    coverage = [100 * manifest["pairs"][pair]["coverage"] for pair in pairs]
    feasible_retention = [
        100 * manifest["pairs"][pair]["holdout_feasible_fraction"] for pair in pairs
    ]
    niche_retention = [
        100 * manifest["pairs"][pair]["holdout_niche_retention"] for pair in pairs
    ]
    coarse_retention = [
        100 * manifest["pairs"][pair]["coarse_holdout_niche_retention"]
        for pair in pairs
    ]
    x = np.arange(3)
    axes[2].bar(x - 0.27, coverage, width=0.18, color="#0072B2", label="coverage")
    axes[2].bar(
        x - 0.09,
        feasible_retention,
        width=0.18,
        color="#D55E00",
        label="holdout feasible",
    )
    axes[2].bar(
        x + 0.09,
        niche_retention,
        width=0.18,
        color="#009E73",
        label="same niche",
    )
    axes[2].bar(
        x + 0.27,
        coarse_retention,
        width=0.18,
        color="#CC79A7",
        label="coarse niche",
    )
    axes[2].axhline(40, color="#0072B2", ls="--", lw=1.0)
    axes[2].axhline(60, color="#424242", ls=":", lw=1.0)
    axes[2].set(
        title="Reachability and robustness remain distinct",
        xticks=x,
        xticklabels=pairs,
        ylabel="Percent",
    )
    axes[2].legend(fontsize=7)
    figure.tight_layout()
    save(figure, output)


def mo_pareto(output: Path) -> None:
    data = rows(RESULTS / "mo" / "pareto.csv")
    distance = np.array([float(row["objective_distance_km"]) for row in data])
    vehicles = np.array([float(row["objective_vehicles"]) for row in data])
    makespan = np.array([float(row["objective_makespan_s"]) / 3600.0 for row in data])
    selected = np.array([row["selected"] == "true" for row in data])
    figure, axis = plt.subplots(figsize=(6.7, 4.3))
    scatter = axis.scatter(
        distance,
        makespan,
        c=vehicles,
        cmap="plasma_r",
        s=np.where(selected, 95, 42),
        edgecolors=np.where(selected, "#111111", "none"),
    )
    axis.set(
        title="Soft-window feasible nondominated population",
        xlabel="Distance (km)",
        ylabel="Makespan (h)",
    )
    figure.colorbar(scatter, ax=axis, label="Vehicles used")
    figure.tight_layout()
    save(figure, output)


RENDERERS = {
    "architecture.svg": architecture,
    "decoder.svg": decoder,
    "staircase.svg": staircase,
    "scenario-stress.svg": scenario_stress,
    "so-comparison.svg": so_comparison,
    "route-map.svg": route_map,
    "baseline-comparison.svg": baseline_comparison,
    "descriptor-gate.svg": descriptor_gate,
    "mo-pareto.svg": mo_pareto,
}


def render(directory: Path) -> None:
    validate_artifacts()
    for name in FIGURES:
        RENDERERS[name](directory / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="render to a temporary directory and fail if checked-in figures differ",
    )
    arguments = parser.parse_args()
    configure()
    if arguments.check:
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary)
            render(generated)
            stale = [
                name
                for name in FIGURES
                if not (IMAGES / name).is_file()
                or not filecmp.cmp(generated / name, IMAGES / name, shallow=False)
            ]
            if stale:
                raise SystemExit("missing or stale field-service figures:\n" + "\n".join(stale))
        print("field-service-routing figures are current")
    else:
        render(IMAGES)
        for name in FIGURES:
            print(IMAGES / name)


if __name__ == "__main__":
    main()
