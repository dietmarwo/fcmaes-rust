#!/usr/bin/env python3
"""Render deterministic energy-hub figures from checked-in Rust artifacts."""

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
COLORS = {
    "seed": "#999999",
    "cma": "#0072B2",
    "de": "#009E73",
    "bite": "#D55E00",
}
FIGURES = [
    "architecture.svg",
    "landscape.svg",
    "dispatch-stack.svg",
    "so-comparison.svg",
    "descriptor-pilot.svg",
    "qd-archive.svg",
    "mo-pareto.svg",
    "annual-hydrogen.svg",
    "budget-accounting.svg",
]
REQUIRED = {
    "schema_version",
    "tutorial",
    "formulation",
    "command",
    "seed",
    "workers",
    "requested_evaluations",
    "actual_evaluations",
    "elapsed_seconds",
    "objectives",
    "descriptors",
    "artifacts",
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
            "svg.hashsalt": "fcmaes-energy-hub-bilevel-v1",
        }
    )


def validate_artifacts() -> None:
    manifests = sorted(RESULTS.glob("*/run.json"))
    if len(manifests) != 6:
        raise ValueError(f"expected six publication manifests, found {len(manifests)}")
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        missing = REQUIRED - manifest.keys()
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        if manifest["schema_version"] != 1:
            raise ValueError(f"{path} uses the wrong schema")
        if manifest["tutorial"] != "energy-hub-bilevel":
            raise ValueError(f"{path} names the wrong tutorial")
        for artifact in manifest["artifacts"].values():
            if not (path.parent / artifact).is_file():
                raise ValueError(f"{path} references missing {artifact}")

    landscape = json.loads(
        (RESULTS / "landscape" / "run.json").read_text(encoding="utf-8")
    )
    if landscape["convexity_max_relative_violation"] > 1.0e-8:
        raise ValueError("convex landscape baseline fails its invariant")
    if landscape["boundary_sign_disagreements"] == 0:
        raise ValueError("boundary derivative probe is unexpectedly inert")

    pilot = json.loads((RESULTS / "pilot" / "run.json").read_text(encoding="utf-8"))
    qd = json.loads((RESULTS / "qd" / "run.json").read_text(encoding="utf-8"))
    if pilot["qd_decision"] != "accepted" or pilot["selected_pair"] != "d1":
        raise ValueError("figures expect the corrected D1 descriptor verdict")
    if qd["actual_evaluations"] != qd["requested_evaluations"]:
        raise ValueError("accepted QD arm did not consume its registered budget")
    if qd["qd"].get("grid_shape") != [12, 10]:
        raise ValueError("publication QD artifact does not expose the native 12 x 10 grid")
    for row in rows(RESULTS / "qd" / "qd_archive.csv"):
        if not (0 <= int(row["grid_x"]) < 12 and 0 <= int(row["grid_y"]) < 10):
            raise ValueError("QD archive contains an invalid native-grid coordinate")

    scalar_rows = rows(RESULTS / "so" / "best.csv")
    if scalar_rows[0]["optimizer"] != "seed":
        raise ValueError("scalar comparison does not publish its seed baseline first")
    if not any(row["feasible"] == "1" for row in scalar_rows[1:]):
        raise ValueError("scalar comparison has no feasible design")
    for row in rows(RESULTS / "mo" / "pareto.csv"):
        if any(
            float(row[column]) > 1.0e-12
            for column in (
                "constraint_self_sufficiency",
                "constraint_cycles",
                "constraint_lp_status",
            )
        ):
            raise ValueError("MODE artifact contains an infeasible point")
    annual = json.loads((RESULTS / "annual" / "run.json").read_text(encoding="utf-8"))
    hourly = annual["hourly_validation"]
    if max(
        hourly["max_balance_residual_kw"],
        hourly["max_storage_residual_kwh"],
    ) > 1.0e-6:
        raise ValueError("hourly annual replay violates a balance invariant")


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "energy-hub-bilevel/plot_results.py"},
    )
    plt.close(figure)
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n",
        encoding="utf-8",
    )


def architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 3.55))
    axis.set_xlim(0, 10.2)
    axis.set_ylim(0, 3.55)
    axis.axis("off")
    boxes = [
        (0.15, 1.28, 1.55, 0.95, "10 normalized\nouter controls", "#E3F2FD"),
        (2.00, 1.28, 1.55, 0.95, "tiers + booleans\n+ capacities", "#FFF3E0"),
        (3.85, 1.28, 1.55, 0.95, "five named\nLP dispatches", "#E8F5E9"),
        (5.70, 1.28, 1.55, 0.95, "robust LCOE\n+ residuals", "#F3E5F5"),
        (7.65, 1.78, 2.25, 0.72, "retry · MODE\ncandidate parallelism", "#FFEBEE"),
        (7.65, 0.75, 2.25, 0.72, "chronological H₂\n6 h size → 1 h replay", "#E0F2F1"),
    ]
    for x, y, width, height, label, color in boxes:
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
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    for start, end in [
        ((1.70, 1.75), (2.00, 1.75)),
        ((3.55, 1.75), (3.85, 1.75)),
        ((5.40, 1.75), (5.70, 1.75)),
        ((7.25, 1.75), (7.65, 2.14)),
        ((7.25, 1.75), (7.65, 1.11)),
    ]:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#455A64", "lw": 1.35},
        )
    axis.text(
        5.05,
        3.18,
        "Bilevel sizing: global search outside, proven-optimal pure-Rust LP inside",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        4.95,
        0.36,
        "microlp owns serial dispatch · fcmaes owns independent candidate parallelism",
        ha="center",
        color="#37474F",
    )
    save(figure, output)


def landscape(output: Path) -> None:
    data = rows(RESULTS / "landscape" / "landscape.csv")
    manifest = json.loads(
        (RESULTS / "landscape" / "run.json").read_text(encoding="utf-8")
    )
    coordinate = np.array([float(row["coordinate"]) for row in data])
    convex = np.array([float(row["convex_total_cost"]) for row in data]) / 1.0e6
    tiered = np.array([float(row["tiered_total_cost"]) for row in data]) / 1.0e6
    ratio = np.array([float(row["ratio_lcoe"]) for row in data])
    delivered = np.array([float(row["delivered_lcoe"]) for row in data])
    figure, axes = plt.subplots(1, 2, figsize=(9.7, 3.75))
    axes[0].plot(coordinate, convex, color="#0072B2", lw=2.0, label="linear CAPEX")
    axes[0].step(
        coordinate,
        tiered,
        where="mid",
        color="#D55E00",
        lw=1.7,
        label="tiered grid CAPEX",
    )
    axes[0].set(
        title="The continuous total-cost baseline is convex",
        xlabel="Capacity-path coordinate",
        ylabel="Annual total cost (million)",
    )
    axes[0].legend()
    axes[1].plot(coordinate, ratio, color="#0072B2", lw=1.7, label="LCOE, fixed architecture")
    axes[1].step(
        coordinate,
        delivered,
        where="mid",
        color="#CC79A7",
        lw=1.8,
        label="LCOE + tiers + switches",
    )
    axes[1].set(
        title="The delivered outer objective crosses discrete pieces",
        xlabel="Normalized outer path",
        ylabel="Cost (currency/kWh)",
    )
    axes[1].legend()
    figure.text(
        0.50,
        0.01,
        f"finite differences disagree at {manifest['boundary_sign_disagreements']}/"
        f"{manifest['boundary_probes']} registered boundary probes; "
        f"convexity violation = {manifest['convexity_max_relative_violation']:.1e}",
        ha="center",
        color="#37474F",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save(figure, output)


def dispatch_stack(output: Path) -> None:
    data = rows(RESULTS / "so" / "dispatch.csv")
    hour = np.array([float(row["hour"]) for row in data])
    load = np.array([float(row["load_kw"]) for row in data])
    renewable = np.array([float(row["renewable_kw"]) for row in data])
    imported = np.array([float(row["import_kw"]) for row in data])
    exported = np.array([float(row["export_kw"]) for row in data])
    soc = np.array([float(row["soc_kwh"]) for row in data])
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    axes[0].plot(hour, load, color="#222222", lw=1.5, label="electrical load")
    axes[0].fill_between(hour, 0, renewable, color="#F0E442", alpha=0.55, label="available renewable")
    axes[0].fill_between(hour, 0, imported, color="#0072B2", alpha=0.50, label="grid import")
    axes[0].plot(hour, exported, color="#009E73", lw=1.0, label="grid export")
    axes[0].set(title="Selected robust design: twelve independently cyclic days", ylabel="Power (kW)")
    axes[0].legend(ncol=4, loc="upper right")
    axes[1].fill_between(hour, 0, soc, color="#CC79A7", alpha=0.55)
    axes[1].set(xlabel="Representative-horizon hour", ylabel="Battery SOC (kWh)")
    figure.tight_layout()
    save(figure, output)


def so_comparison(output: Path) -> None:
    best = rows(RESULTS / "so" / "best.csv")
    manifest = json.loads((RESULTS / "so" / "run.json").read_text(encoding="utf-8"))
    names = [row["optimizer"] for row in best]
    lcoe = [float(row["mean_lcoe"]) for row in best]
    objective = [float(row["objective"]) for row in best]
    colors = [COLORS[name] for name in names]
    arm_budget = {row["optimizer"]: row["budget"] for row in manifest["arms"]}
    optimizer_names = [name for name in names if name in arm_budget]
    optimizer_colors = [COLORS[name] for name in optimizer_names]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.65))
    locations = np.arange(len(names))
    axes[0].bar(locations - 0.17, lcoe, 0.34, color=colors, alpha=0.60, label="mean LCOE")
    axes[0].bar(locations + 0.17, objective, 0.34, color=colors, label="penalized objective")
    for index, row in enumerate(best):
        if row["feasible"] == "0":
            axes[0].text(index, objective[index], " infeasible", rotation=90, va="bottom", ha="center")
    axes[0].set_xticks(locations, [name.upper() for name in names])
    axes[0].set(
        title="Explicit seed baseline and equal-budget optimizers",
        ylabel="Cost (currency/kWh)",
    )
    axes[0].legend()
    axes[1].bar(
        [name.upper() for name in optimizer_names],
        [arm_budget[name]["simplex_iterations"] for name in optimizer_names],
        color=optimizer_colors,
    )
    axes[1].set(title="The same outer budget does not mean equal LP work", ylabel="Simplex pivots")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    figure.tight_layout()
    save(figure, output)


def descriptor_pilot(output: Path) -> None:
    data = rows(RESULTS / "pilot" / "pilot.csv")
    manifest = json.loads((RESULTS / "pilot" / "run.json").read_text(encoding="utf-8"))
    x = np.array([float(row["d1_axis1_train"]) for row in data])
    y = np.array([float(row["d1_axis2_train"]) for row in data])
    held_x = np.array([float(row["d1_axis1_holdout"]) for row in data])
    held_y = np.array([float(row["d1_axis2_holdout"]) for row in data])
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].scatter(x, y, c="#0072B2", s=18, alpha=0.65, label="training")
    for index in np.linspace(0, len(x) - 1, min(32, len(x)), dtype=int):
        axes[0].plot(
            [x[index], held_x[index]],
            [y[index], held_y[index]],
            color="#D55E00",
            alpha=0.28,
            lw=0.8,
        )
    axes[0].scatter(held_x, held_y, c="#D55E00", marker="x", s=18, label="battery derating")
    axes[0].set(
        title="D1 is emergent, but holdout niches move",
        xlabel="Daily battery throughput / installed kWh",
        ylabel="Peak import / grid capacity",
    )
    axes[0].legend()
    diagnostic = manifest["diagnostics"]["d1"]
    values = [
        abs(diagnostic["rank_correlation"]),
        max(diagnostic["clipping_axis_1"], diagnostic["clipping_axis_2"]),
        diagnostic["coverage"],
        diagnostic["minimum_seed_coverage"],
        diagnostic["holdout_niche_retention"],
    ]
    limits = [0.7, 0.1, 0.4, None, 0.6]
    passing = [
        values[0] < limits[0],
        values[1] < limits[1],
        values[2] > limits[2],
        None,
        values[4] > limits[4],
    ]
    labels = ["|ρ|", "clipping", "coverage", "min-seed\ncoverage", "retention"]
    axes[1].bar(
        labels,
        values,
        color=[
            "#999999" if passed is None else "#009E73" if passed else "#D55E00"
            for passed in passing
        ],
    )
    for index, limit in enumerate(limits):
        if limit is None:
            continue
        axes[1].plot([index - 0.38, index + 0.38], [limit, limit], color="#222222", ls="--", lw=1.0)
    axes[1].set(
        title="Pre-registered D1 verdict: accepted",
        ylabel="Measured fraction or absolute correlation",
        ylim=(0, 0.82),
    )
    figure.tight_layout()
    save(figure, output)


def qd_archive(output: Path) -> None:
    data = rows(RESULTS / "qd" / "qd_archive.csv")
    progress = rows(RESULTS / "qd" / "qd_convergence.csv")
    quality = np.array([float(row["quality_train"]) for row in data])
    x = np.array([int(row["grid_x"]) for row in data])
    y = np.array([int(row["grid_y"]) for row in data])
    retained = np.array([row["retained_niche"] == "1" for row in data])

    grid = np.full((10, 12), np.nan)
    grid[y, x] = quality
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    image = axes[0].pcolormesh(
        np.arange(13) - 0.5,
        np.arange(11) - 0.5,
        np.ma.masked_invalid(grid),
        cmap="viridis_r",
        vmin=float(quality.min()),
        vmax=float(quality.max()),
        shading="flat",
    )
    axes[0].scatter(
        x[~retained],
        y[~retained],
        marker="x",
        color="#D55E00",
        s=22,
        linewidth=1.0,
        label="moves after derating",
    )
    axes[0].set(
        title="Native 12 × 10 MAP-Elites archive",
        xlabel="Throughput niche column",
        ylabel="Peak-import niche row",
        xticks=range(0, 12, 2),
        yticks=range(0, 10, 2),
        xlim=(-0.5, 11.5),
        ylim=(-0.5, 9.5),
    )
    axes[0].legend(loc="upper right")
    colorbar = figure.colorbar(image, ax=axes[0])
    colorbar.set_label("Robust mean LCOE")

    evaluations = np.array([int(row["evaluations"]) for row in progress])
    coverage = np.array([float(row["coverage"]) for row in progress])
    best = np.array([float(row["best_quality"]) for row in progress])
    axes[1].plot(evaluations, 100.0 * coverage, color="#0072B2", lw=2.0, label="coverage")
    axes[1].set(
        title="The accepted pilot authorizes a measured QD run",
        xlabel="Outer candidate calls",
        ylabel="Archive coverage (%)",
        ylim=(0, max(60.0, 1.05 * float((100.0 * coverage).max()))),
    )
    twin = axes[1].twinx()
    twin.plot(evaluations, best, color="#D55E00", lw=1.6, label="best LCOE")
    twin.set_ylabel("Best robust mean LCOE")
    handles, labels = axes[1].get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    axes[1].legend(handles + extra_handles, labels + extra_labels, loc="center right")
    figure.tight_layout()
    save(figure, output)


def mo_pareto(output: Path) -> None:
    data = rows(RESULTS / "mo" / "pareto.csv")
    capex = np.array([float(row["objective_annualized_capex"]) for row in data]) / 1000
    co2 = np.array([float(row["objective_co2_kg"]) for row in data]) / 1000
    curtail = np.array([float(row["objective_curtailed_kwh"]) for row in data]) / 1000
    lcoe = np.array([float(row["mean_lcoe"]) for row in data])
    selected = np.array([row["selected"] == "1" for row in data])
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    scatter = axis.scatter(
        capex,
        co2,
        c=lcoe,
        s=22 + 58 * curtail / max(curtail.max(), 1.0),
        cmap="viridis_r",
        alpha=0.75,
        edgecolor="none",
    )
    axis.scatter(
        capex[selected],
        co2[selected],
        facecolor="none",
        edgecolor="#D55E00",
        linewidth=1.5,
        s=110,
        label="documented extremes",
    )
    axis.set(
        title="MODE retains capital–emissions–curtailment trade-offs",
        xlabel="Annualized CAPEX (thousand currency/year)",
        ylabel="Grid emissions (t CO₂/year)",
    )
    axis.legend()
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Mean robust LCOE")
    figure.tight_layout()
    save(figure, output)


def annual_hydrogen(output: Path) -> None:
    data = rows(RESULTS / "annual" / "dispatch.csv")
    h2 = np.array([float(row["hydrogen_store_kwh"]) for row in data])
    electrolysis = np.array([float(row["electrolyser_kw"]) for row in data])
    bought = np.array([float(row["hydrogen_buy_kw"]) for row in data])
    renewable = np.array([float(row["renewable_kw"]) for row in data])
    load = np.array([float(row["load_kw"]) for row in data])
    days = np.arange(365)
    daily = lambda values: values[: 365 * 24].reshape(365, 24).mean(axis=1)
    figure, axes = plt.subplots(2, 1, figsize=(9.3, 5.0), sharex=True)
    axes[0].plot(days, daily(h2), color="#CC79A7", lw=1.7, label="H₂ store")
    axes[0].fill_between(days, 0, daily(electrolysis), color="#009E73", alpha=0.45, label="electrolyser")
    axes[0].plot(days, daily(bought), color="#D55E00", lw=1.0, label="purchased H₂")
    axes[0].set(title="The selected annual design is replayed at all 8,760 hours", ylabel="Daily mean (kW or kWh)")
    axes[0].legend(ncol=3)
    axes[1].plot(days, daily(renewable), color="#E69F00", lw=1.4, label="renewable")
    axes[1].plot(days, daily(load), color="#222222", lw=1.4, label="electrical load")
    axes[1].set(xlabel="Day of year", ylabel="Daily mean power (kW)")
    axes[1].legend()
    figure.tight_layout()
    save(figure, output)


def budget_accounting(output: Path) -> None:
    formulations = ["landscape", "so", "qd", "mo", "annual"]
    labels = ["Landscape", "SO (3 arms)", "QD", "MODE", "Annual"]
    manifests = [
        json.loads((RESULTS / name / "run.json").read_text(encoding="utf-8"))
        for name in formulations
    ]
    candidates = [manifest["budget"]["candidate_evaluations"] for manifest in manifests]
    solves = [manifest["budget"]["lp_solves"] for manifest in manifests]
    pivots = [manifest["budget"]["simplex_iterations"] for manifest in manifests]
    elapsed = [manifest["elapsed_seconds"] for manifest in manifests]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, candidates, 0.36, label="outer candidates", color="#0072B2")
    axes[0].bar(x + 0.18, solves, 0.36, label="inner LP solves", color="#E69F00")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, rotation=12)
    axes[0].set(title="Evaluation count alone hides nested work", ylabel="Count (log scale)")
    axes[0].legend()
    scatter = axes[1].scatter(
        pivots,
        elapsed,
        s=90,
        c=np.arange(len(labels)),
        cmap="viridis",
    )
    for index, label in enumerate(labels):
        axes[1].annotate(label, (pivots[index], elapsed[index]), xytext=(5, 4), textcoords="offset points")
    axes[1].set(
        title="Pivots and wall time are both reported",
        xlabel="Cumulative simplex pivots",
        ylabel="Wall time (s; loaded machine)",
    )
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    del scatter
    figure.tight_layout()
    save(figure, output)


def render(directory: Path) -> None:
    configure()
    validate_artifacts()
    architecture(directory / "architecture.svg")
    landscape(directory / "landscape.svg")
    dispatch_stack(directory / "dispatch-stack.svg")
    so_comparison(directory / "so-comparison.svg")
    descriptor_pilot(directory / "descriptor-pilot.svg")
    qd_archive(directory / "qd-archive.svg")
    mo_pareto(directory / "mo-pareto.svg")
    annual_hydrogen(directory / "annual-hydrogen.svg")
    budget_accounting(directory / "budget-accounting.svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument(
        "--check",
        action="store_true",
        help="render in a temporary directory and fail if checked-in SVGs differ",
    )
    args = parser.parse_args()
    if args.write:
        render(IMAGES)
        for name in FIGURES:
            print(IMAGES / name)
        return
    with tempfile.TemporaryDirectory() as temporary:
        rendered = Path(temporary)
        render(rendered)
        stale = [
            name
            for name in FIGURES
            if not (IMAGES / name).is_file()
            or not filecmp.cmp(IMAGES / name, rendered / name, shallow=False)
        ]
    if stale:
        raise SystemExit(
            "missing or stale energy-hub-bilevel figures:\n"
            + "\n".join(str(IMAGES / name) for name in stale)
        )
    print("energy-hub-bilevel figures are current")


if __name__ == "__main__":
    main()
