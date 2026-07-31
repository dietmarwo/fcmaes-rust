#!/usr/bin/env python3
"""Render deterministic water-network-scheduling SVG figures."""

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
    "network-topology.svg",
    "control-precedence.svg",
    "so-comparison.svg",
    "pump-schedule.svg",
    "hydraulic-trace.svg",
    "scenario-stress.svg",
    "resolution-study.svg",
    "descriptor-gate.svg",
    "qd-catalogue.svg",
    "mo-pareto.svg",
    "parallelism.svg",
]
COLORS = {"seed": "#999999", "cma": "#0072B2", "de": "#009E73", "bite": "#D55E00"}


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
            "svg.hashsalt": "fcmaes-water-network-scheduling-v1",
        }
    )


def validate_artifacts() -> None:
    validation = json.loads(
        (RESULTS / "validation" / "run.json").read_text(encoding="utf-8")
    )
    checks = validation["checks"]
    if checks["failed_at_step"] is not None:
        raise ValueError("publication validation contains failed hydraulic steps")
    if checks["max_continuity_residual_m3_s"] > 1.0e-6:
        raise ValueError("continuity residual exceeds the frozen tolerance")
    if not 0.0 < checks["energy_oracle_max_relative_error"] < 1.0e-6:
        raise ValueError("offline energy oracle is either identity or outside tolerance")
    if checks["analytic_pipe_relative_error"] > 1.0e-5:
        raise ValueError("laminar analytic check exceeds the frozen tolerance")
    if checks["override_witness_steps"] == 0:
        raise ValueError("threshold witness did not exercise the safety override")
    pilot = json.loads((RESULTS / "pilot" / "run.json").read_text(encoding="utf-8"))
    if pilot["qd_decision"] != "rejected":
        raise ValueError("publication figures expect the measured rejected gate")
    qd = json.loads((RESULTS / "qd" / "run.json").read_text(encoding="utf-8"))
    if qd.get("status") != "skipped" or qd.get("actual_evaluations") is not None:
        raise ValueError("rejected QD arm must be recorded as skipped")
    if not all(row["feasible"] == "true" for row in rows(RESULTS / "so" / "arms.csv")):
        raise ValueError("scalar publication retained an infeasible plan")
    if rows(RESULTS / "so" / "arms.csv")[0]["arm"] != "seed":
        raise ValueError("scalar comparison does not publish its seed baseline")


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "water-network-scheduling/plot_results.py"},
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
        arrowprops={"arrowstyle": "->", "color": "#455A64", "lw": 1.3},
    )


def architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.6, 3.8))
    axis.set(xlim=(0, 10.6), ylim=(0, 3.8))
    axis.axis("off")
    box(axis, 0.15, 1.35, 1.55, 0.9, "28 normalized\ncoordinates", "#E3F2FD")
    box(axis, 2.05, 1.35, 1.65, 0.9, "quantized speeds\nthreshold decoder", "#FFF3E0")
    box(axis, 4.05, 1.35, 1.55, 0.9, "independent\nEPANET-RS EPS", "#E8F5E9")
    box(axis, 5.95, 1.35, 1.55, 0.9, "SI energy +\nconstraint replay", "#F3E5F5")
    box(axis, 7.9, 2.0, 2.35, 0.7, "CMA · DE · BiteOpt\ncandidate parallelism", "#FFEBEE")
    box(axis, 7.9, 0.85, 2.35, 0.7, "MODE · pilot gate\nMAP-Elites catalogue", "#E0F2F1")
    for start, end in [
        ((1.70, 1.8), (2.05, 1.8)),
        ((3.70, 1.8), (4.05, 1.8)),
        ((5.60, 1.8), (5.95, 1.8)),
        ((7.50, 1.8), (7.90, 2.35)),
        ((7.50, 1.8), (7.90, 1.2)),
    ]:
        arrow(axis, start, end)
    axis.text(
        5.3,
        3.35,
        "One owner of parallelism; one explicit boundary between hydraulics and economics",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        5.3,
        0.32,
        "No generated RULES · no hidden ENERGY report · typed DDA/PDA scenarios",
        ha="center",
        color="#37474F",
    )
    save(figure, output)


def parse_network() -> tuple[dict[str, tuple[float, float]], list[tuple[str, str, str]]]:
    sections: dict[str, list[list[str]]] = {}
    section = ""
    for raw in (ROOT / "network" / "synthetic-zone.inp").read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[]").upper()
            sections.setdefault(section, [])
        else:
            sections.setdefault(section, []).append(line.split())
    coords = {part[0]: (float(part[1]), float(part[2])) for part in sections["COORDINATES"]}
    links: list[tuple[str, str, str]] = []
    for kind in ("PIPES", "PUMPS", "VALVES"):
        links.extend((part[1], part[2], kind) for part in sections[kind])
    return coords, links


def topology(output: Path) -> None:
    coords, links = parse_network()
    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    for start, end, kind in links:
        x = [coords[start][0], coords[end][0]]
        y = [coords[start][1], coords[end][1]]
        color = {"PIPES": "#90A4AE", "PUMPS": "#D55E00", "VALVES": "#CC79A7"}[kind]
        axis.plot(x, y, color=color, lw=2.5 if kind != "PIPES" else 1.0, alpha=0.9)
    names = list(coords)
    elevations = {
        f"J{i:02}": 32 + i * 0.8 for i in range(1, 21)
    } | {"R1": 43, "H1": 43, "T1": 70, "VU": 42, "VD": 44}
    values = [elevations.get(name, 40) for name in names]
    scatter = axis.scatter(
        [coords[name][0] for name in names],
        [coords[name][1] for name in names],
        c=values,
        cmap="viridis",
        s=[95 if name in {"R1", "T1"} else 38 for name in names],
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    for name in ("R1", "T1", "PRV1", "J01", "J10", "J20"):
        if name in coords:
            axis.text(coords[name][0], coords[name][1] + 15, name, ha="center", fontsize=8)
    axis.set_title("Synthetic single-zone network (node colour: elevation)")
    axis.set_xlabel("synthetic map x")
    axis.set_ylabel("synthetic map y")
    axis.set_aspect("equal")
    figure.colorbar(scatter, ax=axis, label="elevation (m)", shrink=0.75)
    save(figure, output)


def control_precedence(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.6, 3.8))
    axis.set(xlim=(0, 9.6), ylim=(0, 3.8))
    axis.axis("off")
    box(axis, 0.2, 1.35, 1.55, 0.9, "read tank level\nand 2 h period", "#E3F2FD")
    box(axis, 2.15, 2.35, 2.1, 0.75, "level ≥ high\nboth pumps OFF", "#FFEBEE")
    box(axis, 2.15, 1.35, 2.1, 0.75, "level ≤ low\npriority pump ≥ 0.8", "#FFF3E0")
    box(axis, 2.15, 0.35, 2.1, 0.75, "inside band\nschedule applies", "#E8F5E9")
    box(axis, 4.85, 1.35, 1.75, 0.9, "set state status,\nspeed and PRV", "#F3E5F5")
    box(axis, 7.2, 1.35, 2.05, 0.9, "solve one step\nrecord SI trace", "#E0F2F1")
    for target in ((2.15, 2.72), (2.15, 1.72), (2.15, 0.72)):
        arrow(axis, (1.75, 1.8), target)
    for start in ((4.25, 2.72), (4.25, 1.72), (4.25, 0.72)):
        arrow(axis, start, (4.85, 1.8))
    arrow(axis, (6.60, 1.8), (7.20, 1.8))
    axis.text(4.8, 3.5, "Safety overrides have explicit precedence", ha="center", weight="bold")
    save(figure, output)


def so_comparison(output: Path) -> None:
    data = rows(RESULTS / "so" / "arms.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    labels = [row["arm"] for row in data]
    objective = [float(row["objective"]) for row in data]
    wall = [float(row["elapsed_seconds"]) for row in data]
    colors = [COLORS[label] for label in labels]
    axes[0].bar(labels, objective, color=colors)
    axes[0].set(ylabel="robust cost", title="Equal requested budgets")
    axes[0].set_ylim(min(objective) - 2, max(objective) + 2)
    axes[1].bar(labels, wall, color=colors)
    axes[1].set(ylabel="wall time (s)", title="Measured optimizer time")
    figure.suptitle("Scalar robust scheduling: the seed baseline is explicit", weight="bold")
    save(figure, output)


def pump_schedule(output: Path) -> None:
    arm_rows = rows(RESULTS / "so" / "arms.csv")
    optimized = [row for row in arm_rows if row["arm"] != "seed"]
    best = min(optimized, key=lambda row: float(row["objective"]))["arm"]
    data = [
        row
        for row in rows(RESULTS / "so" / "schedule.csv")
        if row["arm"] == best
    ]
    figure, axis = plt.subplots(figsize=(9.8, 4.0))
    for period in range(12):
        hour = period * 2
        tariff = 0.09 if hour <= 5 else 0.31 if 16 <= hour <= 20 else 0.16
        axis.axvspan(hour, hour + 2, color="#E0E0E0", alpha=0.25 + tariff)
    for pump, color in ((1, "#0072B2"), (2, "#D55E00")):
        selected = [row for row in data if int(row["pump"]) == pump]
        x = [int(row["start_hour"]) for row in selected] + [24]
        y = [float(row["relative_speed"]) for row in selected]
        axis.step(x, y + [y[-1]], where="post", label=f"pump {pump}", color=color, lw=2)
    axis.set(
        xlim=(0, 24),
        ylim=(-0.05, 1.08),
        xlabel="clock hour",
        ylabel="relative speed",
        title=f"Best scalar schedule ({best}); darker bands have higher tariffs",
    )
    axis.legend()
    save(figure, output)


def hydraulic_trace(output: Path) -> None:
    data = rows(RESULTS / "validation" / "trace.csv")
    witness = rows(RESULTS / "validation" / "override_trace.csv")
    time = [float(row["time_s"]) / 3600 for row in data]
    witness_time = [float(row["time_s"]) / 3600 for row in witness]
    figure, axes = plt.subplots(2, 1, figsize=(9.6, 5.7), sharex=True)
    axes[0].plot(
        time,
        [float(row["tank_level_m"]) for row in data],
        color="#0072B2",
        lw=2,
        label="baseline",
    )
    axes[0].plot(
        witness_time,
        [float(row["tank_level_m"]) for row in witness],
        color="#D55E00",
        lw=1.4,
        label="threshold witness",
    )
    overridden = [row["safety_override"] == "true" for row in witness]
    axes[0].scatter(
        np.array(witness_time)[overridden],
        np.array([float(row["tank_level_m"]) for row in witness])[overridden],
        marker="x",
        color="#D55E00",
        s=20,
        label="override active",
    )
    axes[0].axhspan(1, 10, color="#E8F5E9", alpha=0.5)
    axes[0].set(ylabel="tank level (m)", title="Baseline and positive override witness")
    axes[0].legend(ncol=3)
    axes[1].fill_between(
        time,
        [float(row["min_pressure_m"]) for row in data],
        [float(row["max_pressure_m"]) for row in data],
        color="#56B4E9",
        alpha=0.4,
        label="junction envelope",
    )
    axes[1].axhline(20, color="#D55E00", ls="--", label="DDA requirement")
    axes[1].set(xlabel="clock hour", ylabel="pressure (m)")
    axes[1].legend()
    save(figure, output)


def scenario_stress(output: Path) -> None:
    data = rows(RESULTS / "scenarios" / "scenario_metrics.csv")
    labels = [row["scenario"].replace("_", " ") for row in data]
    pressure = [float(row["min_pressure_m"]) for row in data]
    colors = ["#CC79A7" if row["analysis_type"] == "PDA" else "#0072B2" for row in data]
    figure, axis = plt.subplots(figsize=(10.4, 5.3))
    axis.barh(np.arange(len(data)), pressure, color=colors)
    axis.axvline(20, color="#D55E00", ls="--", label="DDA pressure target")
    axis.set_yticks(np.arange(len(data)), labels)
    axis.invert_yaxis()
    axis.set(xlabel="minimum pressure (m)", title="Named scenario stress (magenta = PDA holdout)")
    axis.legend()
    save(figure, output)


def resolution_study(output: Path) -> None:
    data = rows(RESULTS / "resolution" / "resolution_study.csv")
    baseline = [row for row in data if row["case"] == "baseline"]
    witness = [row for row in data if row["case"] == "override-witness"]
    timestep = np.array([float(row["hydraulic_timestep_s"]) / 60 for row in baseline])
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    axes[0].plot(
        timestep,
        [float(row["energy_kwh"]) for row in baseline],
        "o-",
        color="#0072B2",
        label="baseline",
    )
    axes[0].plot(
        timestep,
        [float(row["energy_kwh"]) for row in witness],
        "s--",
        color="#D55E00",
        label="override witness",
    )
    axes[0].set(xlabel="hydraulic step (min)", ylabel="energy (kWh)", title="Integral convergence")
    axes[0].invert_xaxis()
    axes[0].legend()
    axes[1].plot(
        timestep,
        [float(row["peak_kw_hourly"]) for row in baseline],
        "o-",
        label="fixed 1 h billing",
        color="#009E73",
    )
    axes[1].plot(
        timestep,
        [float(row["peak_kw_native"]) for row in baseline],
        "s--",
        label="native sample",
        color="#D55E00",
    )
    axes[1].set(xlabel="hydraulic step (min)", ylabel="peak power (kW)", title="Resolution is explicit")
    axes[1].invert_xaxis()
    axes[1].legend()
    axes[2].plot(
        timestep,
        [int(row["override_steps"]) for row in witness],
        "o-",
        color="#D55E00",
        label="override steps",
    )
    axes[2].plot(
        timestep,
        [int(row["starts"]) for row in witness],
        "s--",
        color="#CC79A7",
        label="pump starts",
    )
    axes[2].set(
        xlabel="hydraulic step (min)",
        ylabel="count",
        title="Threshold events are resolution-sensitive",
    )
    axes[2].invert_xaxis()
    axes[2].legend()
    save(figure, output)


def descriptor_gate(output: Path) -> None:
    data = rows(RESULTS / "pilot" / "pilot.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9.7, 4.1))
    x = np.array([float(row["d1_axis1_train"]) for row in data])
    y = np.array([float(row["d1_axis2_train"]) for row in data])
    held_x = np.array([float(row["d1_axis1_holdout"]) for row in data])
    held_y = np.array([float(row["d1_axis2_holdout"]) for row in data])
    scatter = axes[0].scatter(
        x,
        y,
        c=[float(row["operating_cost"]) for row in data],
        cmap="viridis_r",
        s=14,
        alpha=0.75,
    )
    for index in np.linspace(0, len(data) - 1, min(36, len(data)), dtype=int):
        axes[0].plot(
            [x[index], held_x[index]],
            [y[index], held_y[index]],
            color="#D55E00",
            alpha=0.28,
            lw=0.8,
        )
    pilot = json.loads((RESULTS / "pilot" / "run.json").read_text())
    d1 = pilot["pairs"]["D1"]
    axes[0].set(
        xlim=(0.15, 0.35),
        ylim=(0.08, 0.23),
        xlabel="off-peak energy fraction",
        ylabel="tank turnover",
        title="Training → unseen-demand movement",
    )
    values = [
        abs(d1["rank_correlation"]),
        max(d1["clipping_axis_1"], d1["clipping_axis_2"]),
        d1["coverage"],
        d1["holdout_niche_retention"],
        d1["timestep_retention"],
    ]
    limits = [0.7, 0.1, 0.4, 0.6, None]
    passing = [
        values[0] < 0.7,
        values[1] < 0.1,
        values[2] > 0.4,
        values[3] > 0.6,
        None,
    ]
    labels = ["|ρ|", "clipping", "coverage", "holdout\nretention", "timestep\nretention"]
    axes[1].bar(
        labels,
        values,
        color=[
            "#999999" if passed is None else "#009E73" if passed else "#D55E00"
            for passed in passing
        ],
    )
    for index, limit in enumerate(limits):
        if limit is not None:
            axes[1].plot(
                [index - 0.38, index + 0.38],
                [limit, limit],
                color="#222222",
                ls="--",
                lw=1.0,
            )
    axes[1].set(
        ylim=(0, 1.0),
        ylabel="fraction or absolute correlation",
        title="D1 rejected on the native 10 × 10 grid",
    )
    figure.colorbar(scatter, ax=axes[0], label="robust operating cost")
    save(figure, output)


def qd_catalogue(output: Path) -> None:
    manifest = json.loads((RESULTS / "qd" / "run.json").read_text())
    if manifest.get("status") == "skipped":
        pilot = json.loads((RESULTS / "pilot" / "run.json").read_text())
        figure, axis = plt.subplots(figsize=(7.3, 4.4))
        names = ["D1", "D2"]
        coverage = [pilot["pairs"][name]["coverage"] for name in names]
        retention = [pilot["pairs"][name]["holdout_niche_retention"] for name in names]
        x = np.arange(len(names))
        axis.bar(x - 0.18, coverage, 0.36, label="coverage", color="#0072B2")
        axis.bar(x + 0.18, retention, 0.36, label="holdout retention", color="#D55E00")
        axis.axhline(0.4, color="#0072B2", ls="--", lw=1.0)
        axis.axhline(0.6, color="#D55E00", ls="--", lw=1.0)
        axis.set_xticks(x, names)
        axis.set(
            ylim=(0, 1.0),
            ylabel="measured fraction",
            title="QD was not executed: neither emergent pair clears both gates",
        )
        axis.legend()
        save(figure, output)
        return
    data = rows(RESULTS / "qd" / "qd_archive.csv")
    figure, axis = plt.subplots(figsize=(7.3, 5.0))
    scatter = axis.scatter(
        [float(row["descriptor_1"]) for row in data],
        [float(row["descriptor_2"]) for row in data],
        c=[float(row["quality"]) for row in data],
        cmap="viridis_r",
        s=42,
        edgecolor="white",
        linewidth=0.35,
    )
    axis.set(
        xlim=(0.15, 0.35),
        ylim=(0.08, 0.23),
        xlabel="off-peak energy fraction",
        ylabel="tank turnover",
        title=f"MAP-Elites strategy catalogue: {len(data)}/100 niches",
    )
    figure.colorbar(scatter, ax=axis, label="robust operating cost")
    save(figure, output)


def mo_pareto(output: Path) -> None:
    data = rows(RESULTS / "mo" / "mo_pareto.csv")
    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    scatter = axis.scatter(
        [float(row["energy_cost"]) for row in data],
        [float(row["switching_cost"]) for row in data],
        c=[float(row["excess_pressure"]) for row in data],
        cmap="plasma",
        s=[60 if row["selected"] == "true" else 20 for row in data],
        alpha=0.8,
    )
    axis.set(
        xlabel="worst energy cost",
        ylabel="switching cost",
        title=f"Constrained MODE: {len(data)} nondominated schedules",
    )
    figure.colorbar(scatter, ax=axis, label="excess-pressure proxy")
    save(figure, output)


def parallelism(output: Path) -> None:
    data = rows(RESULTS / "benchmark" / "parallelism_benchmark.csv")
    labels = ["candidate\nparallel" if row["arrangement"].startswith("candidate") else "internal EPS\nparallel" for row in data]
    throughput = [float(row["candidates_per_second"]) for row in data]
    figure, axis = plt.subplots(figsize=(6.5, 4.1))
    axis.bar(labels, throughput, color=["#009E73", "#CC79A7"])
    axis.set(ylabel="candidates / second", title="Equal-work tank-free benchmark")
    ratio = throughput[0] / throughput[1]
    axis.text(0.5, max(throughput) * 0.9, f"{ratio:.1f}×", ha="center", weight="bold")
    save(figure, output)


RENDERERS = {
    "architecture.svg": architecture,
    "network-topology.svg": topology,
    "control-precedence.svg": control_precedence,
    "so-comparison.svg": so_comparison,
    "pump-schedule.svg": pump_schedule,
    "hydraulic-trace.svg": hydraulic_trace,
    "scenario-stress.svg": scenario_stress,
    "resolution-study.svg": resolution_study,
    "descriptor-gate.svg": descriptor_gate,
    "qd-catalogue.svg": qd_catalogue,
    "mo-pareto.svg": mo_pareto,
    "parallelism.svg": parallelism,
}


def render(directory: Path) -> None:
    configure()
    validate_artifacts()
    for name in FIGURES:
        RENDERERS[name](directory / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        with tempfile.TemporaryDirectory() as temporary:
            rendered = Path(temporary)
            render(rendered)
            stale = [
                name
                for name in FIGURES
                if not (IMAGES / name).is_file()
                or not filecmp.cmp(rendered / name, IMAGES / name, shallow=False)
            ]
            if stale:
                print("missing or stale water-network-scheduling figures:")
                for name in stale:
                    print(IMAGES / name)
                return 1
        print("water-network-scheduling figures are current")
        return 0
    render(IMAGES)
    for name in FIGURES:
        print(IMAGES / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
