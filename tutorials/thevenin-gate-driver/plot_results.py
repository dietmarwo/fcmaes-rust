#!/usr/bin/env python3
"""Render deterministic figures from the checked-in gate-driver evidence."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "publication"
IMAGES = ROOT / "images"


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
            "svg.hashsalt": "fcmaes-thevenin-gate-driver-v1",
        }
    )


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={
            "Date": None,
            "Creator": "thevenin-gate-driver/plot_results.py",
        },
    )
    plt.close(figure)
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n",
        encoding="utf-8",
    )


def validate_artifacts() -> None:
    manifest_path = RESULTS / "mo" / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
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
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"{manifest_path} lacks fields {sorted(missing)}")
    if manifest["schema_version"] != 1:
        raise ValueError("MODE manifest does not use schema version 1")
    if manifest["tutorial"] != "thevenin-gate-driver":
        raise ValueError("MODE manifest names the wrong tutorial")
    for artifact in manifest["artifacts"].values():
        if not (manifest_path.parent / artifact).is_file():
            raise ValueError(f"MODE manifest references missing {artifact}")
    pareto = rows(RESULTS / "mo" / "pareto.csv")
    if len(pareto) != manifest["pareto_points"]:
        raise ValueError("Pareto CSV and manifest counts differ")
    if any(
        float(row["constraint_peak_current_a"]) > 0.0
        or float(row["constraint_settling_time_ns"]) > 0.0
        for row in pareto
    ):
        raise ValueError("Pareto artifact contains an infeasible design")
    if sum(int(row["selected"]) for row in pareto) < 2:
        raise ValueError("Pareto artifact lacks selected representatives")

    validation = RESULTS / "validation"
    summary = json.loads((validation / "summary.json").read_text(encoding="utf-8"))
    if not summary["passed"] or summary["rows"] != 49:
        raise ValueError("cross-simulator publication gate did not pass")
    comparison = rows(validation / "comparison.csv")
    if len(comparison) != summary["rows"]:
        raise ValueError("comparison CSV and summary counts differ")
    for metric, values in summary["metrics"].items():
        observed = max(float(row[metric]) for row in comparison)
        if not math.isclose(observed, values["maximum"], rel_tol=1e-12):
            raise ValueError(f"summary maximum for {metric} is stale")
        if observed > values["limit"]:
            raise ValueError(f"cross-simulator gate failed for {metric}")

    timestep = rows(validation / "timestep.csv")
    by_design: dict[int, dict[float, dict[str, str]]] = defaultdict(dict)
    for row in timestep:
        by_design[int(row["design_id"])][float(row["step_s"])] = row
    convergence_limits = {
        "rise_time_ns": 0.01,
        "overshoot_percent": 0.01,
        "peak_driver_current_a": 0.01,
        "settling_time_ns": 0.1,
        "final_gate_voltage_v": 0.01,
    }
    for design, steps in by_design.items():
        if 50e-12 not in steps or 25e-12 not in steps:
            raise ValueError(f"timestep study lacks refinement for design {design}")
        for metric, limit in convergence_limits.items():
            difference = abs(
                float(steps[50e-12][metric]) - float(steps[25e-12][metric])
            )
            if difference > limit:
                raise ValueError(
                    f"timestep gate failed for design {design}, {metric}"
                )

    scaling = rows(validation / "scaling.csv")
    if not scaling or any(int(row["failures"]) for row in scaling):
        raise ValueError("scaling evidence is missing or contains failures")


def optimization_figure(output: Path) -> None:
    pareto = rows(RESULTS / "mo" / "pareto.csv")
    waveforms = rows(RESULTS / "mo" / "waveforms.csv")
    rise = np.array([float(row["objective_rise_time_ns"]) for row in pareto])
    overshoot = np.array(
        [float(row["objective_overshoot_percent"]) for row in pareto]
    )
    snubber = np.array([float(row["snubber_resistance_ohm"]) for row in pareto])
    selected = np.array([bool(int(row["selected"])) for row in pareto])

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    scatter = axes[0].scatter(
        rise,
        overshoot,
        c=snubber,
        cmap="viridis",
        s=28,
        edgecolor="#263238",
        linewidth=0.25,
    )
    axes[0].scatter(
        rise[selected],
        overshoot[selected],
        marker="*",
        s=165,
        facecolor="#F9A825",
        edgecolor="#5D4037",
        linewidth=0.8,
        label="reported representatives",
        zorder=4,
    )
    axes[0].set(
        title="Feasible rise-time / overshoot frontier",
        xlabel="10–90% rise time (ns)",
        ylabel="overshoot (%)",
    )
    axes[0].legend(loc="best")
    colorbar = figure.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("snubber resistance (Ω)")

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in waveforms:
        grouped[int(row["point_id"])].append(row)
    colors = ["#0072B2", "#D55E00", "#009E73", "#7E57C2"]
    for color, (point_id, waveform) in zip(colors, sorted(grouped.items()), strict=False):
        representative = pareto[point_id]
        axes[1].plot(
            [float(row["time_ns"]) for row in waveform],
            [float(row["gate_v"]) for row in waveform],
            color=color,
            linewidth=1.35,
            label=(
                f"{float(representative['objective_rise_time_ns']):.2f} ns, "
                f"{float(representative['objective_overshoot_percent']):.2f}%"
            ),
        )
    axes[1].axhline(10.0, color="#263238", linewidth=0.9, linestyle="--")
    axes[1].fill_between(
        [0.0, 120.0],
        [9.8, 9.8],
        [10.2, 10.2],
        color="#90CAF9",
        alpha=0.18,
        label="±2% settling band",
    )
    axes[1].set(
        title="Replayed selected gate waveforms",
        xlabel="time (ns)",
        ylabel="gate voltage (V)",
        xlim=(0.0, 85.0),
    )
    axes[1].legend(loc="best")
    figure.tight_layout()
    save(figure, output)


def validation_figure(output: Path) -> None:
    comparison = rows(RESULTS / "validation" / "comparison.csv")
    summary = json.loads(
        (RESULTS / "validation" / "summary.json").read_text(encoding="utf-8")
    )
    timestep = rows(RESULTS / "validation" / "timestep.csv")
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))

    thevenin_rise = np.array(
        [float(row["thevenin_rise_time_ns"]) for row in comparison]
    )
    ngspice_rise = np.array(
        [float(row["ngspice_rise_time_ns"]) for row in comparison]
    )
    maximum = max(thevenin_rise.max(), ngspice_rise.max())
    axes[0].plot([0, maximum], [0, maximum], color="#546E7A", linestyle="--")
    axes[0].scatter(
        ngspice_rise,
        thevenin_rise,
        c=[float(row["u_snubber"]) for row in comparison],
        cmap="plasma",
        s=26,
        edgecolor="#263238",
        linewidth=0.25,
    )
    axes[0].set(
        title="49-design reference grid",
        xlabel="ngspice rise time (ns)",
        ylabel="thevenin rise time (ns)",
        aspect="equal",
    )

    metric_order = [
        "rise_time_abs_ns",
        "overshoot_abs_percentage_points",
        "peak_current_abs_a",
        "settling_time_abs_ns",
        "final_voltage_abs_v",
    ]
    labels = ["rise\n(ns)", "overshoot\n(pp)", "current\n(A)", "settling\n(ns)", "final V\n(V)"]
    ratios = [
        summary["metrics"][metric]["maximum"]
        / summary["metrics"][metric]["limit"]
        for metric in metric_order
    ]
    axes[1].bar(labels, ratios, color="#009E73", edgecolor="#004D40", linewidth=0.5)
    axes[1].axhline(1.0, color="#D55E00", linestyle="--", linewidth=1.2)
    axes[1].set(
        title="Maximum error / acceptance limit",
        ylabel="fraction of limit",
        ylim=(0.0, 1.08),
    )

    colors = ["#0072B2", "#D55E00", "#009E73"]
    for design_id, color in enumerate(colors):
        design_rows = sorted(
            (
                row
                for row in timestep
                if int(row["design_id"]) == design_id
            ),
            key=lambda row: float(row["step_s"]),
            reverse=True,
        )
        axes[2].plot(
            [float(row["step_s"]) * 1e12 for row in design_rows],
            [float(row["rise_time_ns"]) for row in design_rows],
            marker="o",
            color=color,
            label=f"design {design_id}",
        )
    axes[2].set(
        title="Maximum-timestep refinement",
        xlabel="maximum timestep (ps)",
        ylabel="rise time (ns)",
        xscale="log",
    )
    axes[2].invert_xaxis()
    axes[2].legend(loc="best")
    figure.tight_layout()
    save(figure, output)


def scaling_figure(output: Path) -> None:
    scaling = rows(RESULTS / "validation" / "scaling.csv")
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in scaling:
        grouped[int(row["workers"])].append(
            float(row["evaluations_per_second"])
        )
    workers = sorted(grouped)
    medians = [float(np.median(grouped[count])) for count in workers]
    serial = medians[0]
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    axis.plot(workers, medians, color="#0072B2", marker="o", linewidth=1.6)
    for count in workers:
        axis.scatter(
            [count] * len(grouped[count]),
            grouped[count],
            color="#90CAF9",
            edgecolor="#0D47A1",
            linewidth=0.35,
            s=28,
            zorder=3,
        )
    for count, median in zip(workers, medians, strict=True):
        axis.annotate(
            f"{median / serial:.2f}×",
            (count, median),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    axis.set(
        title="Parallel candidate evaluation",
        xlabel="fcmaes worker threads",
        ylabel="transient evaluations / second",
        xticks=workers,
    )
    figure.tight_layout()
    save(figure, output)


def render(directory: Path) -> list[Path]:
    configure()
    validate_artifacts()
    outputs = [
        directory / "optimization.svg",
        directory / "validation.svg",
        directory / "scaling.svg",
    ]
    optimization_figure(outputs[0])
    validation_figure(outputs[1])
    scaling_figure(outputs[2])
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        for path in render(IMAGES):
            print(f"wrote {path.relative_to(ROOT)}")
        return 0
    with tempfile.TemporaryDirectory() as temporary:
        generated = render(Path(temporary))
        stale = [
            IMAGES / path.name
            for path in generated
            if not (IMAGES / path.name).is_file()
            or not filecmp.cmp(path, IMAGES / path.name, shallow=False)
        ]
    if stale:
        print("missing or stale figures:")
        for path in stale:
            print(path)
        return 1
    print("figures match checked-in evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

