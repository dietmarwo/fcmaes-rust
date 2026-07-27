#!/usr/bin/env python3
"""Render deterministic publication figures from the checked-in CSV evidence."""

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
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "publication"
IMAGES = ROOT / "images"
COLORS = {"cma": "#0072B2", "de": "#009E73", "bite": "#D55E00"}
LABELS = {"cma": "CMA-ES retry", "de": "DE retry", "bite": "BiteOpt retry"}
REQUIRED_MANIFEST_FIELDS = {
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
            "svg.hashsalt": "fcmaes-sindr-circuit-design-v1",
        }
    )


def validate_artifacts() -> None:
    for path in RESULTS.glob("*/run.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        missing = REQUIRED_MANIFEST_FIELDS - manifest.keys()
        if missing:
            raise ValueError(f"{path} lacks required fields: {sorted(missing)}")
        if manifest["schema_version"] != 1:
            raise ValueError(f"{path} does not use result schema version 1")
        if manifest["tutorial"] != "sindr-circuit-design":
            raise ValueError(f"{path} names the wrong tutorial")
        for artifact in manifest["artifacts"].values():
            if not (path.parent / artifact).is_file():
                raise ValueError(f"{path} references missing artifact {artifact}")

    qd = json.loads((RESULTS / "qd" / "run.json").read_text(encoding="utf-8"))
    archive = rows(RESULTS / "qd" / "archive.csv")
    if len(archive) != qd["qd"]["occupied"]:
        raise ValueError("QD manifest occupied count disagrees with archive.csv")
    lower = qd["qd"]["descriptor_lower"]
    upper = qd["qd"]["descriptor_upper"]
    for row in archive:
        descriptor = [
            float(row["descriptor_log10_f0"]),
            float(row["descriptor_peak_gain_db"]),
        ]
        if not all(
            low <= value <= high
            for value, low, high in zip(descriptor, lower, upper, strict=True)
        ):
            raise ValueError("QD archive contains an out-of-range descriptor")
    if qd["ac_solves"] != (
        qd["optimization_ac_solves"] + qd["range_study_ac_solves"]
    ):
        raise ValueError("QD physical solve accounting does not reconcile")

    mo = json.loads((RESULTS / "mo" / "run.json").read_text(encoding="utf-8"))
    pareto = rows(RESULTS / "mo" / "pareto.csv")
    if len(pareto) != mo["pareto_points"]:
        raise ValueError("MODE manifest Pareto count disagrees with pareto.csv")
    if any(float(row["constraint_peak_db"]) > 0.0 for row in pareto):
        raise ValueError("MODE Pareto artifact contains an infeasible point")


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "sindr-circuit-design/plot_results.py"},
    )
    plt.close(figure)
    # Matplotlib emits spaces before newlines in path data. Normalize those so
    # `git diff --check` remains useful while preserving deterministic SVGs.
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n",
        encoding="utf-8",
    )


def parabolic_peak(frequencies: np.ndarray, gains: np.ndarray) -> tuple[float, float]:
    index = int(np.argmax(gains))
    x0, x1, x2 = np.log(frequencies[index - 1 : index + 2])
    y0, y1, y2 = gains[index - 1 : index + 2]
    shift = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
    peak_log_frequency = x1 + 0.5 * shift * (x2 - x0)
    peak_gain = y1 - 0.25 * (y0 - y2) * shift
    return math.exp(peak_log_frequency), peak_gain


def feature_figure(output: Path) -> None:
    curve = rows(RESULTS / "so" / "feature_curve.csv")
    smoothness = rows(RESULTS / "so" / "feature_smoothness.csv")
    frequencies = np.array([float(row["frequency_hz"]) for row in curve])
    gains = np.array([float(row["gain_db"]) for row in curve])
    grid_index = int(np.argmax(gains))
    smooth_frequency, smooth_gain = parabolic_peak(frequencies, gains)

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    axes[0].semilogx(frequencies, gains, color="#455A64", linewidth=1.4)
    axes[0].scatter(
        frequencies,
        gains,
        s=22,
        facecolor="#ECEFF1",
        edgecolor="#37474F",
        linewidth=0.6,
        zorder=3,
        label="AC grid",
    )
    axes[0].scatter(
        [frequencies[grid_index]],
        [gains[grid_index]],
        marker="s",
        s=75,
        facecolor="#D55E00",
        edgecolor="#6D2C00",
        zorder=4,
        label=f"Grid maximum: {frequencies[grid_index] / 1e3:.2f} kHz",
    )
    axes[0].scatter(
        [smooth_frequency],
        [smooth_gain],
        marker="*",
        s=150,
        facecolor="#009E73",
        edgecolor="#004D40",
        zorder=5,
        label=f"Interpolated: {smooth_frequency / 1e3:.2f} kHz",
    )
    axes[0].axhline(smooth_gain - 3.0103, color="#7E57C2", linestyle="--", linewidth=1)
    axes[0].set(
        title="Features come from the curve, not grid indices",
        xlabel="Frequency (Hz)",
        ylabel="Gain (dB)",
    )
    axes[0].legend(loc="best")

    r1 = np.array([float(row["r1_ohm"]) for row in smoothness])
    grid = np.array([float(row["grid_peak_hz"]) for row in smoothness])
    interpolated = np.array(
        [float(row["interpolated_peak_hz"]) for row in smoothness]
    )
    axes[1].step(
        r1,
        grid / 1e3,
        where="mid",
        color="#D55E00",
        linewidth=1.7,
        label="Grid arg-max",
    )
    axes[1].plot(
        r1,
        interpolated / 1e3,
        color="#009E73",
        marker="o",
        markersize=3,
        linewidth=1.7,
        label="Log-frequency parabola",
    )
    axes[1].set(
        title="Interpolation removes the staircase objective",
        xlabel="R1 (Ω)",
        ylabel="Extracted centre frequency (kHz)",
    )
    axes[1].legend(loc="best")
    figure.suptitle("AC feature extraction for optimization", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def so_figure(output: Path) -> None:
    convergence = rows(RESULTS / "so" / "convergence.csv")
    best = {row["optimizer"]: row for row in rows(RESULTS / "so" / "best.csv")}
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for optimizer in ("cma", "de", "bite"):
        selected = [row for row in convergence if row["optimizer"] == optimizer]
        evaluations = [int(row["evaluations"]) for row in selected]
        values = [float(row["best_objective"]) for row in selected]
        axes[0].step(
            evaluations,
            values,
            where="post",
            color=COLORS[optimizer],
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=LABELS[optimizer],
        )
    axes[0].set_yscale("log")
    axes[0].set(
        title="Best completed-retry result",
        xlabel="Objective evaluations",
        ylabel="Weighted target error (lower is better)",
    )
    axes[0].legend(loc="best")

    optimizers = ["cma", "de", "bite"]
    frequency_error = [
        100.0 * abs(float(best[name]["peak_hz"]) / 10_000.0 - 1.0)
        for name in optimizers
    ]
    q_error = [
        100.0 * abs(float(best[name]["q"]) / 5.0 - 1.0) for name in optimizers
    ]
    locations = np.arange(len(optimizers))
    width = 0.36
    axes[1].bar(
        locations - width / 2,
        frequency_error,
        width,
        color="#56B4E9",
        label="Centre-frequency error",
    )
    axes[1].bar(
        locations + width / 2,
        q_error,
        width,
        color="#CC79A7",
        label="Q error",
    )
    axes[1].set_xticks(locations, [LABELS[name] for name in optimizers], rotation=12)
    axes[1].set(
        title="Replay of each retained design",
        ylabel="Absolute target error (%)",
    )
    axes[1].legend(loc="best")
    figure.suptitle("Equal requested budgets, independently replayed optima", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def mo_figure(output: Path) -> None:
    data = rows(RESULTS / "mo" / "pareto.csv")
    cutoff = np.array([float(row["objective_cutoff_error"]) for row in data])
    ripple = np.array([float(row["objective_passband_ripple_db"]) for row in data])
    capacitance = np.array(
        [float(row["objective_total_capacitance_nf"]) for row in data]
    )
    selected = np.array([row["selected"] == "1" for row in data])
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    scatter = axes[0].scatter(
        cutoff,
        ripple,
        c=capacitance,
        cmap="viridis_r",
        s=42,
        edgecolor="#263238",
        linewidth=0.3,
        alpha=0.9,
    )
    axes[0].scatter(
        cutoff[selected],
        ripple[selected],
        marker="*",
        s=160,
        facecolor="#FFB300",
        edgecolor="#4E342E",
        linewidth=0.7,
        zorder=4,
    )
    colorbar = figure.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("Total capacitance (nF)")
    axes[0].set(
        title=f"Feasible nondominated set ({len(data)} points)",
        xlabel="Cutoff error (decades)",
        ylabel="Pass-band ripple (dB)",
    )

    scatter = axes[1].scatter(
        cutoff,
        capacitance,
        c=ripple,
        cmap="magma_r",
        s=42,
        edgecolor="#263238",
        linewidth=0.3,
        alpha=0.9,
    )
    axes[1].scatter(
        cutoff[selected],
        capacitance[selected],
        marker="*",
        s=160,
        facecolor="#00E5FF",
        edgecolor="#004D40",
        linewidth=0.7,
        zorder=4,
    )
    colorbar = figure.colorbar(scatter, ax=axes[1], pad=0.02)
    colorbar.set_label("Pass-band ripple (dB)")
    axes[1].set(
        title="A catalogue, not one weighted compromise",
        xlabel="Cutoff error (decades)",
        ylabel="Total capacitance (nF)",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markersize=11,
            markerfacecolor="#FFB300",
            markeredgecolor="#4E342E",
            label="Objective extremes / compromise",
        )
    ]
    axes[0].legend(handles=handles, loc="best")
    figure.suptitle("Constrained MODE: fourth-order low-pass trade-offs", fontsize=13)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    save(figure, output)


def qd_figure(output: Path) -> None:
    data = rows(RESULTS / "qd" / "archive.csv")
    progress = rows(RESULTS / "qd" / "convergence.csv")
    manifest = json.loads((RESULTS / "qd" / "run.json").read_text(encoding="utf-8"))
    side = int(manifest["qd"]["grid_shape"][0])
    heatmap = np.full((side, side), np.nan)
    for row in data:
        heatmap[int(row["grid_y"]), int(row["grid_x"])] = float(
            row["quality_robustness_db"]
        )
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    image = axes[0].imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=(
            manifest["qd"]["descriptor_lower"][0],
            manifest["qd"]["descriptor_upper"][0],
            manifest["qd"]["descriptor_lower"][1],
            manifest["qd"]["descriptor_upper"][1],
        ),
        cmap="viridis_r",
        interpolation="nearest",
    )
    colorbar = figure.colorbar(image, ax=axes[0], pad=0.02)
    colorbar.set_label("Tolerance sensitivity (dB, lower is better)")
    axes[0].set(
        title=f"E12 catalogue: {len(data)}/{side * side} niches",
        xlabel="log₁₀ centre frequency (Hz)",
        ylabel="Peak gain (dB)",
    )

    evaluations = [int(row["evaluations"]) for row in progress]
    coverage = [100.0 * float(row["coverage"]) for row in progress]
    invalid = [100.0 * float(row["invalid_fraction"]) for row in progress]
    axes[1].plot(evaluations, coverage, color="#009E73", linewidth=1.9)
    axes[1].set(
        title="MAP-Elites search progress",
        xlabel="Logical candidate evaluations",
        ylabel="Archive coverage (%)",
        ylim=(0.0, 105.0),
    )
    twin = axes[1].twinx()
    twin.plot(evaluations, invalid, color="#D55E00", linewidth=1.4)
    twin.set_ylabel("Invalid evaluations (%)", color="#A33C00")
    twin.set_ylim(0.0, max(10.0, max(invalid) * 1.25))
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#009E73", lw=1.9, label="Coverage"),
            Line2D([0], [0], color="#D55E00", lw=1.4, label="Invalid share"),
        ],
        loc="best",
    )
    figure.suptitle("Quality diversity over manufacturable E12 designs", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def bode_figure(output: Path) -> None:
    data = rows(RESULTS / "qd" / "elites.csv")
    niches = sorted({int(row["niche_id"]) for row in data})
    figure, axis = plt.subplots(figsize=(7.8, 4.2))
    palette = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(niches)))
    for color, niche in zip(palette, niches, strict=True):
        selected = [row for row in data if int(row["niche_id"]) == niche]
        frequency = [float(row["frequency_hz"]) for row in selected]
        gain = [float(row["gain_db"]) for row in selected]
        centre = 10 ** float(selected[0]["descriptor_log10_f0"])
        peak = float(selected[0]["descriptor_peak_gain_db"])
        axis.semilogx(
            frequency,
            gain,
            color=color,
            linewidth=1.7,
            label=f"{centre / 1e3:.1f} kHz, {peak:.1f} dB",
        )
    axis.set(
        title="Six replayed elites span the catalogue",
        xlabel="Frequency (Hz)",
        ylabel="Gain (dB)",
    )
    axis.legend(title="Centre, peak", loc="best", ncol=2)
    figure.tight_layout()
    save(figure, output)


def render(destination: Path) -> list[Path]:
    validate_artifacts()
    configure()
    generated = [
        destination / "feature-extraction.svg",
        destination / "so-convergence.svg",
        destination / "mo-pareto.svg",
        destination / "qd-archive.svg",
        destination / "bode-elites.svg",
    ]
    feature_figure(generated[0])
    so_figure(generated[1])
    mo_figure(generated[2])
    qd_figure(generated[3])
    bode_figure(generated[4])
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        for path in render(IMAGES):
            print(path)
        return 0
    stale: list[Path] = []
    with tempfile.TemporaryDirectory() as temporary:
        for generated in render(Path(temporary)):
            checked_in = IMAGES / generated.name
            if not checked_in.is_file() or not filecmp.cmp(
                generated, checked_in, shallow=False
            ):
                stale.append(checked_in)
    if stale:
        print("missing or stale sindr-circuit-design figures:")
        for path in stale:
            print(path)
        return 1
    if not (IMAGES / "architecture.svg").is_file():
        print(f"missing architecture diagram: {IMAGES / 'architecture.svg'}")
        return 1
    print("figures match checked-in evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
