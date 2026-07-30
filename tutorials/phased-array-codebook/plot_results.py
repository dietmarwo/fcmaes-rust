#!/usr/bin/env python3
"""Render deterministic phased-array figures from checked-in CSV evidence."""

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
from matplotlib import patches
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "publication"
IMAGES = ROOT / "images"
COLORS = {"cma": "#0072B2", "de": "#009E73", "bite": "#D55E00"}
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
            "svg.hashsalt": "fcmaes-phased-array-codebook-v1",
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
        if manifest["tutorial"] != "phased-array-codebook":
            raise ValueError(f"{path} names the wrong tutorial")
        for artifact in manifest["artifacts"].values():
            if not (path.parent / artifact).is_file():
                raise ValueError(f"{path} references missing {artifact}")

    qd = json.loads((RESULTS / "qd" / "run.json").read_text(encoding="utf-8"))
    archive = rows(RESULTS / "qd" / "qd_archive.csv")
    if len(archive) != qd["qd"]["occupied"]:
        raise ValueError("QD occupied count disagrees with qd_archive.csv")
    columns, row_count = qd["qd"]["grid_shape"]
    if columns * row_count != qd["qd"]["capacity"]:
        raise ValueError("QD grid shape does not match archive capacity")
    if any(
        (int(row["grid_x"]), int(row["grid_y"]))
        != (int(row["niche_id"]) % columns, int(row["niche_id"]) // columns)
        for row in archive
    ):
        raise ValueError("QD coordinates are not the inverse archive mapping")
    if any(float(row["constraint_robust_psll"]) > 1e-12 for row in archive):
        raise ValueError("QD archive contains an infeasible robust-PSLL row")

    mo = rows(RESULTS / "mo" / "pareto.csv")
    for row in mo:
        if any(
            float(row[column]) > 1e-12
            for column in ("constraint_null_db", "constraint_kernel")
        ):
            raise ValueError("MODE artifact contains an infeasible point")

    pilot = json.loads((RESULTS / "pilot" / "run.json").read_text(encoding="utf-8"))
    if pilot["qd"]["decision"] not in {"accepted", "primary-secondary"}:
        raise ValueError("checked-in QD evidence must clear D1 or the symmetric D2 fallback")


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "phased-array-codebook/plot_results.py"},
    )
    plt.close(figure)
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n",
        encoding="utf-8",
    )


def architecture_figure(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.0, 3.5))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3.5)
    axis.axis("off")
    boxes = [
        (0.2, 1.25, 1.6, 1.0, "normalized\ncandidate", "#E3F2FD"),
        (2.1, 1.25, 1.7, 1.0, "6-bit phase +\n5-bit attenuation", "#FFF3E0"),
        (4.1, 1.25, 1.5, 1.0, "direct/FFT\narray factor", "#E8F5E9"),
        (5.9, 1.25, 1.7, 1.0, "49 named\ntraining patterns", "#F3E5F5"),
        (7.9, 1.25, 1.8, 1.0, "retry · MODE ·\nMAP-Elites", "#FFEBEE"),
    ]
    for x, y, width, height, text, color in boxes:
        axis.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="#455A64",
                linewidth=1.2,
            )
        )
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center")
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        axis.annotate(
            "",
            xy=(right[0], 1.75),
            xytext=(left[0] + left[2], 1.75),
            arrowprops={"arrowstyle": "->", "color": "#455A64", "lw": 1.4},
        )
    axis.text(
        5.0,
        3.12,
        "One serial, auditable RF evaluation; fcmaes owns candidate parallelism",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        4.95,
        0.48,
        "checked-in perturbations → measured peak/HPBW/PSLL → register-code artifacts",
        ha="center",
        color="#37474F",
    )
    save(figure, output)


def staircase_figure(output: Path) -> None:
    staircase = rows(RESULTS / "validation" / "staircase.csv")
    x = np.array([float(row["coordinate"]) for row in staircase])
    code = np.array([int(row["phase_code"]) for row in staircase])
    objective = np.array([float(row["objective"]) for row in staircase])
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.65))
    element_x = np.arange(16) - 7.5
    axes[0].scatter(element_x, np.zeros(16), s=72, color="#0072B2", zorder=3)
    for index, position in enumerate(element_x):
        axes[0].text(position, 0.13, str(index), ha="center", fontsize=7)
    axes[0].annotate(
        "d = λ/2",
        xy=(-6.5, 0),
        xytext=(-7.0, -0.45),
        arrowprops={"arrowstyle": "<->", "color": "#455A64"},
        ha="center",
    )
    axes[0].set(
        title="Stage A: centered 16-element ULA",
        xlabel="Element position (λ/2 units)",
        xlim=(-8.2, 8.2),
        ylim=(-0.65, 0.55),
    )
    axes[0].set_yticks([])
    axes[1].step(x, code, where="post", color="#D55E00", lw=1.7, label="6-bit code")
    twin = axes[1].twinx()
    twin.step(
        x,
        objective,
        where="post",
        color="#0072B2",
        alpha=0.75,
        lw=1.2,
        label="objective",
    )
    axes[1].set(
        title="The objective inherits the hardware staircase",
        xlabel="One normalized phase coordinate",
        ylabel="Decoded phase code",
    )
    twin.set_ylabel("Robust objective (dB)")
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#D55E00", lw=1.7, label="phase register"),
            Line2D([0], [0], color="#0072B2", lw=1.2, label="objective"),
        ],
        loc="best",
    )
    figure.tight_layout()
    save(figure, output)


def pattern_figure(output: Path) -> None:
    data = rows(RESULTS / "so" / "pattern.csv")
    angle = np.array([float(row["angle_deg"]) for row in data])
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    for column, label, color in [
        ("uniform_db", "Uniform quantized", "#7F8C8D"),
        ("optimized_db", "BiteOpt robust design", "#D55E00"),
        ("chebyshev_reference_db", "Continuous Chebyshev reference", "#0072B2"),
    ]:
        axis.plot(
            angle,
            [float(row[column]) for row in data],
            label=label,
            color=color,
            lw=1.5,
        )
    axis.axvline(20.0, color="#37474F", ls=":", lw=1.0)
    axis.set(
        title="Quantized robust beam synthesis at 20°",
        xlabel="Signed cut angle (deg)",
        ylabel="Relative level (dB)",
        xlim=(-70, 70),
        ylim=(-60, 2),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    save(figure, output)


def so_figure(output: Path) -> None:
    data = rows(RESULTS / "so" / "convergence.csv")
    figure, axis = plt.subplots(figsize=(7.8, 4.0))
    for optimizer in ("cma", "de", "bite"):
        selected = [row for row in data if row["optimizer"] == optimizer]
        axis.step(
            [float(row["evaluations"]) for row in selected],
            [float(row["best_objective"]) for row in selected],
            where="post",
            color=COLORS[optimizer],
            lw=1.8,
            marker="o",
            ms=3,
            label={"cma": "active CMA-ES", "de": "DE", "bite": "BiteOpt"}[optimizer],
        )
    axis.set(
        title="Equal-budget robust scalar synthesis",
        xlabel="Completed objective evaluations",
        ylabel="Best objective (dB, lower is better)",
    )
    axis.legend(loc="best")
    figure.tight_layout()
    save(figure, output)


def mo_figure(output: Path) -> None:
    data = rows(RESULTS / "mo" / "pareto.csv")
    active = np.array([float(row["objective_active_count"]) for row in data])
    psll = np.array([float(row["objective_psll_db"]) for row in data])
    robust = np.array([float(row["objective_robustness_margin_db"]) for row in data])
    selected = np.array([row["selected"] == "1" for row in data])
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    scatter = axes[0].scatter(
        active,
        psll,
        c=robust,
        cmap="viridis",
        s=34,
        edgecolor="#263238",
        linewidth=0.25,
    )
    axes[0].scatter(
        active[selected],
        psll[selected],
        marker="*",
        s=150,
        facecolor="#FDD835",
        edgecolor="#5D4037",
    )
    axes[0].set(
        title="Hardware count versus nominal sidelobes",
        xlabel="Active elements",
        ylabel="PSLL (dB)",
    )
    figure.colorbar(scatter, ax=axes[0], label="Failure degradation (dB)")
    gain = -np.array([float(row["objective_negative_peak_gain_db"]) for row in data])
    axes[1].scatter(
        gain,
        robust,
        c=active,
        cmap="plasma",
        s=34,
        edgecolor="#263238",
        linewidth=0.25,
    )
    axes[1].set(
        title="Gain and robustness remain competing goals",
        xlabel="Peak gain (dB)",
        ylabel="Failure degradation (dB)",
    )
    figure.tight_layout()
    save(figure, output)


def qd_figure(output: Path) -> None:
    manifest = json.loads((RESULTS / "qd" / "run.json").read_text(encoding="utf-8"))
    pilot = json.loads((RESULTS / "pilot" / "run.json").read_text(encoding="utf-8"))
    columns, row_count = manifest["qd"]["grid_shape"]
    archive = rows(RESULTS / "qd" / "codebook.csv")
    heat = np.full((row_count, columns), np.nan)
    # Grid coordinates are in qd_archive, keyed by niche.
    coordinates = {
        row["niche_id"]: (int(row["grid_x"]), int(row["grid_y"]))
        for row in rows(RESULTS / "qd" / "qd_archive.csv")
    }
    for row in archive:
        x, y = coordinates[row["niche_id"]]
        heat[y, x] = float(row["worst_psll_db"])
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    image = axes[0].imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=(-52, 52, 6, 14),
        cmap="viridis_r",
        vmin=-13,
        vmax=-10,
    )
    axes[0].set(
        title=(
            f"{'Primary' if pilot['qd']['decision'] == 'accepted' else 'Secondary'} "
            f"D1 codebook: {len(archive)}/{columns * row_count} niches"
        ),
        xlabel="Measured peak direction (deg)",
        ylabel="Measured HPBW (deg)",
    )
    figure.colorbar(image, ax=axes[0], label="Worst training PSLL (dB)")
    progress = rows(RESULTS / "qd" / "qd_convergence.csv")
    evaluations = [float(row["evaluations"]) for row in progress]
    coverage = [100 * float(row["coverage"]) for row in progress]
    invalid = [100 * float(row["invalid_fraction"]) for row in progress]
    infeasible = [100 * float(row["infeasible_fraction"]) for row in progress]
    axes[1].plot(evaluations, coverage, color="#009E73", lw=1.8, label="Coverage")
    twin = axes[1].twinx()
    twin.plot(evaluations, invalid, color="#D55E00", lw=1.4, label="Invalid")
    twin.plot(
        evaluations,
        infeasible,
        color="#CC79A7",
        lw=1.2,
        ls="--",
        label="Infeasible",
    )
    axes[1].set(
        title="Raw-code emitters mostly miss valid, robust beams",
        xlabel="Candidate evaluations",
        ylabel="Coverage (%)",
    )
    twin.set_ylabel("Invalid candidates (%)", color="#A33C00")
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#009E73", lw=1.8, label="coverage"),
            Line2D([0], [0], color="#D55E00", lw=1.4, label="invalid"),
            Line2D(
                [0],
                [0],
                color="#CC79A7",
                lw=1.2,
                ls="--",
                label="infeasible",
            ),
        ],
        loc="best",
    )
    figure.tight_layout()
    save(figure, output)


def migration_figure(output: Path) -> None:
    data = rows(RESULTS / "qd" / "holdout_migration.csv")
    train_x = np.array([float(row["train_peak_deg"]) for row in data])
    train_y = np.array([float(row["train_hpbw_deg"]) for row in data])
    hold_x = np.array([float(row["holdout_peak_deg"]) for row in data])
    hold_y = np.array([float(row["holdout_hpbw_deg"]) for row in data])
    moved = np.array([row["moved"] == "1" for row in data])
    retained = int((~moved).sum())
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 4.0),
        gridspec_kw={"width_ratios": [2.3, 1.0]},
    )
    axes[0].scatter(train_x, train_y, color="#263238", s=17, label="same niche")
    if moved.any():
        axes[0].scatter(
            train_x[moved],
            train_y[moved],
            facecolors="none",
            edgecolors="#D55E00",
            linewidths=1.8,
            s=90,
            label="crossed a niche edge",
        )
        for x0, y0, x1, y1 in zip(
            train_x[moved],
            train_y[moved],
            hold_x[moved],
            hold_y[moved],
            strict=True,
        ):
            axes[0].annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "color": "#D55E00", "lw": 1.5},
            )
    axes[0].set(
        title="Training codebook and holdout niche movement",
        xlabel="Peak direction (deg)",
        ylabel="HPBW (deg)",
    )
    axes[0].legend(loc="best")
    axes[1].bar(
        ["same niche", "moved"],
        [retained, int(moved.sum())],
        color=["#009E73", "#D55E00"],
        width=0.65,
    )
    axes[1].set(
        title=f"Holdout retention\n{retained}/{len(data)} = {100 * retained / len(data):.1f}%",
        ylabel="Codebook entries",
        ylim=(0, len(data) * 1.12),
    )
    for index, value in enumerate([retained, int(moved.sum())]):
        axes[1].text(index, value + 0.7, str(value), ha="center")
    figure.tight_layout()
    save(figure, output)


def failure_figure(output: Path) -> None:
    data = rows(RESULTS / "so" / "failure_envelope.csv")
    angle = np.array([float(row["angle_deg"]) for row in data])
    nominal = np.array([float(row["nominal_db"]) for row in data])
    lower = np.array([float(row["envelope_min_db"]) for row in data])
    upper = np.array([float(row["envelope_max_db"]) for row in data])
    figure, axis = plt.subplots(figsize=(8.2, 4.1))
    axis.fill_between(angle, lower, upper, color="#90CAF9", alpha=0.55, label="16 single failures")
    axis.plot(angle, nominal, color="#D55E00", lw=1.5, label="nominal")
    axis.set(
        title="Single-element failures are optimized explicitly",
        xlabel="Signed cut angle (deg)",
        ylabel="Level relative to nominal peak (dB)",
        xlim=(-70, 70),
        ylim=(-60, 2),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    save(figure, output)


def validation_figure(output: Path) -> None:
    data = {row["metric"]: row for row in rows(RESULTS / "validation" / "validation.csv")}
    labels = ["direct ULA", "FFT ULA", "direct URA", "FFT URA"]
    seconds = [
        float(data["direct_linear"]["value"]) * 1e-6,
        float(data["fft_linear"]["value"]) * 1e-6,
        float(data["direct_planar"]["value"]) * 1e-3,
        float(data["fft_planar"]["value"]) * 1e-6,
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    axes[0].bar(labels, seconds, color=["#0072B2", "#56B4E9", "#009E73", "#90D4B8"])
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].set(
        title="Independent FFT-node kernel is faster",
        ylabel="Warm evaluation time (s, log scale)",
    )
    coarse = float(data["coarse_planar_directivity"]["value"])
    fine = float(data["fine_planar_directivity"]["value"])
    axes[1].bar(["90×180", "180×360"], [coarse, fine], color=["#9575CD", "#5E35B1"])
    axes[1].set(
        title="Planar directivity converges on the polar grid",
        ylabel="One-sided directivity (dBi)",
        ylim=(min(coarse, fine) - 0.08, max(coarse, fine) + 0.08),
    )
    for index, value in enumerate([coarse, fine]):
        axes[1].text(index, value + 0.008, f"{value:.3f}", ha="center")
    figure.tight_layout()
    save(figure, output)


def render(destination: Path) -> list[Path]:
    validate_artifacts()
    configure()
    generated = [
        destination / "architecture.svg",
        destination / "quantization-staircase.svg",
        destination / "pattern-cuts.svg",
        destination / "so-convergence.svg",
        destination / "mo-pareto.svg",
        destination / "qd-codebook.svg",
        destination / "holdout-migration.svg",
        destination / "failure-envelope.svg",
        destination / "kernel-validation.svg",
    ]
    architecture_figure(generated[0])
    staircase_figure(generated[1])
    pattern_figure(generated[2])
    so_figure(generated[3])
    mo_figure(generated[4])
    qd_figure(generated[5])
    migration_figure(generated[6])
    failure_figure(generated[7])
    validation_figure(generated[8])
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
        print("missing or stale phased-array-codebook figures:")
        for path in stale:
            print(path)
        return 1
    print("phased-array figures match checked-in evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
