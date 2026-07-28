#!/usr/bin/env python3
"""Render deterministic optical-design figures from native Rust artifacts."""

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
COLORS = {"cma": "#0072B2", "de": "#009E73", "bite": "#D55E00"}


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
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
            "svg.hashsalt": "fcmaes-optical-lens-design-v1",
        }
    )


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "optical-lens-design/plot_results.py"},
    )
    plt.close(figure)
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n",
        encoding="utf-8",
    )


def validate_artifacts() -> None:
    validation = json.loads(
        (RESULTS / "validation" / "summary.json").read_text(encoding="utf-8")
    )
    if not validation["passed"]:
        raise ValueError("checked-in optical reference gate does not pass")
    for formulation in ("so", "mo"):
        manifest = json.loads(
            (RESULTS / formulation / "run.json").read_text(encoding="utf-8")
        )
        if manifest["schema_version"] != 1:
            raise ValueError("unexpected result schema")
        if manifest["tutorial"] != "optical-lens-design":
            raise ValueError("manifest names the wrong tutorial")
        for artifact in manifest["artifacts"].values():
            if not (RESULTS / formulation / artifact).is_file():
                raise ValueError(f"missing artifact: {artifact}")
    pareto = rows(RESULTS / "mo" / "pareto.csv")
    if not pareto:
        raise ValueError("empty optical Pareto front")
    constraint_columns = [
        "constraint_edge_thickness_mm",
        "constraint_efl_mm",
        "constraint_lost_rays",
    ]
    if any(
        float(row[column]) > 0.0 for row in pareto for column in constraint_columns
    ):
        raise ValueError("optical Pareto artifact contains an infeasible point")


def architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 3.0))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = [
        (0.2, 1.05, 1.7, 0.9, "11 controls\ncurvature + spacing", "#E3F2FD"),
        (2.35, 1.05, 1.7, 0.9, "sequential tracer\nsphere + Snell", "#E8F5E9"),
        (4.5, 1.05, 1.7, 0.9, "metrics\nspot · EFL · shape", "#FFF3E0"),
        (6.65, 1.55, 2.8, 0.72, "CMA / DE / BiteOpt retry\none sharp design", "#F3E5F5"),
        (6.65, 0.55, 2.8, 0.72, "constrained MODE\nspot · length · glass", "#FCE4EC"),
    ]
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#37474F",
            linewidth=1.1,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    for start, end in [
        ((1.9, 1.5), (2.35, 1.5)),
        ((4.05, 1.5), (4.5, 1.5)),
        ((6.2, 1.5), (6.65, 1.91)),
        ((6.2, 1.5), (6.65, 0.91)),
    ]:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#455A64", "lw": 1.4},
        )
    axis.text(
        5,
        2.72,
        "One serial, auditable ray trace per candidate; fcmaes owns parallelism",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    save(figure, output)


def validation(output: Path) -> None:
    data = json.loads(
        (RESULTS / "validation" / "summary.json").read_text(encoding="utf-8")
    )
    published = np.array(data["reference"]["published_on_axis_rms_mm"]) * 1000
    rust = np.array(data["rust"]["on_axis_rms_mm"]) * 1000
    labels = ["F 486.1 nm", "d 587.6 nm", "C 656.3 nm"]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    locations = np.arange(3)
    axes[0].bar(
        locations - 0.18, published, 0.36, color="#90CAF9", label="Optiland"
    )
    axes[0].bar(locations + 0.18, rust, 0.36, color="#0072B2", label="Rust")
    axes[0].set_xticks(locations, labels, rotation=12)
    axes[0].set(
        title="Published prescription: on-axis bundle",
        ylabel="RMS spot radius (µm)",
    )
    axes[0].legend()
    axes[1].bar(
        ["EFL error", "largest spot error"],
        [
            100 * data["measured"]["efl_relative_error"],
            100 * data["measured"]["maximum_on_axis_spot_relative_error"],
        ],
        color=["#009E73", "#D55E00"],
    )
    axes[1].axhline(20, color="#455A64", linestyle="--", label="spot limit")
    axes[1].set(
        title="Independent gate margins",
        ylabel="Relative error (%)",
    )
    axes[1].legend()
    figure.suptitle("Reference reproduction precedes optimization", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def convergence(output: Path) -> None:
    data = rows(RESULTS / "validation" / "ray_convergence.csv")
    rays = np.array([int(row["pupil_rays"]) for row in data])
    spots = np.array([float(row["weighted_rms_spot_um"]) for row in data])
    figure, axis = plt.subplots(figsize=(6.4, 3.7))
    axis.plot(rays, spots, marker="o", color="#0072B2", linewidth=1.8)
    axis.scatter([rays[-2], rays[-1]], [spots[-2], spots[-1]], color="#D55E00")
    axis.set(
        title="The optimized metric is checked for pupil-grid convergence",
        xlabel="Pupil rays per field and wavelength",
        ylabel="Weighted polychromatic RMS spot (µm)",
    )
    save(figure, output)


def so_convergence(output: Path) -> None:
    data = rows(RESULTS / "so" / "convergence.csv")
    best = {row["optimizer"]: row for row in rows(RESULTS / "so" / "best.csv")}
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for optimizer in ("cma", "de", "bite"):
        selected = [row for row in data if row["optimizer"] == optimizer]
        axes[0].step(
            [int(row["evaluations"]) for row in selected],
            [float(row["best_objective"]) for row in selected],
            where="post",
            linewidth=1.7,
            color=COLORS[optimizer],
            label=optimizer.upper(),
        )
    axes[0].set_yscale("log")
    axes[0].set(
        title="Equal requested budgets",
        xlabel="Objective evaluations",
        ylabel="Penalized objective (lower is better)",
    )
    axes[0].legend()
    names = ["cma", "de", "bite"]
    axes[1].bar(
        [name.upper() for name in names],
        [float(best[name]["rms_spot_um"]) for name in names],
        color=[COLORS[name] for name in names],
    )
    axes[1].set(
        title="Independent full-resolution replay",
        ylabel="Weighted RMS spot (µm)",
    )
    figure.suptitle("Multimodal scalar lens search", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def spot_diagrams(output: Path) -> None:
    data = rows(RESULTS / "so" / "spot_diagrams.csv")
    figure, axes = plt.subplots(2, 3, figsize=(9.4, 5.6), sharex=False, sharey=False)
    wavelength_colors = {0.4861: "#0072B2", 0.5876: "#009E73", 0.6563: "#D55E00"}
    for row_index, design in enumerate(("reference", "optimized")):
        for column, field in enumerate((0.0, 14.0, 20.0)):
            axis = axes[row_index, column]
            subset = [
                row
                for row in data
                if row["design"] == design
                and abs(float(row["field_deg"]) - field) < 1e-9
            ]
            for wavelength, color in wavelength_colors.items():
                points = [
                    row
                    for row in subset
                    if abs(float(row["wavelength_um"]) - wavelength) < 1e-6
                ]
                x = np.array([float(row["x_mm"]) for row in points])
                y = np.array([float(row["y_mm"]) for row in points])
                axis.scatter(
                    1000 * (x - x.mean()),
                    1000 * (y - y.mean()),
                    s=6,
                    alpha=0.58,
                    color=color,
                    label=f"{wavelength * 1000:.1f} nm",
                )
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_title(f"{design.title()}, field {field:g}°")
            axis.set_xlabel("Centred x (µm)")
            if column == 0:
                axis.set_ylabel("Centred y (µm)")
    axes[0, 2].legend(loc="best")
    figure.suptitle("Polychromatic geometric spot diagrams", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def mo_pareto(output: Path) -> None:
    data = rows(RESULTS / "mo" / "pareto.csv")
    spot = np.array([float(row["objective_rms_spot_um"]) for row in data])
    length = np.array([float(row["objective_track_length_mm"]) for row in data])
    volume = np.array([float(row["objective_glass_volume_mm3"]) for row in data])
    selected = np.array([row["selected"] == "1" for row in data])
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    scatter = axes[0].scatter(
        length, spot, c=volume, cmap="viridis_r", s=35, edgecolor="#263238", lw=0.3
    )
    axes[0].scatter(
        length[selected],
        spot[selected],
        marker="*",
        s=150,
        color="#FFB300",
        edgecolor="#4E342E",
    )
    figure.colorbar(scatter, ax=axes[0]).set_label("Glass volume (mm³)")
    axes[0].set(
        title=f"Feasible nondominated set ({len(data)} designs)",
        xlabel="Track length (mm)",
        ylabel="RMS spot (µm)",
    )
    scatter = axes[1].scatter(
        volume, spot, c=length, cmap="magma_r", s=35, edgecolor="#263238", lw=0.3
    )
    axes[1].scatter(
        volume[selected],
        spot[selected],
        marker="*",
        s=150,
        color="#00E5FF",
        edgecolor="#004D40",
    )
    figure.colorbar(scatter, ax=axes[1]).set_label("Track length (mm)")
    axes[1].set(
        title="Optical quality versus material",
        xlabel="Glass volume (mm³)",
        ylabel="RMS spot (µm)",
    )
    figure.suptitle("Constrained Cooke-triplet trade-offs", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def render(directory: Path) -> None:
    configure()
    validate_artifacts()
    architecture(directory / "architecture.svg")
    validation(directory / "validation.svg")
    convergence(directory / "ray-convergence.svg")
    so_convergence(directory / "so-convergence.svg")
    spot_diagrams(directory / "spot-diagrams.svg")
    mo_pareto(directory / "mo-pareto.svg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        render(IMAGES)
        print("optical-lens-design figures are current")
        return 0
    with tempfile.TemporaryDirectory() as temporary:
        generated = Path(temporary)
        render(generated)
        expected = sorted(path.name for path in generated.glob("*.svg"))
        stale = [
            IMAGES / name
            for name in expected
            if not (IMAGES / name).is_file()
            or not filecmp.cmp(generated / name, IMAGES / name, shallow=False)
        ]
    if stale:
        print("missing or stale optical-lens-design figures:")
        for path in stale:
            print(path)
        print(
            f"renderer uses matplotlib {matplotlib.__version__}; "
            "checked-in figures use tutorials/python/requirements-lock.txt "
            "(matplotlib 3.11.1)"
        )
        return 1
    print("optical-lens-design figures are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
