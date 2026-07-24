#!/usr/bin/env python3
"""Render deterministic MODE and MAP-Elites SVG summaries from result CSVs."""

from __future__ import annotations

import argparse
import csv
import filecmp
import math
import statistics
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def values(data: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in data]


def configure() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "svg.hashsalt": "fcmaes-cfd-room-ventilation",
        }
    )


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "cfd-room-ventilation/plot_results.py"},
    )
    plt.close(figure)
    print(path)


def plot_mode(directory: Path, output: Path) -> None:
    pareto = rows(directory / "pareto.csv")
    convergence = rows(directory / "convergence.csv")
    if not pareto:
        raise ValueError(f"empty Pareto data: {directory / 'pareto.csv'}")

    exposure = values(pareto, "exposure")
    fan = values(pareto, "fan_power")
    final_mass = values(pareto, "final_mass_fraction")
    selected = [row["selected"] == "1" for row in pareto]

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    scatter = axes[0].scatter(
        exposure,
        fan,
        c=final_mass,
        cmap="viridis_r",
        s=38,
        linewidths=0.35,
        edgecolors="#263238",
        alpha=0.90,
    )
    for x, y, is_selected in zip(exposure, fan, selected, strict=True):
        if is_selected:
            axes[0].scatter(
                [x],
                [y],
                marker="*",
                s=180,
                facecolor="#ffb300",
                edgecolor="#4e342e",
                linewidth=0.8,
                zorder=5,
            )
    colorbar = figure.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("Final pollutant mass fraction")
    axes[0].set(
        title=f"MODE feasible Pareto set ({len(pareto)} points)",
        xlabel="Occupied-zone exposure",
        ylabel="Fan-power proxy",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="",
                markersize=11,
                markerfacecolor="#ffb300",
                markeredgecolor="#4e342e",
                label="Reporting representative",
            )
        ],
        loc="best",
    )

    samples = [
        row
        for row in convergence
        if math.isfinite(float(row["best_quality"]))
    ]
    evaluations = values(samples, "evaluations")
    best = values(samples, "best_quality")
    feasible = [100.0 * value for value in values(samples, "feasible_fraction")]
    axes[1].plot(evaluations, best, color="#1565c0", linewidth=1.8)
    axes[1].set(
        title="MODE search progress",
        xlabel="Objective evaluations",
        ylabel="Best reporting quality (lower is better)",
    )
    axes[1].grid(alpha=0.22)
    twin = axes[1].twinx()
    twin.plot(evaluations, feasible, color="#ef6c00", linewidth=1.2, alpha=0.72)
    twin.set_ylabel("Feasible candidates per batch (%)", color="#bf360c")
    twin.set_ylim(0.0, 105.0)
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#1565c0", lw=1.8, label="Best quality"),
            Line2D([0], [0], color="#ef6c00", lw=1.2, label="Feasible batch share"),
        ],
        loc="best",
    )
    figure.suptitle("Room-ventilation multi-objective optimization", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def plot_qd(directory: Path, output: Path) -> None:
    archive = rows(directory / "archive.csv")
    convergence = rows(directory / "convergence.csv")
    if not archive:
        raise ValueError(f"empty archive data: {directory / 'archive.csv'}")

    flow = values(archive, "descriptor_flow_rate_m2_s")
    low_velocity = values(archive, "descriptor_low_velocity_fraction")
    quality = values(archive, "quality")
    selected = [row["selected"] == "1" for row in archive]

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    scatter = axes[0].scatter(
        flow,
        low_velocity,
        c=quality,
        cmap="plasma_r",
        s=48,
        marker="s",
        linewidths=0.25,
        edgecolors="#263238",
    )
    for x, y, is_selected in zip(flow, low_velocity, selected, strict=True):
        if is_selected:
            axes[0].scatter(
                [x],
                [y],
                marker="*",
                s=190,
                facecolor="#00e5ff",
                edgecolor="#004d40",
                linewidth=0.9,
                zorder=5,
            )
    colorbar = figure.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("Scalar quality (lower is better)")
    axes[0].set(
        title=f"MAP-Elites archive ({len(archive)} occupied niches)",
        xlabel="Fresh-air flow (m²/s)",
        ylabel="Occupied-zone low-velocity fraction",
        xlim=(0.09, 2.025),
        ylim=(0.0, 1.0),
    )
    axes[0].grid(alpha=0.18)
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="",
                markersize=11,
                markerfacecolor="#00e5ff",
                markeredgecolor="#004d40",
                label="Best-quality elite",
            )
        ],
        loc="upper right",
    )

    evaluations = values(convergence, "evaluations")
    coverage = [100.0 * value for value in values(convergence, "coverage")]
    best = values(convergence, "best_quality")
    axes[1].plot(evaluations, coverage, color="#2e7d32", linewidth=1.8)
    axes[1].set(
        title="MAP-Elites search progress",
        xlabel="Objective evaluations",
        ylabel="Archive coverage (%)",
        ylim=(0.0, 100.0),
    )
    axes[1].grid(alpha=0.22)
    twin = axes[1].twinx()
    twin.plot(evaluations, best, color="#6a1b9a", linewidth=1.4)
    twin.set_ylabel("Best scalar quality (lower is better)", color="#6a1b9a")
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#2e7d32", lw=1.8, label="Coverage"),
            Line2D([0], [0], color="#6a1b9a", lw=1.4, label="Best quality"),
        ],
        loc="center right",
    )
    figure.suptitle("Room-ventilation quality-diversity optimization", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def field_arrays(path: Path) -> dict[str, np.ndarray]:
    data = rows(path)
    nx = max(int(row["i"]) for row in data) + 1
    ny = max(int(row["j"]) for row in data) + 1

    def array(column: str, conversion=float) -> np.ndarray:
        result = np.zeros((ny, nx), dtype=float)
        for row in data:
            result[int(row["j"]), int(row["i"])] = conversion(row[column])
        return result

    return {
        "u": array("u_lattice") / 0.05,
        "v": array("v_lattice") / 0.05,
        "concentration": array("concentration"),
        "solid": array("solid", int).astype(bool),
        "source": np.array(
            [
                float(data[0]["source_x_fraction"]) * 5.0,
                float(data[0]["source_y_fraction"]) * 3.0,
            ]
        ),
    }


def plot_fields(directory: Path, output: Path) -> None:
    definitions = [
        ("Baseline", "baseline-field.csv"),
        ("MODE representative", "mode-field.csv"),
        ("MAP-Elites best elite", "map_elites-field.csv"),
    ]
    fields = [(name, field_arrays(directory / filename)) for name, filename in definitions]
    speed_max = max(
        np.hypot(field["u"], field["v"])[~field["solid"]].max()
        for _, field in fields
    )
    concentration_max = max(
        field["concentration"][~field["solid"]].max() for _, field in fields
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 5.8), sharex=True, sharey=True)
    speed_image = None
    concentration_image = None
    for column, (name, field) in enumerate(fields):
        speed = np.hypot(field["u"], field["v"])
        speed_image = axes[0, column].imshow(
            np.ma.masked_where(field["solid"], speed),
            origin="lower",
            extent=(0.0, 5.0, 0.0, 3.0),
            cmap="cividis",
            vmin=0.0,
            vmax=speed_max,
            interpolation="nearest",
        )
        y, x = np.mgrid[
            0.5 * 3.0 / speed.shape[0] : 3.0 : 3.0 / speed.shape[0],
            0.5 * 5.0 / speed.shape[1] : 5.0 : 5.0 / speed.shape[1],
        ]
        mask = field["solid"]
        u = np.where(mask, np.nan, field["u"])
        v = np.where(mask, np.nan, field["v"])
        axes[0, column].quiver(
            x[::3, ::3],
            y[::3, ::3],
            u[::3, ::3],
            v[::3, ::3],
            color="white",
            alpha=0.75,
            scale=12.0,
            width=0.003,
        )
        axes[0, column].contour(
            mask.astype(float),
            levels=[0.5],
            origin="lower",
            extent=(0.0, 5.0, 0.0, 3.0),
            colors="black",
            linewidths=1.0,
        )
        axes[0, column].set_title(name)

        concentration_image = axes[1, column].imshow(
            np.ma.masked_where(mask, field["concentration"]),
            origin="lower",
            extent=(0.0, 5.0, 0.0, 3.0),
            cmap="magma",
            vmin=0.0,
            vmax=concentration_max,
            interpolation="nearest",
        )
        axes[1, column].contour(
            mask.astype(float),
            levels=[0.5],
            origin="lower",
            extent=(0.0, 5.0, 0.0, 3.0),
            colors="white",
            linewidths=0.9,
        )
        axes[1, column].scatter(
            [field["source"][0]],
            [field["source"][1]],
            marker="x",
            color="#00e5ff",
            linewidth=1.6,
            s=42,
            label="Worst training release",
        )
        axes[1, column].set_xlabel("Room x (m)")
        for row in range(2):
            axes[row, column].set_xlim(0.0, 5.0)
            axes[row, column].set_ylim(0.0, 3.0)
            axes[row, column].set_aspect("equal")
    axes[0, 0].set_ylabel("Velocity field\nRoom y (m)")
    axes[1, 0].set_ylabel("Final pollutant field\nRoom y (m)")
    axes[1, 2].legend(loc="upper right")
    assert speed_image is not None and concentration_image is not None
    speed_colorbar = figure.add_axes((0.91, 0.56, 0.014, 0.30))
    concentration_colorbar = figure.add_axes((0.91, 0.13, 0.014, 0.30))
    figure.colorbar(
        speed_image,
        cax=speed_colorbar,
        label="Speed (m/s, lattice conversion)",
    )
    figure.colorbar(
        concentration_image,
        cax=concentration_colorbar,
        label="Normalized pollutant concentration",
    )
    figure.suptitle("Baseline and optimized room-flow fields", fontsize=13)
    figure.subplots_adjust(
        left=0.06, right=0.88, bottom=0.08, top=0.90, wspace=0.10, hspace=0.16
    )
    save(figure, output)


def selected_quality(directory: Path, filename: str) -> float:
    data = rows(directory / filename)
    return float(next(row["quality"] for row in data if row["selected"] == "1"))


def replication_data(seed_root: Path) -> list[dict[str, float | int | str]]:
    data: list[dict[str, float | int | str]] = []
    for method, prefix, result_file in [
        ("MODE", "mode-seed-", "pareto.csv"),
        ("MAP-Elites", "qd-seed-", "archive.csv"),
    ]:
        for directory in sorted(seed_root.glob(f"{prefix}*")):
            seed = int(directory.name.removeprefix(prefix))
            result = rows(directory / result_file)
            validation = rows(directory / "validation.csv")[0]
            convergence = rows(directory / "convergence.csv")
            data.append(
                {
                    "method": method,
                    "seed": seed,
                    "training_quality": selected_quality(directory, result_file),
                    "validation_quality": float(validation["quality"]),
                    "elapsed_seconds": float(convergence[-1]["elapsed_seconds"]),
                    "result_size": len(result),
                    "coverage": len(result) / 400.0 if method == "MAP-Elites" else math.nan,
                    "qd_score": (
                        float(convergence[-1]["qd_score"])
                        if method == "MAP-Elites"
                        else math.nan
                    ),
                }
            )
    return data


def write_replication_summary(data: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    print(path)


def plot_verification(
    verification_directory: Path,
    seed_root: Path,
    output: Path,
    summary: Path,
) -> None:
    resolution = rows(verification_directory / "resolution-study.csv")
    replications = replication_data(seed_root)
    if len(replications) < 6:
        raise ValueError("expected three MODE and three MAP-Elites seed runs")
    write_replication_summary(replications, summary)

    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    styles = {
        "baseline": ("Baseline", "#616161", "o"),
        "mode": ("MODE", "#1565c0", "s"),
        "map_elites": ("MAP-Elites", "#8e24aa", "^"),
    }
    for design, (label, color, marker) in styles.items():
        for source_set, linestyle, alpha in [
            ("training", "--", 0.65),
            ("held_out", "-", 1.0),
        ]:
            selected = [
                row
                for row in resolution
                if row["design"] == design and row["source_set"] == source_set
            ]
            selected.sort(key=lambda row: int(row["nx"]))
            x = [int(row["nx"]) * int(row["ny"]) for row in selected]
            y = values(selected, "quality")
            axes[0].plot(
                x,
                y,
                color=color,
                marker=marker,
                linestyle=linestyle,
                alpha=alpha,
                label=f"{label}, {source_set.replace('_', ' ')}",
            )
            for xi, yi, row in zip(x, y, selected, strict=True):
                if row["feasible"] == "0":
                    axes[0].scatter(
                        [xi],
                        [yi],
                        marker="x",
                        color="#d32f2f",
                        s=70,
                        linewidth=1.5,
                        zorder=5,
                    )
    axes[0].set(
        title="Three-grid resolution sensitivity",
        xlabel="Grid cells",
        ylabel="Reporting scalar quality (lower is better)",
    )
    axes[0].set_xticks([540, 960, 2160], ["30×18", "40×24", "60×36"])
    axes[0].grid(alpha=0.22)
    axes[0].legend(ncol=2, fontsize=7, loc="best")
    axes[0].text(
        0.02,
        0.02,
        "Red ×: constraint violation",
        transform=axes[0].transAxes,
        fontsize=7,
        color="#b71c1c",
    )

    methods = ["MODE", "MAP-Elites"]
    colors = {"training_quality": "#42a5f5", "validation_quality": "#ff8f00"}
    labels = {"training_quality": "Training", "validation_quality": "Held out"}
    for method_index, method in enumerate(methods):
        selected = [row for row in replications if row["method"] == method]
        for metric_index, metric in enumerate(["training_quality", "validation_quality"]):
            sample = [float(row[metric]) for row in selected]
            x = method_index + (-0.14 if metric_index == 0 else 0.14)
            axes[1].bar(
                x,
                statistics.mean(sample),
                width=0.24,
                yerr=statistics.stdev(sample),
                capsize=4,
                color=colors[metric],
                alpha=0.72,
                label=labels[metric] if method_index == 0 else None,
            )
            for jitter, value in zip([-0.045, 0.0, 0.045], sample, strict=True):
                axes[1].scatter(
                    [x + jitter],
                    [value],
                    color="#212121",
                    s=19,
                    zorder=5,
                )
    axes[1].set(
        title="Three optimizer seeds",
        ylabel="Reporting scalar quality (lower is better)",
        xticks=[0, 1],
        xticklabels=methods,
    )
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(loc="best")
    figure.suptitle("Numerical sensitivity and held-out robustness", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def render_publication(
    mode_directory: Path,
    qd_directory: Path,
    verification_directory: Path,
    seed_root: Path,
    summary: Path,
    output_directory: Path,
) -> None:
    configure()
    plot_mode(mode_directory, output_directory / "mode-results.svg")
    plot_qd(qd_directory, output_directory / "qd-results.svg")
    plot_fields(verification_directory, output_directory / "flow-fields.svg")
    plot_verification(
        verification_directory,
        seed_root,
        output_directory / "verification-results.svg",
        summary,
    )


def main() -> int:
    tutorial = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode-dir", type=Path, default=tutorial / "results/mode-seed-42"
    )
    parser.add_argument(
        "--qd-dir", type=Path, default=tutorial / "results/qd-seed-42"
    )
    parser.add_argument(
        "--verification-dir", type=Path, default=tutorial / "results/verification"
    )
    parser.add_argument("--seed-root", type=Path, default=tutorial / "results")
    parser.add_argument(
        "--summary",
        type=Path,
        default=tutorial / "results/replication-summary.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=tutorial / "images")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            generated_images = temporary_root / "images"
            generated_summary = temporary_root / "replication-summary.csv"
            render_publication(
                arguments.mode_dir,
                arguments.qd_dir,
                arguments.verification_dir,
                arguments.seed_root,
                generated_summary,
                generated_images,
            )
            stale = [
                arguments.output_dir / name
                for name in [
                    "mode-results.svg",
                    "qd-results.svg",
                    "flow-fields.svg",
                    "verification-results.svg",
                ]
                if not (arguments.output_dir / name).is_file()
                or not filecmp.cmp(
                    generated_images / name, arguments.output_dir / name, shallow=False
                )
            ]
            if not arguments.summary.is_file() or not filecmp.cmp(
                generated_summary, arguments.summary, shallow=False
            ):
                stale.append(arguments.summary)
            if stale:
                print("missing or stale room-ventilation artifacts:")
                for path in stale:
                    print(path)
                return 1
        print("room-ventilation figures and summary are current")
        return 0
    render_publication(
        arguments.mode_dir,
        arguments.qd_dir,
        arguments.verification_dir,
        arguments.seed_root,
        arguments.summary,
        arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
