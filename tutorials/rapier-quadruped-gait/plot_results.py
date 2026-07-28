#!/usr/bin/env python3
"""Render deterministic quadruped figures from native Rust artifacts."""

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
            "svg.hashsalt": "fcmaes-rapier-quadruped-gait-v1",
        }
    )


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "rapier-quadruped-gait/plot_results.py"},
    )
    plt.close(figure)
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n",
        encoding="utf-8",
    )


def validate_artifacts() -> None:
    scalar = json.loads((RESULTS / "scalar" / "run.json").read_text(encoding="utf-8"))
    qd = json.loads((RESULTS / "qd" / "run.json").read_text(encoding="utf-8"))
    if scalar["tutorial"] != "rapier-quadruped-gait" or qd["schema_version"] != 1:
        raise ValueError("unexpected locomotion manifests")
    archive = rows(RESULTS / "qd" / "qd_archive.csv")
    if len(archive) != qd["occupied"]:
        raise ValueError("QD occupied count disagrees with qd_archive.csv")
    if any(float(row["constraint_fall_m_train"]) > 0 for row in archive):
        raise ValueError("fallen gait retained in training archive")
    if any(float(row["constraint_drift_m_train"]) > 0 for row in archive):
        raise ValueError("laterally infeasible gait retained in training archive")


def architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 3.2))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3.2)
    axis.axis("off")
    boxes = [
        (0.15, 1.1, 1.65, 0.9, "25 CPG controls\n8 joint targets", "#E3F2FD"),
        (2.15, 1.1, 1.65, 0.9, "Rapier 3D\n9 bodies · rough strip", "#E8F5E9"),
        (4.15, 1.1, 1.65, 0.9, "measured behavior\ncontacts · work · motion", "#FFF3E0"),
        (6.25, 1.75, 3.2, 0.75, "MAP-Elites repertoire\nduty factor × torso bob", "#F3E5F5"),
        (6.25, 0.55, 3.2, 0.75, "BiteOpt retry baseline\none distance/work optimum", "#FCE4EC"),
    ]
    for x, y, width, height, label, color in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="#37474F",
                linewidth=1.1,
            )
        )
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    for start, end in [
        ((1.8, 1.55), (2.15, 1.55)),
        ((3.8, 1.55), (4.15, 1.55)),
        ((5.8, 1.55), (6.25, 2.12)),
        ((5.8, 1.55), (6.25, 0.92)),
    ]:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#455A64", "lw": 1.4},
        )
    axis.text(
        5,
        2.95,
        "Serial deterministic rollouts; fcmaes owns the worker pool",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    save(figure, output)


def archive_figure(output: Path) -> None:
    data = rows(RESULTS / "qd" / "qd_archive.csv")
    duty = np.array([float(row["descriptor_duty_factor_train"]) for row in data])
    bob = np.array(
        [float(row["descriptor_body_height_std_mm_train"]) for row in data]
    )
    distance = np.array([float(row["forward_distance_m_train"]) for row in data])
    robust = np.array(
        [float(row["validation_feasible_fraction"]) for row in data]
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    scatter = axes[0].scatter(
        duty,
        bob,
        c=distance,
        cmap="viridis",
        s=58,
        edgecolor="#263238",
        linewidth=0.35,
    )
    figure.colorbar(scatter, ax=axes[0]).set_label("Training distance (m)")
    axes[0].set(
        title=f"Training archive ({len(data)} occupied niches)",
        xlabel="Mean foot duty factor",
        ylabel="Torso-height standard deviation (mm)",
        xlim=(0, 1),
        ylim=(0, 200),
    )
    scatter = axes[1].scatter(
        duty,
        bob,
        c=robust,
        cmap="plasma",
        vmin=0,
        vmax=1,
        s=58,
        edgecolor="#263238",
        linewidth=0.35,
    )
    figure.colorbar(scatter, ax=axes[1]).set_label("Held-out feasible fraction")
    axes[1].set(
        title="Same elites on five unseen terrains",
        xlabel="Training duty factor",
        ylabel="Training torso bob (mm)",
        xlim=(0, 1),
        ylim=(0, 200),
    )
    figure.suptitle("A gait repertoire is not automatically a robust repertoire", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def gait_strip(output: Path) -> None:
    labels = [
        ("low-duty", "Low duty"),
        ("mid-duty", "Middle duty"),
        ("high-duty", "High duty"),
    ]
    feet = [
        ("front_left", "FL"),
        ("front_right", "FR"),
        ("rear_left", "RL"),
        ("rear_right", "RR"),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(9.4, 5.3), sharex=True)
    for axis, (file_label, title) in zip(axes, labels, strict=True):
        data = rows(RESULTS / "qd" / f"replay-{file_label}.csv")
        times = np.array([float(row["time_s"]) for row in data])
        for foot_index, (column, short) in enumerate(feet):
            contact = np.array([int(row[column]) for row in data], dtype=bool)
            axis.fill_between(
                times,
                foot_index - 0.34,
                foot_index + 0.34,
                where=contact,
                step="post",
                color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"][foot_index],
                alpha=0.88,
            )
        axis.set_yticks(range(4), [short for _, short in feet])
        axis.set_ylim(-0.6, 3.6)
        axis.set_title(title, loc="left")
    axes[-1].set_xlabel("Rollout time (s)")
    figure.suptitle("Foot-ground contact strips from three archive niches", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def baseline(output: Path) -> None:
    scalar = rows(RESULTS / "scalar" / "best.csv")[0]
    archive = rows(RESULTS / "qd" / "qd_archive.csv")
    qd = min(archive, key=lambda row: float(row["quality_train"]))
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    names = ["BiteOpt\nsingle gait", "MAP-Elites\nbest elite"]
    axes[0].bar(
        names,
        [
            float(scalar["forward_distance_m"]),
            float(qd["forward_distance_m_train"]),
        ],
        color=["#D55E00", "#0072B2"],
    )
    axes[0].set(title="Equal candidate budget", ylabel="Training distance (m)")
    axes[1].bar(
        names,
        [
            float(scalar["mechanical_work_j"]),
            float(qd["mechanical_work_j_train"]),
        ],
        color=["#D55E00", "#0072B2"],
    )
    axes[1].set(title="Physics replay", ylabel="Integrated motor work (J)")
    figure.suptitle("One optimum versus one repertoire", fontsize=13)
    figure.tight_layout()
    save(figure, output)


def robustness(output: Path) -> None:
    data = rows(RESULTS / "qd" / "qd_archive.csv")
    train = np.array([float(row["forward_distance_m_train"]) for row in data])
    held = np.array([float(row["forward_distance_m_validation"]) for row in data])
    feasible = np.array(
        [float(row["validation_feasible_fraction"]) for row in data]
    )
    limit_low = min(train.min(), held.min())
    limit_high = max(train.max(), held.max())
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    scatter = axis.scatter(
        train,
        held,
        c=feasible,
        cmap="plasma",
        vmin=0,
        vmax=1,
        s=52,
        edgecolor="#263238",
        linewidth=0.35,
    )
    axis.plot([limit_low, limit_high], [limit_low, limit_high], "--", color="#607D8B")
    figure.colorbar(scatter, ax=axis).set_label("Held-out feasible fraction")
    axis.set(
        title="Terrain generalization is measured, not assumed",
        xlabel="Training-terrain distance (m)",
        ylabel="Mean held-out distance (m)",
    )
    save(figure, output)


def render(directory: Path) -> None:
    configure()
    validate_artifacts()
    architecture(directory / "architecture.svg")
    archive_figure(directory / "archive.svg")
    gait_strip(directory / "gait-strip.svg")
    baseline(directory / "baseline.svg")
    robustness(directory / "robustness.svg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        render(IMAGES)
        print("rapier-quadruped-gait figures are current")
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
        print("missing or stale rapier-quadruped-gait figures:")
        for path in stale:
            print(path)
        print(
            f"renderer uses matplotlib {matplotlib.__version__}; "
            "checked-in figures use tutorials/python/requirements-lock.txt "
            "(matplotlib 3.11.1)"
        )
        return 1
    print("rapier-quadruped-gait figures are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
