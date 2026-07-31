#!/usr/bin/env python3
"""Render deterministic foundations diagrams and measured result figures."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import math
import statistics
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "publication"
IMAGES = ROOT / "images"
FIGURES = [
    "architecture.svg",
    "indicator-geometry.svg",
    "lesson-ladder.svg",
    "campaign-results.svg",
    "lennard-jones-workflow.svg",
    "lennard-jones-encoding.svg",
    "lennard-jones-scaling.svg",
    "lennard-jones-pilot.svg",
]
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#78909C"


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
            "svg.hashsalt": "fcmaes-foundations-v1",
        }
    )


def validate_artifacts() -> None:
    root = json.loads((RESULTS / "run.json").read_text(encoding="utf-8"))
    if root["schema_version"] != 2 or root["status"] != "completed":
        raise ValueError("publication campaign is not completed schema-v2 conformance evidence")
    if root.get("claim_scope") != (
        "deterministic-conformance-demonstration; not a statistical optimizer benchmark"
    ):
        raise ValueError("publication campaign overstates its statistical claim scope")
    if root["skipped_gates"] != ["wfg", "bbob"]:
        raise ValueError("publication gate set changed")
    scalar = rows(RESULTS / "so" / "arms.csv")
    multi = rows(RESULTS / "mo" / "indicators.csv")
    fronts = rows(RESULTS / "mo" / "fronts.csv")
    convergence = rows(RESULTS / "mo" / "convergence.csv")
    if len(scalar) != 24 or {row["arm"] for row in scalar} != {
        "initial-population",
        "random",
        "de",
    }:
        raise ValueError("single-objective evidence lacks its three controls")
    if len(multi) != 36 or {row["arm"] for row in multi} != {
        "initial",
        "random",
        "mode",
    }:
        raise ValueError("multi-objective evidence lacks its three controls")
    if any(
        int(row["deterministic_recheck_points"]) != int(row["front_size"])
        or float(row["deterministic_recheck_max_abs_error"]) != 0.0
        or float(row["hypervolume"]) <= 0.0
        or len(json.loads(row["normalization_ideal"])) != int(row["objectives"])
        or len(json.loads(row["normalization_nadir"])) != int(row["objectives"])
        for row in multi
    ):
        raise ValueError("primary hypervolume, deterministic recheck, or normalization evidence is incomplete")
    for problem in {row["problem"] for row in multi}:
        references = {row["reference_point"] for row in multi if row["problem"] == problem}
        if len(references) != 1:
            raise ValueError(f"{problem} arms do not share one campaign reference point")
    for row in multi:
        outside = int(row["fixed_outside_reference"])
        fixed = row["fixed_hypervolume"]
        kind = row["fixed_hypervolume_kind"]
        if outside and (fixed or kind != "not-applicable-outside-reference"):
            raise ValueError("fixed-box hypervolume filtered an out-of-box front")
        if not outside and (not fixed or kind not in {"exact", "monte-carlo"}):
            raise ValueError("eligible fixed-box hypervolume is missing")
    if len(fronts) != sum(int(row["front_size"]) for row in multi):
        raise ValueError("front decisions do not reproduce the indicator row counts")
    if len(convergence) != 48:
        raise ValueError("MODE convergence needs four checkpoints per problem")
    if any(float(row["hypervolume"]) <= 0.0 for row in convergence):
        raise ValueError("a published convergence checkpoint has degenerate primary hypervolume")
    for suite in ["wfg", "bbob"]:
        gate = json.loads((RESULTS / suite / "run.json").read_text(encoding="utf-8"))
        if (
            gate.get("schema_version") != 2
            or gate.get("status") != "skipped"
            or gate.get("reason") != "reference-fixtures-unavailable"
            or gate.get("preset") != "publication"
            or gate.get("seed") != 42
            or not gate.get("command")
        ):
            raise ValueError(f"{suite} evidence gate is not explicit")
    expected = (ROOT / "results" / "expected" / "ladder.txt").read_text(encoding="utf-8")
    actual = (RESULTS / "ladder" / "output.txt").read_text(encoding="utf-8")
    if actual != expected:
        raise ValueError("publication ladder differs from its reviewed fixture")
    lj_root = json.loads((RESULTS / "lennard-jones" / "run.json").read_text(encoding="utf-8"))
    lj_rows = rows(RESULTS / "lennard-jones" / "scaling.csv")
    if (
        lj_root.get("schema_version") != 2
        or lj_root.get("status") != "completed"
        or lj_root.get("preset") != "publication"
        or lj_root.get("seeds_per_case") != 10
        or len(lj_rows) != 700
    ):
        raise ValueError("Lennard-Jones publication evidence is incomplete")
    required_arms = {
        "random",
        "lbfgs-multistart",
        "basin-hopping",
        "de-retry",
        "cma-retry",
        "crfmnes-retry",
        "bite-retry",
    }
    if {row["optimizer"] for row in lj_rows} != required_arms:
        raise ValueError("Lennard-Jones evidence lacks a mandatory arm")
    if any(
        int(row["pair_traversals"]) > int(lj_root["pair_traversal_budget_per_arm"])
        or int(row["pair_terms_evaluated"])
        != int(row["pair_traversals"])
        * int(row["n_atoms"])
        * (int(row["n_atoms"]) - 1)
        // 2
        or not math.isfinite(float(row["best_energy"]))
        or not math.isfinite(float(row["target_relative_gap"]))
        or (row["success"] == "true")
        != (float(row["gap"]) <= float(lj_root["success_tolerance"]))
        or (row["within_1_percent"] == "true")
        != (float(row["target_relative_gap"]) <= 0.01)
        or (row["within_5_percent"] == "true")
        != (float(row["target_relative_gap"]) <= 0.05)
        or (row["within_10_percent"] == "true")
        != (float(row["target_relative_gap"]) <= 0.10)
        or row["reference_structure_audited"] != "true"
        for row in lj_rows
    ):
        raise ValueError("Lennard-Jones budget or finite-result contract failed")
    audit = json.loads(
        (RESULTS / "lennard-jones" / "reference-audit.json").read_text(encoding="utf-8")
    )
    if (
        audit.get("status") != "completed"
        or not audit.get("all_match")
        or audit.get("actual_evaluations") != 5
        or len(audit.get("audits", [])) != 5
        or any(
            len(row.get("coordinate_sha256", "")) != 64
            or float(row["absolute_error"]) > float(row["tolerance"])
            for row in audit.get("audits", [])
        )
        or not lj_root.get("reference_structure_audited")
    ):
        raise ValueError("Lennard-Jones reference structures were not independently audited")
    configuration = rows(RESULTS / "lennard-jones" / "crfmnes-configuration.csv")
    if (
        len(configuration) != 81
        or sum(row["primary_configuration"] == "true" for row in configuration) != 9
        or any(
            int(row["pair_traversals"]) != int(lj_root["pair_traversal_budget_per_arm"])
            for row in configuration
        )
    ):
        raise ValueError("CR-FM-NES configuration sensitivity evidence is incomplete")
    pilot = json.loads(
        (RESULTS / "lennard-jones" / "pilot" / "run.json").read_text(encoding="utf-8")
    )
    qd = json.loads((RESULTS / "lennard-jones" / "qd" / "run.json").read_text(encoding="utf-8"))
    if (
        pilot.get("verdict") != "rejected"
        or qd.get("status") != "skipped"
        or qd.get("actual_evaluations") is not None
        or qd.get("artifacts") != {}
    ):
        raise ValueError("Lennard-Jones descriptor verdict and QD decision disagree")


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "foundations/plot_results.py"},
    )
    plt.close(figure)
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n",
        encoding="utf-8",
    )


def box(axis: plt.Axes, x: float, y: float, width: float, text: str, color: str) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.8,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#455A64",
            linewidth=1.1,
        )
    )
    axis.text(x + width / 2, y + 0.4, text, ha="center", va="center")


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": "#455A64", "lw": 1.2},
    )


def architecture(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 3.0))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4)
    axis.axis("off")
    box(axis, 0.3, 2.5, 2.1, "Known suites\nclassic · ZDT · DTLZ", "#DDEBF7")
    box(axis, 0.3, 0.7, 2.1, "Local-only gates\nCEC · WFG · BBOB", "#ECEFF1")
    box(axis, 3.3, 1.6, 2.0, "Rust evaluator\nfallible + bounded", "#E8F5E9")
    box(axis, 6.2, 2.5, 2.0, "DE / MODE\nrandom + baseline", "#FFF3E0")
    box(axis, 6.2, 0.7, 2.0, "Core indicators\nexact or typed MC", "#F3E5F5")
    box(axis, 9.2, 1.6, 2.2, "Schema-v1 evidence\nCSV · JSON · SVG", "#E0F2F1")
    for start, end in [
        ((2.4, 2.9), (3.3, 2.1)),
        ((2.4, 1.1), (3.3, 1.9)),
        ((5.3, 2.1), (6.2, 2.9)),
        ((5.3, 1.9), (6.2, 1.1)),
        ((8.2, 2.9), (9.2, 2.1)),
        ((8.2, 1.1), (9.2, 1.9)),
    ]:
        arrow(axis, start, end)
    axis.set_title("One benchmark definition feeds optimization and independent measurement")
    save(figure, path)


def indicator_geometry(path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    front = [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)]
    reference = (5.0, 5.0)
    for point, color in zip(front, [BLUE, GREEN, ORANGE], strict=True):
        axes[0].add_patch(
            Rectangle(
                point,
                reference[0] - point[0],
                reference[1] - point[1],
                facecolor=color,
                edgecolor=color,
                alpha=0.18,
            )
        )
    axes[0].scatter(*zip(*front, strict=True), color=BLUE, zorder=3, label="approximation")
    axes[0].scatter(*reference, color="black", marker="x", s=60, label="reference (5, 5)")
    axes[0].set(xlim=(0, 5.3), ylim=(0, 5.3), xlabel="$f_1$ (min)", ylabel="$f_2$ (min)")
    axes[0].set_title("Hypervolume = union of dominated boxes")
    axes[0].legend(loc="lower left")

    analytic = [(x / 20, 1.0 - math.sqrt(x / 20)) for x in range(21)]
    approximation = [(0.1, 0.85), (0.35, 0.62), (0.7, 0.28)]
    axes[1].plot(*zip(*analytic, strict=True), color=GREY, label="analytic front")
    axes[1].scatter(*zip(*approximation, strict=True), color=PURPLE, label="measured front")
    for point in approximation:
        nearest = min(analytic, key=lambda target: math.dist(point, target))
        axes[1].plot([point[0], nearest[0]], [point[1], nearest[1]], color=PURPLE, ls="--", lw=1)
    axes[1].set(xlabel="normalized $f_1$", ylabel="normalized $f_2$", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    axes[1].set_title("IGD+ measures distance to a known front")
    axes[1].legend(loc="upper right")
    figure.tight_layout()
    save(figure, path)


def lesson_ladder(path: Path) -> None:
    labels = [
        "L1\nfirst run",
        "L2\ncompare",
        "L3\nseeds",
        "L4\nconstraints",
        "L5\nfronts",
        "L6\nmixed",
        "L7\narchive",
    ]
    colors = ["#DDEBF7", "#DDEBF7", "#E8F5E9", "#E8F5E9", "#F3E5F5", "#FFF3E0", "#E0F2F1"]
    figure, axis = plt.subplots(figsize=(9.2, 3.4))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 5.2)
    axis.axis("off")
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        x = 0.35 + index * 1.32
        y = 0.35 + index * 0.55
        box(axis, x, y, 1.08, label, color)
        if index:
            arrow(axis, (x - 0.24, y - 0.15), (x, y + 0.2))
    axis.text(0.4, 4.75, "Each rung adds one idea and one reviewed failure mode", fontsize=11)
    save(figure, path)


def campaign_results(path: Path) -> None:
    scalar = rows(RESULTS / "so" / "arms.csv")
    multi = rows(RESULTS / "mo" / "indicators.csv")
    problems = sorted({row["problem"] for row in scalar})
    so_gain = []
    for problem in problems:
        selected = {row["arm"]: float(row["best"]) for row in scalar if row["problem"] == problem}
        so_gain.append(math.log10(max(selected["random"], 1e-300) / max(selected["de"], 1e-300)))
    mo_problems = sorted({row["problem"] for row in multi})
    igd_gain = []
    hv_gain = []
    for problem in mo_problems:
        selected = {row["arm"]: row for row in multi if row["problem"] == problem}
        igd_gain.append(
            float(selected["random"]["igd_plus"])
            / max(float(selected["mode"]["igd_plus"]), 1e-300)
        )
        hv_gain.append(
            float(selected["mode"]["hypervolume"])
            / max(float(selected["random"]["hypervolume"]), 1e-300)
        )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    axes[0].bar(range(len(problems)), so_gain, color=BLUE)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(range(len(problems)), problems, rotation=45, ha="right")
    axes[0].set_ylabel("log10(random best / DE best)")
    axes[0].set_title("DE improvement at 4,000 evaluations")

    colors = [GREEN if value >= 1.0 else ORANGE for value in igd_gain]
    axes[1].bar(range(len(mo_problems)), igd_gain, color=colors)
    axes[1].axhline(1.0, color="black", lw=0.8)
    axes[1].set_xticks(range(len(mo_problems)), mo_problems, rotation=45, ha="right")
    axes[1].set_ylabel("random IGD+ / MODE IGD+")
    axes[1].set_title("IGD+ gain (higher is better)")

    colors = [GREEN if value >= 1.0 else ORANGE for value in hv_gain]
    axes[2].bar(range(len(mo_problems)), hv_gain, color=colors)
    axes[2].axhline(1.0, color="black", lw=0.8)
    axes[2].set_xticks(range(len(mo_problems)), mo_problems, rotation=45, ha="right")
    axes[2].set_ylabel("MODE HV / random HV")
    axes[2].set_title("Shared-reference HV gain")
    figure.tight_layout()
    save(figure, path)


def lennard_jones_workflow(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.4, 3.2))
    axis.set_xlim(0, 13.2)
    axis.set_ylim(0, 4.2)
    axis.axis("off")
    box(axis, 0.2, 1.7, 1.9, "Compact candidate\nseparation ≥ 0.75", "#DDEBF7")
    box(axis, 2.8, 1.7, 1.8, "Free 3N or\nfixed 3N−6", "#E8F5E9")
    box(axis, 5.3, 1.7, 1.9, "O(N²) pairs\nvalue + gradient", "#FFF3E0")
    box(axis, 7.9, 2.7, 2.0, "fcmaes-core\n4 × retry", "#F3E5F5")
    box(axis, 7.9, 0.7, 2.0, "argmin L-BFGS\nexternal adapter", "#F3E5F5")
    box(axis, 10.7, 1.7, 2.2, "Versioned evidence\ntarget · budget · time", "#E0F2F1")
    arrow(axis, (2.1, 2.1), (2.8, 2.1))
    arrow(axis, (4.6, 2.1), (5.3, 2.1))
    arrow(axis, (7.2, 2.2), (7.9, 3.0))
    arrow(axis, (7.2, 2.0), (7.9, 1.0))
    arrow(axis, (9.9, 3.0), (10.7, 2.3))
    arrow(axis, (9.9, 1.0), (10.7, 1.9))
    axis.set_title("The gradient solver crosses an adapter boundary; the model and evidence do not")
    save(figure, path)


def lennard_jones_encoding(path: Path) -> None:
    points = [
        (-0.9, -0.4), (-0.4, 0.75), (0.25, -0.75), (0.85, 0.45),
        (0.0, 0.0), (-0.7, 0.35), (0.65, -0.25),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].scatter(*zip(*points, strict=True), s=105, color=BLUE, edgecolor="white", zorder=3)
    axes[0].annotate("translation", xy=(0.55, 0.8), xytext=(-0.8, 0.8), arrowprops={"arrowstyle": "->"})
    axes[0].annotate("rotation", xy=(0.7, -0.6), xytext=(-0.85, -0.8), arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-0.3"})
    axes[0].set_title("Free: 3N coordinates, six null directions")
    axes[0].set_aspect("equal")
    axes[0].set(xlim=(-1.15, 1.15), ylim=(-1.05, 1.05), xlabel="x", ylabel="y")

    fixed = [(0.0, 0.0), (0.9, 0.0), (0.25, 0.8), (-0.55, 0.5), (0.65, 0.55), (-0.25, -0.65), (0.8, -0.5)]
    axes[1].scatter(*zip(*fixed, strict=True), s=105, color=GREEN, edgecolor="white", zorder=3)
    axes[1].scatter([0.0, 0.9, 0.25], [0.0, 0.0, 0.8], s=145, facecolor="none", edgecolor=ORANGE, linewidth=2, zorder=4)
    axes[1].text(0.0, -0.12, "atom 0", ha="center")
    axes[1].text(0.9, -0.12, "atom 1: +x", ha="center")
    axes[1].text(0.25, 0.92, "atom 2: +y half-plane", ha="center")
    axes[1].set_title("Fixed frame: same distances, 3N−6 decisions")
    axes[1].set_aspect("equal")
    axes[1].set(xlim=(-1.15, 1.15), ylim=(-1.05, 1.05), xlabel="canonical x", ylabel="canonical y")
    figure.tight_layout()
    save(figure, path)


def lennard_jones_scaling(path: Path) -> None:
    data = rows(RESULTS / "lennard-jones" / "scaling.csv")
    atoms = sorted({int(row["n_atoms"]) for row in data})
    arms = [
        "random", "de-retry", "cma-retry", "crfmnes-retry", "bite-retry",
        "lbfgs-multistart", "basin-hopping",
    ]
    colors = dict(zip(arms, plt.cm.tab10.colors[: len(arms)], strict=True))
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    for axis, parameterization in zip(axes[0], ["free", "fixed-frame"], strict=True):
        for arm in arms:
            rates = []
            for size in atoms:
                selected = [
                    row for row in data
                    if int(row["n_atoms"]) == size
                    and row["parameterization"] == parameterization
                    and row["optimizer"] == arm
                ]
                rates.append(sum(row["success"] == "true" for row in selected) / len(selected))
            axis.plot(atoms, rates, marker="o", color=colors[arm], label=arm)
        axis.set(xlabel="atoms N", ylabel="success fraction", ylim=(-0.03, 1.03))
        axis.set_title(f"{parameterization}: target + 1e−3")
    for arm in arms:
        relative_gaps = []
        for size in atoms:
            selected = [
                row for row in data
                if int(row["n_atoms"]) == size and row["optimizer"] == arm
            ]
            relative_gaps.append(
                statistics.median(float(row["target_relative_gap"]) for row in selected)
            )
        axes[1, 0].plot(atoms, relative_gaps, marker="o", color=colors[arm], label=arm)
    axes[1, 0].set(
        xlabel="atoms N",
        ylabel="median target-relative gap",
        yscale="log",
        title="Discriminating quality metric (both encodings)",
    )
    for arm in arms:
        dimensions = []
        overhead = []
        for size in atoms:
            selected = [
                row for row in data
                if int(row["n_atoms"]) == size
                and row["parameterization"] == "fixed-frame"
                and row["optimizer"] == arm
            ]
            dimensions.append(int(selected[0]["dimension"]))
            overhead.append(statistics.median(float(row["estimated_optimizer_overhead_seconds"]) for row in selected))
        axes[1, 1].plot(dimensions, overhead, marker="o", color=colors[arm], label=arm)
    axes[1, 1].set(
        xlabel="fixed-frame dimension",
        ylabel="median estimated overhead (s)",
        yscale="symlog",
        title="Diagnostic wall−pair time (host not reserved)",
    )
    handles, labels = axes[1, 1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    save(figure, path)


def lennard_jones_pilot(path: Path) -> None:
    data = rows(RESULTS / "lennard-jones" / "pilot" / "pilot.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for arm, color in zip(range(3), [BLUE, GREEN, ORANGE], strict=True):
        selected = [row for row in data if int(row["seed_arm"]) == arm]
        axes[0].scatter(
            [float(row["radius_gyration_normalized"]) for row in selected],
            [float(row["mean_coordination"]) for row in selected],
            s=20, alpha=0.7, color=color, label=f"seed arm {arm + 1}",
        )
    axes[0].set(xlim=(0.25, 0.75), ylim=(0, 12), xlabel="normalized radius of gyration", ylabel="mean coordination", title="Reachable region in registered bounds")
    axes[0].legend(loc="upper right")
    train = [int(row["fine_niche"]) for row in data if row["fine_niche"]]
    holdout = [int(row["holdout_fine_niche"]) for row in data if row["holdout_fine_niche"]]
    train_grid = [[0] * 12 for _ in range(12)]
    holdout_grid = [[0] * 12 for _ in range(12)]
    for cell in train:
        train_grid[cell // 12][cell % 12] += 1
    for cell in holdout:
        holdout_grid[cell // 12][cell % 12] += 1
    image_plot = axes[1].imshow(train_grid, origin="lower", cmap="viridis", vmin=0)
    occupied = sum(value > 0 for row in train_grid for value in row)
    axes[1].set(xlabel="radius bin", ylabel="coordination bin", title=f"12×12 train occupancy: {occupied}/144 niches")
    figure.colorbar(image_plot, ax=axes[1], label="candidates")
    figure.tight_layout()
    save(figure, path)


def render(output: Path) -> None:
    configure()
    architecture(output / "architecture.svg")
    indicator_geometry(output / "indicator-geometry.svg")
    lesson_ladder(output / "lesson-ladder.svg")
    campaign_results(output / "campaign-results.svg")
    lennard_jones_workflow(output / "lennard-jones-workflow.svg")
    lennard_jones_encoding(output / "lennard-jones-encoding.svg")
    lennard_jones_scaling(output / "lennard-jones-scaling.svg")
    lennard_jones_pilot(output / "lennard-jones-pilot.svg")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=IMAGES)
    arguments = parser.parse_args()
    validate_artifacts()
    if arguments.check:
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory)
            render(generated)
            stale = [name for name in FIGURES if not filecmp.cmp(generated / name, IMAGES / name, shallow=False)]
            if stale:
                print("missing or stale foundations figures:")
                for name in stale:
                    print(IMAGES / name)
                return 1
        print("foundations figures are current")
        return 0
    render(arguments.output)
    for name in FIGURES:
        print(arguments.output / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
