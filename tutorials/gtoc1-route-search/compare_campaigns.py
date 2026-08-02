#!/usr/bin/env python3
"""Render a matched MGA route-discovery comparison from campaign artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARMS = ("gemma4", "random", "evolutionary")
ASSISTED_ARM = "gemma4-assisted"
LEGACY_ARMS = ("agent", "random", "evolutionary")
L0_CONSTRAINT_THRESHOLD = 1.0e-8


def arm_directory(root: Path, arm: str) -> Path:
    """Accept ``agent`` as a backwards-compatible Gemma directory name."""
    direct = root / arm
    if direct.is_dir():
        return direct
    if arm == "gemma4" and (root / "agent").is_dir():
        return root / "agent"
    return direct


def load_mga_arm(directory: Path) -> dict[str, object]:
    with (directory / "run.json").open(encoding="utf-8") as stream:
        run = json.load(stream)
    with (directory / "archive.csv").open(encoding="utf-8", newline="") as stream:
        archive = list(csv.DictReader(stream))
    qualified = [
        row
        for row in archive
        if row["evaluation_found"].lower() == "true"
        and math.isfinite(float(row["mga_score"]))
        and float(row["mga_score"]) > 0.0
    ]
    ranked = sorted(
        qualified, key=lambda row: float(row["mga_score"]), reverse=True
    )
    portfolio_target = int(run["configuration"].get("portfolio_size", 10))
    portfolio = ranked[:portfolio_target]
    budget = run["budget"]
    return {
        "run": run,
        "status": run["status"],
        "target": run["configuration"]["accepted_candidates"],
        "attempt_limit": run["configuration"]["maximum_proposal_attempts"],
        "accepted": budget["accepted_candidates"],
        "attempts": budget["proposal_attempts"],
        "qualified": len(qualified),
        "best_score": float(ranked[0]["mga_score"]) if ranked else None,
        "best_variant": ranked[0]["variant_key"] if ranked else None,
        "portfolio_size": len(portfolio),
        "portfolio_sum": sum(float(row["mga_score"]) for row in portfolio),
        "niches": budget["niches"],
        "worker_seconds": budget["l0_worker_seconds"],
        "wall_seconds": run["elapsed_seconds"],
        "actual_evaluations": run["actual_evaluations"],
        "tokens": budget["agent_input_tokens"] + budget["agent_output_tokens"],
        "duplicates": budget.get("duplicate_variants", 0)
        + budget.get("duplicate_sequences", 0),
        "diversity_rejections": budget["diversity_rejections"],
        "transport_failures": budget["transport_failures"],
    }


def comparable_mga_configuration(rows: dict[str, dict[str, object]]) -> bool:
    """Check all numerical and outer-protocol fields shared by the arms."""
    keys = (
        "accepted_candidates",
        "maximum_proposal_attempts",
        "bootstrap_candidates",
        "protected_top",
        "portfolio_size",
        "root_seed",
        "maximum_level",
        "grammar",
        "derivation",
        "inner_budget",
    )
    configurations = [rows[arm]["run"]["configuration"] for arm in ARMS]
    return all(
        all(configuration[key] == configurations[0][key] for key in keys)
        for configuration in configurations[1:]
    )


def render_mga(root: Path) -> str:
    rows = {arm: load_mga_arm(arm_directory(root, arm)) for arm in ARMS}
    arms = list(ARMS)
    assisted_directory = root / ASSISTED_ARM
    if assisted_directory.is_dir():
        rows[ASSISTED_ARM] = load_mga_arm(assisted_directory)
        arms.append(ASSISTED_ARM)
    matched = comparable_mga_configuration(rows)
    completed = all(rows[arm]["status"] == "completed" for arm in arms)
    text = [
        "# Matched MGA route-discovery comparison",
        "",
        (
            "The Gemma 4, grammar-random, and evolutionary arms use the same "
            "accepted-candidate target, attempt ceiling, canonical direction policy, "
            "duplicate filter, inner DE–CMA-ES budget, workers, root seed, and top-N metric."
            if matched
            else "**Configuration mismatch:** these arms are not a defensible comparison."
        ),
        "",
        "A route is *MGA-qualified* when the Rust optimizer returns a finite impulsive "
        "MGA score. This is a downstream candidate, not a continuous-thrust GTOC1 solution.",
        "",
        "| Arm | Status | Accepted / target | MGA-qualified | Best score | Top-N | Top-N sum | Niches |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        row = rows[arm]
        best = "—" if row["best_score"] is None else f"{row['best_score']:.3f}"
        text.append(
            f"| {arm} | {row['status']} | {row['accepted']} / {row['target']} | "
            f"{row['qualified']} | {best} | {row['portfolio_size']} | "
            f"{row['portfolio_sum']:.3f} | {row['niches']} |"
        )
    text.extend(
        [
            "",
            "| Arm | Attempts / ceiling | Duplicate rejected | Diversity rejected | "
            "Transport failed | MGA evaluations | Worker-h | Wall-h | Agent tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        row = rows[arm]
        text.append(
            f"| {arm} | {row['attempts']} / {row['attempt_limit']} | "
            f"{row['duplicates']} | {row['diversity_rejections']} | "
            f"{row['transport_failures']} | {row['actual_evaluations']} | "
            f"{row['worker_seconds'] / 3600:.3f} | {row['wall_seconds'] / 3600:.3f} | "
            f"{row['tokens']} |"
        )
    text.append("")
    if ASSISTED_ARM in rows:
        cold = rows["gemma4"]
        assisted = rows[ASSISTED_ARM]
        portfolio_gain = 100.0 * (
            assisted["portfolio_sum"] / cold["portfolio_sum"] - 1.0
        )
        wall_speedup = cold["wall_seconds"] / assisted["wall_seconds"]
        text.extend(
            [
                "The first three rows are the blind matched comparison. "
                "`gemma4-assisted` is a separately named, prior-informed follow-up: "
                "it uses the completed random and evolutionary archives to construct "
                "a length-stratified candidate menu and therefore is not a fourth "
                "independent arm.",
                "",
                f"Relative to cold Gemma, the assisted follow-up improves the top-N "
                f"sum by {portfolio_gain:.1f}% and completes in {wall_speedup:.2f}× "
                "less wall time. These are one-seed protocol results, not a general "
                "model-capability estimate.",
                "",
            ]
        )
    text.extend(
        [
            (
                "Every reported arm completed. Repeat the blind matched protocol and "
                "the explicitly prior-informed follow-up across predeclared seeds "
                "before drawing a proposer-capability conclusion."
                if completed
                else "At least one arm is incomplete; retain this as work in progress, not a comparison result."
            ),
            "",
        ]
    )
    return "\n".join(text)


def legacy_arm_directory(root: Path, arm: str) -> Path:
    """Resolve the retained endpoint-repair/Sims–Flanagan audit layout."""

    repaired = root / "evolutionary-repaired"
    if arm == "evolutionary" and repaired.is_dir():
        return repaired
    return root / arm


def load_legacy_arm(directory: Path) -> dict[str, object]:
    """Load one retained pre-MGA L0/L1 audit arm."""

    with (directory / "run.json").open(encoding="utf-8") as stream:
        run = json.load(stream)
    with (directory / "archive.csv").open(encoding="utf-8", newline="") as stream:
        archive = list(csv.DictReader(stream))
    finite = [row for row in archive if math.isfinite(float(row["constraint_l0"]))]
    feasible = [
        row
        for row in finite
        if float(row["constraint_l0"]) <= L0_CONSTRAINT_THRESHOLD
    ]
    gaps = [float(row["surrogate_gap"]) for row in archive if row.get("surrogate_gap")]
    budget = run["budget"]
    best_feasible = max(
        feasible,
        key=lambda row: float(row["estimated_score_l0"]),
        default=None,
    )
    return {
        "run": run,
        "status": run["status"],
        "target": run["configuration"]["accepted_candidates"],
        "attempt_limit": run["configuration"]["maximum_proposal_attempts"],
        "accepted": budget["accepted_candidates"],
        "attempts": budget["proposal_attempts"],
        "l0_feasible": len(feasible),
        "lowest_l0_constraint": min(
            (float(row["constraint_l0"]) for row in finite), default=math.inf
        ),
        "best_feasible_score": (
            float(best_feasible["estimated_score_l0"])
            if best_feasible is not None
            else None
        ),
        "best_feasible_variant": (
            best_feasible["variant_key"] if best_feasible is not None else None
        ),
        "l1_promotions": budget.get("l1_promotions", 0),
        "l1_passed": budget.get("l1_threshold_passed", 0),
        "niches": budget["niches"],
        "mean_gap": sum(gaps) / len(gaps) if gaps else None,
        "worker_seconds": budget["l0_worker_seconds"]
        + budget.get("l1_worker_seconds", 0.0),
        "wall_seconds": run["elapsed_seconds"],
        "actual_evaluations": run["actual_evaluations"],
        "tokens": budget["agent_input_tokens"] + budget["agent_output_tokens"],
        "diversity_rejections": budget["diversity_rejections"],
        "transport_failures": budget["transport_failures"],
    }


def comparable_legacy_configuration(rows: dict[str, dict[str, object]]) -> bool:
    """Check fields shared by the retained numerical comparison arms."""

    keys = (
        "accepted_candidates",
        "maximum_proposal_attempts",
        "bootstrap_candidates",
        "protected_top",
        "root_seed",
        "maximum_level",
        "grammar",
        "derivation",
        "inner_budget",
        "promotion",
        "refinement",
    )
    configurations = []
    for arm in LEGACY_ARMS:
        configuration = json.loads(json.dumps(rows[arm]["run"]["configuration"]))
        configuration["promotion"].setdefault("variants", [])
        configurations.append(configuration)
    return all(
        all(configuration[key] == configurations[0][key] for key in keys)
        for configuration in configurations[1:]
    )


def render_legacy(root: Path) -> str:
    """Preserve deterministic checks for the historical L0/L1 evidence."""

    rows = {
        arm: load_legacy_arm(legacy_arm_directory(root, arm)) for arm in LEGACY_ARMS
    }
    live = int(rows["agent"]["tokens"]) > 0
    matched_config = comparable_legacy_configuration(rows)
    completed = [arm for arm in LEGACY_ARMS if rows[arm]["status"] == "completed"]
    all_completed = len(completed) == len(LEGACY_ARMS)
    title = (
        "# Live L0 seed-42 route-search audit"
        if live
        else "# Offline route-search protocol comparison"
    )
    text = [
        title,
        "",
        (
            "The three manifests request the same accepted-candidate target, "
            "proposal ceiling, L0 inner budget, variant cap, worker allocation, "
            "root seed, and promotion policy."
            if matched_config
            else "**Configuration mismatch:** these arms are not comparable."
        ),
        "",
    ]
    if live:
        completion = (
            "All three arms completed after the evolutionary baseline was repaired."
            if all_completed
            else "Not every configured arm completed."
        )
        text.extend(
            [
                "This is one MiniMax-M3 seed-42 L0 audit, not an agent-capability "
                f"conclusion. {completion} L0 feasibility means only that the "
                "Lambert screen's launch and periapsis constraints pass. Scores "
                "remain surrogate diagnostics until L1/L2 validation.",
                "",
            ]
        )
    else:
        text.extend(
            [
                "This is a transport and protocol fixture, not an "
                "agent-capability comparison. The `agent` arm is a deterministic "
                "mock whose first three proposals are historical routes.",
                "",
            ]
        )
    text.extend(
        [
            "| Arm | Status | Accepted / target | L0 admissible | "
            "Lowest L0 violation | Best admissible L0 diagnostic | Niches |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in LEGACY_ARMS:
        row = rows[arm]
        diagnostic = (
            "—"
            if row["best_feasible_score"] is None
            else f"{row['best_feasible_score']:.3f}"
        )
        text.append(
            f"| {arm} | {row['status']} | {row['accepted']} / {row['target']} | "
            f"{row['l0_feasible']} | {row['lowest_l0_constraint']:.6g} | "
            f"{diagnostic} | {row['niches']} |"
        )
    text.extend(
        [
            "",
            "| Arm | Attempts / ceiling | Diversity rejected | Transport failed | "
            "Actual L0 evaluations | Worker-h | Wall-h | Agent tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in LEGACY_ARMS:
        row = rows[arm]
        text.append(
            f"| {arm} | {row['attempts']} / {row['attempt_limit']} | "
            f"{row['diversity_rejections']} | {row['transport_failures']} | "
            f"{row['actual_evaluations']} | {row['worker_seconds'] / 3600:.3f} | "
            f"{row['wall_seconds'] / 3600:.3f} | {row['tokens']} |"
        )
    text.append("")
    if live:
        random_best = rows["random"]["best_feasible_variant"]
        random_score = rows["random"]["best_feasible_score"]
        text.extend(
            [
                f"The random arm's leading L0-admissible variant is "
                f"`{random_best}` with diagnostic estimated score "
                f"`{random_score:.3f}`. It is not an impulsive or "
                "continuous-thrust-feasible GTOC1 solution.",
                "",
            ]
        )
        if rows["evolutionary"]["status"] == "completed":
            text.extend(
                [
                    "The completed evolutionary arm uses independent grammar-random "
                    "bootstrap seeds and random immigrants during exploration; "
                    "exploitation still mutates feasibility-first elites. The original "
                    "one-route bootstrap failure and the bootstrap-only 39-route "
                    "saturation run remain preserved separately as protocol evidence.",
                    "",
                ]
            )
        else:
            text.extend(
                [
                    "The evolutionary arm is not negative optimizer evidence. Its "
                    "first random bootstrap route was accepted; each later one-edit "
                    "mutation was rejected by the bootstrap minimum edit distance "
                    "of three, leaving the archive below the six-candidate bootstrap "
                    "threshold for all 119 remaining attempts.",
                    "",
                ]
            )
    if not all_completed:
        completed_text = (
            " and ".join(completed)
            if len(completed) <= 2
            else ", ".join(completed[:-1]) + f", and {completed[-1]}"
        )
        text.extend(
            [
                f"Only {completed_text} completed. The failed arm makes "
                "this an audit, not a completed matched three-arm comparison.",
                "",
            ]
        )
    if all(int(rows[arm]["l1_promotions"]) == 0 for arm in LEGACY_ARMS):
        text.extend(
            [
                "No arm ran L1, so there are no Sims–Flanagan promotions or "
                "measured surrogate gaps.",
                "",
            ]
        )
    return "\n".join(text)


def render(root: Path) -> str:
    """Dispatch by archive schema so retained historical evidence stays auditable."""

    mga_archive = arm_directory(root, "gemma4") / "archive.csv"
    legacy_archive = legacy_arm_directory(root, "agent") / "archive.csv"
    probe = mga_archive if mga_archive.is_file() else legacy_archive
    with probe.open(encoding="utf-8", newline="") as stream:
        fields = next(csv.reader(stream))
    if "mga_score" in fields:
        return render_mga(root)
    if "constraint_l0" in fields:
        return render_legacy(root)
    raise ValueError(f"unsupported route-search archive schema: {probe}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/mga-matched-seed42"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output or arguments.results / "comparison.md"
    rendered = render(arguments.results)
    if arguments.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"missing or stale comparison: {output}")
        print(f"comparison is current: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
