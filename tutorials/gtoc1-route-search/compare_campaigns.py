#!/usr/bin/env python3
"""Regenerate a feasibility-first route-search campaign audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARMS = ("agent", "random", "evolutionary")
L0_CONSTRAINT_THRESHOLD = 1.0e-8


def arm_directory(root: Path, arm: str) -> Path:
    """Prefer the completed repaired evolutionary arm when it is present."""

    repaired = root / "evolutionary-repaired"
    if arm == "evolutionary" and repaired.is_dir():
        return repaired
    return root / arm


def load_arm(directory: Path) -> dict[str, object]:
    with (directory / "run.json").open(encoding="utf-8") as stream:
        run = json.load(stream)
    with (directory / "archive.csv").open(encoding="utf-8", newline="") as stream:
        archive = list(csv.DictReader(stream))
    finite = [
        row
        for row in archive
        if math.isfinite(float(row["constraint_l0"]))
    ]
    feasible = [
        row
        for row in finite
        if float(row["constraint_l0"]) <= L0_CONSTRAINT_THRESHOLD
    ]
    gaps = [
        float(row["surrogate_gap"])
        for row in archive
        if row.get("surrogate_gap")
    ]
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
            (float(row["constraint_l0"]) for row in finite),
            default=math.inf,
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


def comparable_configuration(rows: dict[str, dict[str, object]]) -> bool:
    """Check only fields that must match across numerical comparison arms."""

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
    configs = []
    for arm in ARMS:
        config = json.loads(json.dumps(rows[arm]["run"]["configuration"]))
        config["promotion"].setdefault("variants", [])
        configs.append(config)
    return all(
        all(config[key] == configs[0][key] for key in keys)
        for config in configs[1:]
    )


def render(root: Path) -> str:
    rows = {arm: load_arm(arm_directory(root, arm)) for arm in ARMS}
    live = int(rows["agent"]["tokens"]) > 0
    matched_config = comparable_configuration(rows)
    completed = [arm for arm in ARMS if rows[arm]["status"] == "completed"]
    all_completed = len(completed) == len(ARMS)
    title = (
        "# Live L0 seed-42 route-search audit"
        if live
        else "# Offline route-search protocol comparison"
    )
    text = [
        title,
        "",
        "The three manifests request the same accepted-candidate target, "
        "proposal ceiling, L0 inner budget, variant cap, worker allocation, "
        "root seed, and promotion policy."
        if matched_config
        else "**Configuration mismatch:** these arms are not comparable.",
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
    for arm in ARMS:
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
    for arm in ARMS:
        row = rows[arm]
        text.append(
            f"| {arm} | {row['attempts']} / {row['attempt_limit']} | "
            f"{row['diversity_rejections']} | {row['transport_failures']} | "
            f"{row['actual_evaluations']} | {row['worker_seconds'] / 3600:.3f} | "
            f"{row['wall_seconds'] / 3600:.3f} | {row['tokens']} |"
        )
    text.extend([""])
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
    if all(int(rows[arm]["l1_promotions"]) == 0 for arm in ARMS):
        text.extend(
            [
                "No arm ran L1, so there are no Sims–Flanagan promotions or "
                "measured surrogate gaps.",
                "",
            ]
        )
    return "\n".join(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/protocol-evidence"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output or args.results / "comparison.md"
    rendered = render(args.results)
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"missing or stale comparison: {output}")
        print(f"comparison is current: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
