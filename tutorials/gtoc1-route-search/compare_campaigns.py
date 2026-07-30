#!/usr/bin/env python3
"""Regenerate the matched-budget route-search comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARMS = ("agent", "random", "evolutionary")
L0_CONSTRAINT_THRESHOLD = 1.0e-8


def load_arm(directory: Path) -> dict[str, object]:
    with (directory / "run.json").open(encoding="utf-8") as stream:
        run = json.load(stream)
    with (directory / "archive.csv").open(encoding="utf-8", newline="") as stream:
        archive = list(csv.DictReader(stream))
    finite_constraints = [
        float(row["constraint_l0"])
        for row in archive
        if math.isfinite(float(row["constraint_l0"]))
    ]
    gaps = [
        float(row["surrogate_gap"])
        for row in archive
        if row.get("surrogate_gap")
    ]
    budget = run["budget"]
    return {
        "status": run["status"],
        "accepted": budget["accepted_candidates"],
        "l0_feasible": sum(
            constraint <= L0_CONSTRAINT_THRESHOLD
            for constraint in finite_constraints
        ),
        "lowest_l0_constraint": min(finite_constraints, default=math.inf),
        "l1_promotions": budget.get("l1_promotions", 0),
        "l1_passed": budget.get("l1_threshold_passed", 0),
        "niches": budget["niches"],
        "mean_gap": sum(gaps) / len(gaps) if gaps else None,
        "worker_seconds": budget["l0_worker_seconds"]
        + budget.get("l1_worker_seconds", 0.0),
        "wall_seconds": run["elapsed_seconds"],
        "tokens": budget["agent_input_tokens"] + budget["agent_output_tokens"],
    }


def render(root: Path) -> str:
    rows = {arm: load_arm(root / arm) for arm in ARMS}
    text = [
        "# Matched-budget route-search comparison",
        "",
        "All arms use the same accepted-candidate target, L0 inner budget, "
        "variant cap, worker allocation, root seed, and promotion policy.",
        "",
        "This is a transport and protocol fixture, not an agent-capability "
        "comparison. The `agent` arm is a deterministic mock whose first three "
        "proposals are the historical JPL, JPL2 and Jena routes. Route scores "
        "are intentionally omitted.",
        "",
        "| Arm | Status | Accepted | L0 feasible | Lowest L0 violation | "
        "L1 promotions | L1 passing | Niches | Mean surrogate gap | Worker-s | "
        "Wall-s | Agent tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = rows[arm]
        gap = "—" if row["mean_gap"] is None else f"{row['mean_gap']:.3f}"
        text.append(
            f"| {arm} | {row['status']} | {row['accepted']} | "
            f"{row['l0_feasible']} | {row['lowest_l0_constraint']:.6g} | "
            f"{row['l1_promotions']} | {row['l1_passed']} | {row['niches']} | "
            f"{gap} | {row['worker_seconds']:.3f} | "
            f"{row['wall_seconds']:.3f} | {row['tokens']} |"
        )
    text.extend(
        [
            "",
            "No arm ran L1, so this fixture contains no Sims–Flanagan "
            "promotions and no surrogate-gap observations.",
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
