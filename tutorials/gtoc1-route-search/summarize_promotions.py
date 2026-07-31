#!/usr/bin/env python3
"""Render the predeclared random-arm L1 follow-up and its status diagram."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


L0_CONSTRAINT_THRESHOLD = 1.0e-8


def load_rows(l0_archive: Path, promoted_archive: Path) -> list[dict[str, object]]:
    with l0_archive.open(encoding="utf-8", newline="") as stream:
        admissible = [
            row
            for row in csv.DictReader(stream)
            if row["evaluation_found"] == "true"
            and float(row["constraint_l0"]) <= L0_CONSTRAINT_THRESHOLD
        ]
    admissible.sort(key=lambda row: float(row["estimated_score_l0"]), reverse=True)
    ranks = {row["variant_key"]: index for index, row in enumerate(admissible, 1)}
    with promoted_archive.open(encoding="utf-8") as stream:
        promoted = {
            result["variant_key"]: result
            for result in json.load(stream)["results"]
            if result["l1"] is not None
        }
    selected_ranks = [1, (len(admissible) + 1) // 2, len(admissible)]
    rows = []
    for rank in selected_ranks:
        l0 = admissible[rank - 1]
        result = promoted[l0["variant_key"]]
        l1 = result["l1"]
        outcome = l1["outcome"]
        rows.append(
            {
                "rank": ranks[result["variant_key"]],
                "variant": result["variant_key"],
                "l0": float(l0["estimated_score_l0"]),
                "l1": l1["score"],
                "gap": result["surrogate_gap"],
                "mismatch": l1["maximum_normalized_mismatch"],
                "throttle": l1["maximum_throttle_norm"],
                "worker_seconds": float(l1["worker_seconds"]),
                "actual_evaluations": int(l1["actual_evaluations"]),
                "outcome": "passed"
                if l1["threshold_passed"]
                else outcome["code"],
                "message": "" if outcome is None else outcome["message"],
            }
        )
    return rows


def value(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def markdown(rows: list[dict[str, object]]) -> str:
    text = [
        "# Targeted random-arm L1 follow-up",
        "",
        "The controls were predeclared from the 15 L0-admissible random routes "
        "before any L1 result was inspected: rank 1, the median rank 8, and "
        "rank 15. L1 is the impulsive Sims–Flanagan approximation, not the "
        "continuous-thrust L2 validation.",
        "",
        "| L0 rank | Variant | L0 diagnostic | L1 outcome | L1 score | "
        "L0−L1 gap | Max mismatch | Max throttle | Worker-s | Actual evals |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"| {row['rank']} | `{row['variant']}` | {value(row['l0'])} | "
            f"{row['outcome']} | {value(row['l1'])} | {value(row['gap'])} | "
            f"{value(row['mismatch'], 6)} | {value(row['throttle'], 6)} | "
            f"{value(row['worker_seconds'], 1)} | {row['actual_evaluations']} |"
        )
    text.extend(
        [
            "",
            "No promotion passed the declared L1 threshold. The leader returned "
            "a finite diagnostic score, but its normalized endpoint mismatch "
            "remained far above `1e-7`. Both controls encountered typed Kepler "
            "propagation failures; their zero actual-evaluation fields mean the "
            "exception occurred before the retry layer returned its evaluation "
            "counter, not that the failed promotion consumed no compute. "
            "Worker-seconds retain the observed cost.",
            "",
        ]
    )
    return "\n".join(text)


def svg(rows: list[dict[str, object]]) -> str:
    cards = []
    for index, row in enumerate(rows):
        x = 40 + index * 320
        finite = row["l1"] is not None
        color = "#b86500" if finite else "#a63838"
        outcome = (
            f"L1 {value(row['l1'], 0)} · mismatch {value(row['mismatch'], 3)}"
            if finite
            else f"L1 {row['outcome']}"
        )
        cards.append(
            f'  <rect x="{x}" y="125" width="280" height="285" rx="16" '
            f'fill="#ffffff" stroke="{color}" stroke-width="3"/>\n'
            f'  <text x="{x + 140}" y="170" text-anchor="middle" '
            'font-family="system-ui" font-size="22" font-weight="700" '
            f'fill="#132238">L0 rank {row["rank"]}</text>\n'
            f'  <text x="{x + 140}" y="212" text-anchor="middle" '
            'font-family="system-ui" font-size="17" fill="#40556f">'
            f'L0 {value(row["l0"], 0)}</text>\n'
            f'  <path d="M{x + 140} 230 V275" stroke="#657a92" stroke-width="3" '
            'marker-end="url(#arrow)"/>\n'
            f'  <text x="{x + 140}" y="315" text-anchor="middle" '
            f'font-family="system-ui" font-size="16" font-weight="700" fill="{color}">'
            f'{html.escape(outcome)}</text>\n'
            f'  <text x="{x + 140}" y="350" text-anchor="middle" '
            'font-family="system-ui" font-size="14" fill="#40556f">'
            f'{value(row["worker_seconds"], 1)} worker-s</text>\n'
            f'  <text x="{x + 140}" y="380" text-anchor="middle" '
            'font-family="system-ui" font-size="13" fill="#40556f">'
            "threshold not passed</text>\n"
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="480" '
        'viewBox="0 0 1000 480" role="img" aria-labelledby="title desc">\n'
        '  <title id="title">Predeclared random-arm L1 promotions</title>\n'
        '  <desc id="desc">L0 leader, median, and lowest admissible routes were '
        'promoted; none passed the L1 closure threshold.</desc>\n'
        '  <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" '
        'refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" '
        'fill="#657a92"/></marker></defs>\n'
        '  <rect width="1000" height="480" fill="#f7f9fc"/>\n'
        '  <text x="500" y="48" text-anchor="middle" font-family="system-ui" '
        'font-size="25" font-weight="700" fill="#132238">'
        "Predeclared random-arm L1 follow-up</text>\n"
        '  <text x="500" y="82" text-anchor="middle" font-family="system-ui" '
        'font-size="15" fill="#40556f">leader + median admissible control + '
        "lowest admissible control</text>\n"
        + "".join(cards)
        + "</svg>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l0-archive", type=Path, default=Path("results/live-l0-seed42/random/archive.csv")
    )
    parser.add_argument(
        "--promoted-archive",
        type=Path,
        default=Path("results/live-l1-seed42/random-targeted/archive.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("results/live-l1-seed42/comparison.md"),
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("images/live-l1-seed42/targeted-promotions.svg"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = load_rows(args.l0_archive, args.promoted_archive)
    outputs = {args.markdown: markdown(rows), args.svg: svg(rows)}
    stale = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(path)
    if stale:
        raise SystemExit("missing or stale L1 summaries:\n" + "\n".join(map(str, stale)))
    if args.check:
        print("targeted L1 summary is current")


if __name__ == "__main__":
    main()
