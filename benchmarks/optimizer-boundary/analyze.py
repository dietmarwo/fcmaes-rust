#!/usr/bin/env python3
"""Render the checked-in optimizer-boundary decision report."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def median(values: list[float]) -> float:
    return statistics.median(values)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    absolute = abs(value)
    if absolute >= 1.0e5 or (absolute != 0.0 and absolute < 1.0e-3):
        return f"{value:.3e}"
    return f"{value:.6g}"


def sign_p(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(trials, k) for k in range(tail + 1)) / 2**trials)


def paired_stats(
    rows: list[dict[str, str]], left: str, right: str
) -> tuple[int, int, int, float, float, float, float]:
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][row["arm"]] = float(row["validation_score"])
    ratios: list[float] = []
    wins = losses = ties = 0
    for values in by_seed.values():
        if left not in values or right not in values:
            continue
        a, b = values[left], values[right]
        tolerance = 1.0e-12 * max(1.0, abs(a), abs(b))
        if b < a - tolerance:
            wins += 1
        elif b > a + tolerance:
            losses += 1
        else:
            ties += 1
        if a > 0.0 and b > 0.0 and math.isfinite(a) and math.isfinite(b):
            ratios.append(math.log(b / a))
    estimate = median(ratios) if ratios else math.nan
    rng = random.Random(0xFC0A_014)
    samples = []
    if ratios:
        for _ in range(10_000):
            samples.append(median([rng.choice(ratios) for _ in ratios]))
        samples.sort()
        lo = samples[int(0.025 * len(samples))]
        hi = samples[int(0.975 * len(samples))]
    else:
        lo = hi = math.nan
    return wins, losses, ties, sign_p(wins, losses), estimate, lo, hi


def render_refiner(rows: list[dict[str, str]]) -> str:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["protocol"], row["problem"])].append(row)

    lines = [
        "## Corrected Nelder–Mead experiment",
        "",
        "Scores below are medians over independent optimizer seeds and use held-out "
        "validation for both ReBop variants. Lower is better. `de-head` is the exact "
        "DE prefix supplied to the hybrid; it separates improvement by the tail from "
        "differences in DE random streams.",
        "",
        "| Protocol | Problem | DE | DE head | DE→NM | NM serial | NM multistart |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    order = ["de", "de-head", "de+nm", "nm-serial", "nm-multistart"]
    for key in sorted(groups):
        protocol, problem = key
        arm_values: dict[str, list[float]] = defaultdict(list)
        for row in groups[key]:
            arm_values[row["arm"]].append(float(row["validation_score"]))
        values = [fmt(median(arm_values[arm])) if arm_values[arm] else "—" for arm in order]
        lines.append(f"| `{protocol}` | {problem} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "### Paired DE→NM evidence",
            "",
            "`W/L/T` compares held-out DE→NM against full-budget DE for the same root "
            "seed. The effect is the median log score ratio; negative favors DE→NM. "
            "The interval is a deterministic paired bootstrap over seeds.",
            "",
            "| Protocol | Problem | W/L/T | sign p | median log ratio | 95% bootstrap CI | tail vs DE head W/L/T |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in sorted(groups):
        protocol, problem = key
        wins, losses, ties, p_value, effect, lo, hi = paired_stats(
            groups[key], "de", "de+nm"
        )
        hw, hl, ht, _, _, _, _ = paired_stats(groups[key], "de-head", "de+nm")
        lines.append(
            f"| `{protocol}` | {problem} | {wins}/{losses}/{ties} | {p_value:.4f} | "
            f"{fmt(effect)} | [{fmt(lo)}, {fmt(hi)}] | {hw}/{hl}/{ht} |"
        )

    lines.extend(
        [
            "",
            "Resource accounting is exact. With one worker, a 16-member DE generation "
            "costs 16 rounds. With 16 workers it costs one round. A serial simplex "
            "costs one round per objective call; multistart runs one simplex per worker. "
            "Every NM tail receives at least `2 × (dimension + 1)` calls, so every "
            "reported simplex is initialized and has a genuine descent budget.",
        ]
    )
    return "\n".join(lines)


def trace_at_deadline(
    trace: list[dict[str, str]], latency: float, nominal_calls: int
) -> tuple[float, int]:
    deadline = latency * nominal_calls
    feasible = [
        row
        for row in trace
        if float(row["overhead_seconds"]) + int(row["call"]) * latency <= deadline
    ]
    if not feasible:
        return math.inf, 0
    row = feasible[-1]
    return float(row["best_score"]), int(row["call"])


def render_bo(rows: list[dict[str, str]]) -> str:
    traces: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        traces[(row["problem"], int(row["seed"]), row["arm"])].append(row)
    for trace in traces.values():
        trace.sort(key=lambda row: int(row["call"]))

    problems = sorted({key[0] for key in traces})
    seeds = sorted({key[1] for key in traces})
    latencies = [0.001, 0.01, 0.1, 0.5, 1.0]
    budgets = [25, 60, 150]
    lines = [
        "## Equal-wall-time Bayesian experiment",
        "",
        "The objective landscape is evaluated normally, while each best-so-far trace "
        "records optimizer overhead with simulator time subtracted. For an assumed "
        "per-evaluation latency `c`, call `i` completes at `overhead(i) + i·c`. DE and "
        "EGO are then compared at the same deadline `nominal calls × c`; BO cannot "
        "spend its modelling time twice.",
        "",
        "| Problem | nominal calls | latency | DE score (calls) | BO score (calls) | BO vs DE W/L/T | sign p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for problem in problems:
        for budget in budgets:
            for latency in latencies:
                de_values = []
                bo_values = []
                de_calls = []
                bo_calls = []
                wins = losses = ties = 0
                for seed in seeds:
                    de, dc = trace_at_deadline(traces[(problem, seed, "de")], latency, budget)
                    bo, bc = trace_at_deadline(traces[(problem, seed, "bo")], latency, budget)
                    de_values.append(de)
                    bo_values.append(bo)
                    de_calls.append(dc)
                    bo_calls.append(bc)
                    tolerance = 1.0e-12 * max(1.0, abs(de), abs(bo))
                    if bo < de - tolerance:
                        wins += 1
                    elif bo > de + tolerance:
                        losses += 1
                    else:
                        ties += 1
                p_value = sign_p(wins, losses)
                lines.append(
                    f"| {problem} | {budget} | {latency * 1000:.0f} ms | "
                    f"{fmt(median(de_values))} ({median(de_calls):g}) | "
                    f"{fmt(median(bo_values))} ({median(bo_calls):g}) | "
                    f"{wins}/{losses}/{ties} | {p_value:.4f} |"
                )

    final_overheads: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (problem, _, arm), trace in traces.items():
        final_overheads[(problem, arm)].append(float(trace[-1]["overhead_seconds"]))
    lines.extend(
        [
            "",
            "### Measured optimizer overhead at the end of the trace",
            "",
            "| Problem | DE | EGO |",
            "|---|---:|---:|",
        ]
    )
    for problem in problems:
        lines.append(
            f"| {problem} | {median(final_overheads[(problem, 'de')]):.4f} s | "
            f"{median(final_overheads[(problem, 'bo')]):.4f} s |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/decision-v2"))
    parser.add_argument("--output", type=Path, default=Path("comparison.md"))
    args = parser.parse_args()
    refiner = read_tsv(args.results / "refiner-raw.tsv")
    bo = read_tsv(args.results / "bo-trace.tsv")
    seeds = len({row["seed"] for row in refiner})
    report = "\n".join(
        [
            "# Optimizer-boundary decision experiment",
            "",
            f"Recorded campaign: {seeds} independent root seeds.",
            "",
            render_refiner(refiner),
            "",
            render_bo(bo),
            "",
            "Raw per-seed and per-evaluation artifacts are `refiner-raw.tsv` and "
            "`bo-trace.tsv`. They, rather than this rendered table, are authoritative.",
            "",
        ]
    )
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
