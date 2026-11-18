#!/usr/bin/env python3
"""Analyze the candidate-optimizer campaign (PROTOCOL.md sections 3.5-3.7).

Reads the raw TSV written by `candidate-harness run` and writes a Markdown
report: per-cell statistics, best incumbent, Mann-Whitney U tests with Holm
correction, ranks, gates G1/G2, the decision per candidate, violations,
incomplete cells and timing.

Every consistency check raises an exception; the script never uses `assert`,
so it behaves identically under `python -O`.

Usage:
    python3 analyze.py [--raw results/raw.tsv] [--out results/report.md]
                       [--reps 20] [--solvers a,b,...] [--problems a,b,...]
                       [--threads 2]
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import mannwhitneyu, rankdata

INCUMBENTS = ["bite", "de", "cma", "crfmnes", "da", "de_cma"]
CANDIDATES = ["lshade", "jso", "lsrtde", "lsrtde_b", "bipop", "amalgam"]
# Reported with all statistics but excluded from the inclusion decision
# (DEVIATIONS.md, entry 6).
INFORMATIONAL = ["lsrtde"]
PROBLEMS = [
    "cassini1", "gtoc1", "sagas", "messenger", "tandem", "cassini2", "rosetta",
    "messenger_full", "optical_lens", "bandpass", "phased_array", "lj13", "lj38",
]
PROTOCOLS = ["S", "R"]
ALPHA = 0.05
HEADER = [
    "solver", "problem", "domain", "dim", "protocol", "rep", "seed", "budget",
    "budget_scale", "evaluations", "best_value", "over_budget", "out_of_bounds",
    "wall_ms",
]


class AnalysisError(Exception):
    """Raised when the raw data violate a consistency check."""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default="results/raw.tsv")
    parser.add_argument("--out", default="results/report.md")
    parser.add_argument("--reps", type=int, default=20,
                        help="repetitions requested per cell (default 20)")
    parser.add_argument("--solvers", default=",".join(INCUMBENTS + CANDIDATES),
                        help="solvers expected in every cell (default: all 12)")
    parser.add_argument("--problems", default=",".join(PROBLEMS),
                        help="problems expected (default: all 13)")
    parser.add_argument("--threads", type=int, default=2,
                        help="campaign threads, for the wall-time projection")
    parser.add_argument("--project-reps", type=int, default=20,
                        help="repetitions of the projected full campaign (default 20)")
    parser.add_argument("--budgets", default=None,
                        help="budgets.tsv; adds an evaluation-time projection")
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != HEADER:
            raise AnalysisError(f"{path}: unexpected header {header}")
        rows = []
        for number, fields in enumerate(reader, start=2):
            if len(fields) != len(HEADER):
                raise AnalysisError(f"{path}:{number}: expected {len(HEADER)} fields")
            row = dict(zip(HEADER, fields))
            try:
                for key in ("dim", "rep", "seed", "budget", "evaluations",
                            "over_budget", "out_of_bounds"):
                    row[key] = int(row[key])
                for key in ("budget_scale", "best_value", "wall_ms"):
                    row[key] = float(row[key])
            except ValueError as error:
                raise AnalysisError(f"{path}:{number}: {error}") from error
            if math.isnan(row["best_value"]):
                raise AnalysisError(f"{path}:{number}: best_value is NaN")
            if row["protocol"] not in PROTOCOLS:
                raise AnalysisError(f"{path}:{number}: unknown protocol {row['protocol']}")
            if row["evaluations"] > row["budget"]:
                raise AnalysisError(f"{path}:{number}: evaluations exceed the budget")
            rows.append(row)
    return rows


def fmt(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (m - position) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.reps <= 0 or args.threads <= 0 or args.project_reps <= 0:
        raise AnalysisError("--reps, --threads and --project-reps must be positive")
    solvers = [s for s in args.solvers.split(",") if s]
    problems = [p for p in args.problems.split(",") if p]
    incumbents = [s for s in solvers if s in INCUMBENTS]
    candidates = [s for s in solvers if s in CANDIDATES]
    unknown = [s for s in solvers if s not in INCUMBENTS + CANDIDATES]
    if unknown:
        raise AnalysisError(f"unknown solvers {unknown}")
    if not incumbents:
        raise AnalysisError("no incumbent among the expected solvers")

    raw_path = Path(args.raw)
    rows = load_rows(raw_path)
    if not rows:
        raise AnalysisError(f"{raw_path}: no rows")

    scales = sorted({row["budget_scale"] for row in rows})
    if len(scales) != 1:
        raise AnalysisError(f"mixed budget scales {scales}")
    scale = scales[0]

    seen = set()
    domains: dict[str, str] = {}
    dims: dict[str, int] = {}
    seeds: dict = {}
    for row in rows:
        key = (row["solver"], row["problem"], row["protocol"], row["rep"])
        if key in seen:
            raise AnalysisError(f"duplicate row {key}")
        seen.add(key)
        seed_key = (row["problem"], row["protocol"], row["rep"])
        if seeds.setdefault(seed_key, row["seed"]) != row["seed"]:
            raise AnalysisError(f"solvers saw different seeds for {seed_key}")
        if domains.setdefault(row["problem"], row["domain"]) != row["domain"]:
            raise AnalysisError(f"inconsistent domain for {row['problem']}")
        if dims.setdefault(row["problem"], row["dim"]) != row["dim"]:
            raise AnalysisError(f"inconsistent dimension for {row['problem']}")

    extra = sorted({row["solver"] for row in rows} - set(solvers))
    extra_problems = sorted({row["problem"] for row in rows} - set(problems))

    # Completeness: a cell is complete when every expected solver has at
    # least the requested repetitions (only repetitions 0..reps-1 count).
    cells = [(p, pr) for p in problems for pr in PROTOCOLS]
    complete, incomplete = [], []
    counts_by_key: dict = defaultdict(int)
    for (s, p, pr, r) in seen:
        if r < args.reps:
            counts_by_key[(s, p, pr)] += 1
    for cell in cells:
        counts = {solver: counts_by_key[(solver, *cell)] for solver in solvers}
        if all(count >= args.reps for count in counts.values()):
            complete.append(cell)
        else:
            incomplete.append((cell, counts))

    def sample(cell, solver) -> np.ndarray:
        """Best values of repetitions 0..reps-1, in repetition order."""
        return np.array([
            row["best_value"]
            for row in sorted(rows, key=lambda r: r["rep"])
            if (row["problem"], row["protocol"]) == cell
            and row["solver"] == solver and row["rep"] < args.reps
        ])

    stats = {}
    for cell in complete:
        stats[cell] = {}
        for solver in solvers:
            data = sample(cell, solver)
            stats[cell][solver] = {
                "median": float(np.median(data)),
                "mean": float(np.mean(data)),
                "best": float(np.min(data)),
                "data": data,
            }

    # Best incumbent per cell: lowest median, ties by lower mean.
    best_incumbent = {}
    for cell in complete:
        best_incumbent[cell] = min(
            incumbents,
            key=lambda s: (stats[cell][s]["median"], stats[cell][s]["mean"],
                           incumbents.index(s)),
        )

    # Ranks by median (average ranks for ties) over all expected solvers.
    ranks = {}
    for cell in complete:
        medians = [stats[cell][s]["median"] for s in solvers]
        ranks[cell] = dict(zip(solvers, rankdata(medians, method="average")))
    mean_rank = {
        s: (float(np.mean([ranks[c][s] for c in complete])) if complete else math.nan)
        for s in solvers
    }

    # Tests per candidate.
    tests = {}
    for candidate in candidates:
        entries = []
        for cell in complete:
            incumbent = best_incumbent[cell]
            x = stats[cell][candidate]["data"]
            y = stats[cell][incumbent]["data"]
            result = mannwhitneyu(x, y, alternative="two-sided")
            u = float(result.statistic)
            diff = stats[cell][candidate]["median"] - stats[cell][incumbent]["median"]
            if diff != 0:
                direction = -1 if diff < 0 else 1
            else:
                half = len(x) * len(y) / 2.0
                direction = -1 if u < half else (1 if u > half else 0)
            entries.append({
                "cell": cell, "incumbent": incumbent, "u": u,
                "p": float(result.pvalue), "direction": direction,
            })
        adjusted = holm([e["p"] for e in entries])
        for entry, p_adj in zip(entries, adjusted):
            entry["p_holm"] = p_adj
            if p_adj <= ALPHA and entry["direction"] < 0:
                entry["outcome"] = "win"
            elif p_adj <= ALPHA and entry["direction"] > 0:
                entry["outcome"] = "loss"
            else:
                entry["outcome"] = "tie"
        tests[candidate] = entries

    # Violations over all rows (complete or not).
    violations = {}
    for solver in sorted({row["solver"] for row in rows}):
        mine = [row for row in rows if row["solver"] == solver]
        violations[solver] = {
            "rows": len(mine),
            "over_budget": sum(r["over_budget"] for r in mine),
            "out_of_bounds": sum(r["out_of_bounds"] for r in mine),
            "violating_rows": sum(1 for r in mine if r["over_budget"] or r["out_of_bounds"]),
        }

    best_incumbent_rank = min(mean_rank[s] for s in incumbents) if complete else math.nan
    best_rank_incumbents = [s for s in incumbents if mean_rank[s] == best_incumbent_rank]
    decisions = {}
    for candidate in candidates:
        has_rows = candidate in violations
        wins = [e for e in tests[candidate] if e["outcome"] == "win"]
        g1 = len(wins) >= 3 and len({e["cell"][0] for e in wins}) >= 2
        g2 = bool(complete) and mean_rank[candidate] < best_incumbent_rank
        clean = has_rows and violations[candidate]["over_budget"] == 0 \
            and violations[candidate]["out_of_bounds"] == 0
        if not has_rows:
            decision = "not run"
        elif (g1 or g2) and clean:
            decision = "Include"
        elif g1 or g2:
            decision = "no rule applies (gate met, but contract violations recorded)"
        elif wins:
            decision = "Watch"
        else:
            decision = "Reject"
        if candidate in INFORMATIONAL and decision != "not run":
            decision = f"informational only (gates alone would give: {decision})"
        decisions[candidate] = {
            "wins": len(wins),
            "losses": sum(1 for e in tests[candidate] if e["outcome"] == "loss"),
            "ties": sum(1 for e in tests[candidate] if e["outcome"] == "tie"),
            "win_problems": len({e["cell"][0] for e in wins}),
            "g1": g1, "g2": g2, "clean": clean, "decision": decision,
        }

    # ---------------------------------------------------------------- report
    out = []
    w = out.append
    w("# Candidate optimizers: campaign report\n")
    w(f"Raw data: `{raw_path.name}` ({len(rows)} rows). Generated by `analyze.py` "
      f"with Python {platform.python_version()}, NumPy {np.__version__}, "
      f"SciPy {scipy.__version__}.\n")
    if scale != 1.0:
        w(f"**Budget scale {scale:g}: this is a smoke test of the harness, not a "
          "campaign result. Its numbers must not be used for any decision.**\n")
    w("## Settings\n")
    w(f"- Requested repetitions per cell: {args.reps}")
    w(f"- Expected solvers: {', '.join(solvers)}")
    w(f"- Incumbents: {', '.join(incumbents)}; candidates: {', '.join(candidates) or 'none'}")
    w(f"- Problems expected: {len(problems)}; cells expected: {len(cells)}; "
      f"complete cells: {len(complete)}")
    w(f"- Budget scale: {scale:g}")
    w(f"- Tests: two-sided Mann-Whitney U (`scipy.stats.mannwhitneyu`), Holm "
      f"correction over each candidate's complete cells, significant when the "
      f"adjusted p-value is at most {ALPHA}")
    if extra:
        w(f"- Rows of solvers outside the expected set (ignored for statistics, "
          f"listed under violations): {', '.join(extra)}")
    if extra_problems:
        w(f"- Rows of problems outside the expected set (ignored): {', '.join(extra_problems)}")
    complete_problems = sorted({p for (p, _) in complete
                                if (p, "S") in complete and (p, "R") in complete})
    outside_space = [p for p in complete_problems if domains.get(p) != "space"]
    w(f"- Problems with both cells complete: {len(complete_problems)} "
      f"({len(outside_space)} outside space trajectories); the protocol requires "
      f"at least 11 and 3")
    w("")

    w("## Decision per candidate\n")
    w(f"Best incumbent mean rank: {fmt(best_incumbent_rank)} "
      f"({', '.join(best_rank_incumbents) if complete else 'n/a'}).\n")
    w("| Candidate | Wins | Losses | Ties | Problems won | Mean rank | G1 | G2 | No violation | Decision |")
    w("|---|---:|---:|---:|---:|---:|:-:|:-:|:-:|---|")
    for candidate in candidates:
        d = decisions[candidate]
        w(f"| {candidate} | {d['wins']} | {d['losses']} | {d['ties']} | {d['win_problems']} | "
          f"{fmt(mean_rank[candidate])} | {'yes' if d['g1'] else 'no'} | "
          f"{'yes' if d['g2'] else 'no'} | {'yes' if d['clean'] else 'no'} | {d['decision']} |")
    w("")
    w("G1: at least 3 winning cells on at least 2 problems. G2: mean rank strictly "
      "better than the best incumbent's mean rank. Include: G1 or G2 and no contract "
      "violation. Watch: at least one winning cell but neither gate. `lsrtde` is "
      "informational only (DEVIATIONS.md, entry 6). Reject: no "
      "winning cell.\n")

    w("## Mean ranks\n")
    w("| Solver | Role | Mean rank |")
    w("|---|---|---:|")
    for solver in sorted(solvers, key=lambda s: (math.isnan(mean_rank[s]), mean_rank[s])):
        role = "incumbent" if solver in INCUMBENTS else "candidate"
        w(f"| {solver} | {role} | {fmt(mean_rank[solver])} |")
    w("")

    w("## Win / loss / tie against the best incumbent\n")
    if candidates and complete:
        w("| Cell | Best incumbent | " + " | ".join(candidates) + " |")
        w("|---|---|" + "---|" * len(candidates))
        for cell in complete:
            entries = []
            for candidate in candidates:
                entry = next(e for e in tests[candidate] if e["cell"] == cell)
                entries.append(f"{entry['outcome']} (p={entry['p_holm']:.3g})")
            w(f"| {cell[0]} {cell[1]} | {best_incumbent[cell]} | " + " | ".join(entries) + " |")
    else:
        w("No complete cell or no candidate.")
    w("")
    w("p-values in this table are Holm-adjusted over the candidate's cells.\n")

    w("## Per-cell statistics\n")
    w("Median, mean and best of the guard-recorded best values per solver, and "
      "the solver's rank by median in the cell.\n")
    for cell in complete:
        w(f"### {cell[0]} ({domains.get(cell[0], '?')}, D={dims.get(cell[0], '?')}), "
          f"protocol {cell[1]}\n")
        w("| Solver | Median | Mean | Best | Rank |")
        w("|---|---:|---:|---:|---:|")
        for solver in sorted(solvers, key=lambda s: ranks[cell][s]):
            st = stats[cell][solver]
            mark = " (best incumbent)" if solver == best_incumbent[cell] else ""
            w(f"| {solver}{mark} | {fmt(st['median'])} | {fmt(st['mean'])} | "
              f"{fmt(st['best'])} | {ranks[cell][solver]:g} |")
        w("")

    w("## Contract violations\n")
    w("| Solver | Rows | Over budget | Out of bounds | Rows with a violation |")
    w("|---|---:|---:|---:|---:|")
    for solver, v in violations.items():
        w(f"| {solver} | {v['rows']} | {v['over_budget']} | {v['out_of_bounds']} | "
          f"{v['violating_rows']} |")
    w("")

    w("## Incomplete cells\n")
    if incomplete:
        w("These cells have fewer repetitions than requested for at least one "
          "expected solver. They are excluded from best-incumbent selection, tests, "
          "ranks and gates.\n")
        w("| Cell | Repetitions per solver |")
        w("|---|---|")
        for cell, counts in incomplete:
            listing = ", ".join(f"{s} {n}" for s, n in counts.items() if n < args.reps)
            w(f"| {cell[0]} {cell[1]} | short: {listing} |")
    else:
        w("None.")
    w("")

    w("## Timing\n")
    total_ms = sum(row["wall_ms"] for row in rows)
    w("| Solver | Rows | Wall time (s) |")
    w("|---|---:|---:|")
    per_solver = defaultdict(float)
    for row in rows:
        per_solver[row["solver"]] += row["wall_ms"]
    for solver, ms in sorted(per_solver.items()):
        w(f"| {solver} | {violations[solver]['rows']} | {ms / 1e3:.1f} |")
    w(f"| total | {len(rows)} | {total_ms / 1e3:.1f} |")
    w("")
    if scale != 1.0:
        reps_present = max(row["rep"] for row in rows) + 1
        factor = (1.0 / scale) * (args.project_reps / reps_present)
        projected_min = total_ms * factor / args.threads / 60e3
        present = sorted(per_solver)
        w(f"Linear projection of the measured run wall times to budget scale 1 and "
          f"{args.project_reps} repetitions on {args.threads} threads, for the "
          f"{len(present)} solvers present: about {projected_min:.0f} minutes (sum "
          f"of run wall times x {factor:g} / {args.threads}); scaled to 12 solvers "
          f"at the same mean cost per solver: about "
          f"{projected_min * 12 / len(present):.0f} minutes. Run wall times include "
          f"any slowdown from other load on the host.\n")
    if args.budgets:
        with open(args.budgets, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            budget_rows = list(reader)
        if not budget_rows:
            raise AnalysisError(f"{args.budgets}: no rows")
        seconds = sum(float(r["mean_eval_us"]) * 1e-6 * int(r["budget"])
                      for r in budget_rows if r["problem"] in problems)
        eval_min = seconds * len(PROTOCOLS) * args.project_reps * 12 / args.threads / 60
        w(f"Objective time alone (measured mean evaluation time x B, summed over "
          f"the {len(problems)} problems): {seconds:.1f} s per solver, protocol and "
          f"repetition; for 12 solvers, 2 protocols and {args.project_reps} "
          f"repetitions on {args.threads} threads about {eval_min:.0f} minutes, "
          f"before any optimizer overhead.\n")

    w("## Limits\n")
    w(f"- {args.reps} repetitions per cell; small effects can stay undetected and "
      "Holm correction is conservative.")
    w("- Two budget shapes only (one run with B, eight runs with B/8); budgets come "
      "from a timing class table, not from problem difficulty.")
    w("- One host; wall times and budget classes depend on it.")
    w("- Published defaults only; no solver was tuned, so the comparison says "
      "nothing about tuned performance.")
    w("- Eight of the thirteen problems are GTOP problems, on which the fcmaes "
      "defaults were developed.")
    w("- The decisions follow the declared rules mechanically; this report draws "
      "no other conclusion.")
    w("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out))
    print(f"wrote {out_path} ({len(complete)} complete cells, "
          f"{len(incomplete)} incomplete)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except AnalysisError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
