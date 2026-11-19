#!/usr/bin/env python3
"""Analyze the candidate-optimizer campaign (PROTOCOL.md sections 3.5-3.7).

Reads the raw TSV written by `candidate-harness run` and writes a Markdown
report: per-cell statistics, best incumbent, Mann-Whitney U tests with Holm
correction, ranks, gates G1/G2, the decision per candidate, violations,
incomplete cells and timing.

With `--round 2` it analyzes a protocol-T file instead (ROUND2.md section 5;
see `main_round2`). With `--round 3` it analyzes the round-3 file (ROUND3.md
section 6; see `main_round3`): cells are (problem, total), one candidate
(`catcmawm`), and the generalist gate G2 includes a resampling check.

Every consistency check raises an exception; the script never uses `assert`,
so it behaves identically under `python -O`.

Usage:
    python3 analyze.py [--raw results/raw.tsv] [--out results/report.md]
                       [--reps 20] [--solvers a,b,...] [--problems a,b,...]
                       [--threads 2]
    python3 analyze.py --round 2 --raw results/round2/raw.tsv \
                       --out results/round2/report.md --reps 20
    python3 analyze.py --round 3 --raw results/round3/raw.tsv \
                       --out results/round3/report.md --reps 20
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


# ----------------------------------------------------------------- round 2
# ROUND2.md: six incumbents, six candidate arms (none informational), the ten
# required problems plus any optional problem that passed the entry rule
# (none did; README.md, "Round 2"), protocol T only.
CANDIDATES_R2 = ["lshade", "jso", "lsrtde_b", "bipop", "jso_cma", "lsrtde_cma"]
PROBLEMS_R2 = [
    "cassini1", "gtoc1", "sagas", "messenger", "tandem", "cassini2", "rosetta",
    "messenger_full", "lj13", "lj38",
]
HEADER_T = HEADER + ["runs", "root_seed", "total", "run_cap", "min_run"]
# Declared protocol-T settings (ROUND2.md section 4) and root seed.
DECLARED_T = {"root_seed": 2, "total": 400_000, "run_cap": 100_000, "min_run": 1_000}
G1_WINS_R2 = 3


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", type=int, choices=[1, 2, 3], default=1,
                        help="1: protocols S and R (PROTOCOL.md); 2: protocol T "
                             "(ROUND2.md); 3: protocol T with two totals (ROUND3.md)")
    parser.add_argument("--raw", default=None,
                        help="raw TSV (default results/raw.tsv, round 2: "
                             "results/round2/raw.tsv)")
    parser.add_argument("--out", default=None,
                        help="report (default results/report.md, round 2: "
                             "results/round2/report.md)")
    parser.add_argument("--reps", type=int, default=20,
                        help="repetitions requested per cell (default 20)")
    parser.add_argument("--solvers", default=None,
                        help="solvers expected in every cell (default: all 12 "
                             "of the round)")
    parser.add_argument("--problems", default=None,
                        help="problems expected (default: all 13; round 2: the "
                             "round-2 set)")
    parser.add_argument("--threads", type=int, default=2,
                        help="campaign threads, for the wall-time projection")
    parser.add_argument("--project-reps", type=int, default=20,
                        help="repetitions of the projected full campaign (default 20)")
    parser.add_argument("--budgets", default=None,
                        help="budgets.tsv; adds an evaluation-time projection "
                             "(round 1 only)")
    args = parser.parse_args(argv)
    if args.round == 3:
        if args.raw is None:
            args.raw = "results/round3/raw.tsv"
        if args.out is None:
            args.out = "results/round3/report.md"
        if args.solvers is None:
            args.solvers = ",".join(INCUMBENTS_R3 + [CANDIDATE_R3])
        if args.problems is None:
            args.problems = ",".join(PROBLEMS_R3)
        return args
    round2 = args.round == 2
    if args.raw is None:
        args.raw = "results/round2/raw.tsv" if round2 else "results/raw.tsv"
    if args.out is None:
        args.out = "results/round2/report.md" if round2 else "results/report.md"
    if args.solvers is None:
        args.solvers = ",".join(INCUMBENTS + (CANDIDATES_R2 if round2 else CANDIDATES))
    if args.problems is None:
        args.problems = ",".join(PROBLEMS_R2 if round2 else PROBLEMS)
    return args


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
    if args.round == 2:
        return main_round2(args)
    if args.round == 3:
        return main_round3(args)
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


# ===================================================================== round 2

def load_rows_t(path: Path) -> list[dict]:
    """Rows of a protocol-T raw file (header HEADER_T)."""
    int_keys = ("dim", "rep", "seed", "budget", "evaluations", "over_budget",
                "out_of_bounds", "runs", "root_seed", "total", "run_cap", "min_run")
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != HEADER_T:
            raise AnalysisError(f"{path}: unexpected header {header} "
                                "(round 2 expects a protocol-T file)")
        rows = []
        for number, fields in enumerate(reader, start=2):
            if len(fields) != len(HEADER_T):
                raise AnalysisError(f"{path}:{number}: expected {len(HEADER_T)} fields")
            row = dict(zip(HEADER_T, fields))
            try:
                for key in int_keys:
                    row[key] = int(row[key])
                for key in ("budget_scale", "best_value", "wall_ms"):
                    row[key] = float(row[key])
            except ValueError as error:
                raise AnalysisError(f"{path}:{number}: {error}") from error
            if math.isnan(row["best_value"]):
                raise AnalysisError(f"{path}:{number}: best_value is NaN")
            if row["protocol"] != "T":
                raise AnalysisError(f"{path}:{number}: protocol {row['protocol']} is not T")
            if row["budget"] != row["total"]:
                raise AnalysisError(f"{path}:{number}: budget differs from total")
            if row["evaluations"] > row["total"]:
                raise AnalysisError(f"{path}:{number}: evaluations exceed the total")
            if row["runs"] < 0:
                raise AnalysisError(f"{path}:{number}: negative run count")
            rows.append(row)
    return rows


def mannwhitney_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U (SciPy default method); a NaN p-value (all
    values identical) counts as 1."""
    result = mannwhitneyu(x, y, alternative="two-sided")
    p = float(result.pvalue)
    return float(result.statistic), (1.0 if math.isnan(p) else p)


def main_round2(args: argparse.Namespace) -> int:
    """Round-2 analysis (ROUND2.md section 5): cells are problems under
    protocol T; gates G1 (at least 3 winning cells) and G2 (mean rank strictly
    better than the best incumbent's)."""
    if args.reps <= 0:
        raise AnalysisError("--reps must be positive")
    if args.budgets:
        raise AnalysisError("--budgets applies to round 1 only")
    solvers = [s for s in args.solvers.split(",") if s]
    problems = [p for p in args.problems.split(",") if p]
    if len(set(solvers)) != len(solvers) or len(set(problems)) != len(problems):
        raise AnalysisError("duplicate solver or problem in the expected sets")
    incumbents = [s for s in solvers if s in INCUMBENTS]
    candidates = [s for s in solvers if s in CANDIDATES_R2]
    unknown = [s for s in solvers if s not in INCUMBENTS + CANDIDATES_R2]
    if unknown:
        raise AnalysisError(f"unknown round-2 solvers {unknown}")
    if not incumbents:
        raise AnalysisError("no incumbent among the expected solvers")

    raw_path = Path(args.raw)
    rows = load_rows_t(raw_path)
    if not rows:
        raise AnalysisError(f"{raw_path}: no rows")

    settings_seen = sorted({(r["root_seed"], r["total"], r["run_cap"], r["min_run"])
                            for r in rows})
    if len(settings_seen) != 1:
        raise AnalysisError(f"mixed protocol-T settings {settings_seen}")
    root_seed, total, run_cap, min_run = settings_seen[0]
    settings = {"root_seed": root_seed, "total": total, "run_cap": run_cap,
                "min_run": min_run}
    declared = settings == DECLARED_T
    if any(r["budget_scale"] != 1.0 for r in rows):
        raise AnalysisError("protocol-T rows must have budget_scale 1")

    seen = set()
    domains: dict[str, str] = {}
    dims: dict[str, int] = {}
    seeds: dict = {}
    for row in rows:
        key = (row["solver"], row["problem"], row["rep"])
        if key in seen:
            raise AnalysisError(f"duplicate row {key}")
        seen.add(key)
        seed_key = (row["problem"], row["rep"])
        if seeds.setdefault(seed_key, row["seed"]) != row["seed"]:
            raise AnalysisError(f"solvers saw different seeds for {seed_key}")
        if domains.setdefault(row["problem"], row["domain"]) != row["domain"]:
            raise AnalysisError(f"inconsistent domain for {row['problem']}")
        if dims.setdefault(row["problem"], row["dim"]) != row["dim"]:
            raise AnalysisError(f"inconsistent dimension for {row['problem']}")

    extra = sorted({row["solver"] for row in rows} - set(solvers))
    extra_problems = sorted({row["problem"] for row in rows} - set(problems))

    # Completeness: every expected solver has repetitions 0..reps-1.
    counts_by_key: dict = defaultdict(int)
    for (s, p, r) in seen:
        if r < args.reps:
            counts_by_key[(s, p)] += 1
    complete, incomplete = [], []
    for problem in problems:
        counts = {solver: counts_by_key[(solver, problem)] for solver in solvers}
        if all(count >= args.reps for count in counts.values()):
            complete.append(problem)
        else:
            incomplete.append((problem, counts))

    by_cell: dict = defaultdict(list)
    for row in rows:
        if row["rep"] < args.reps:
            by_cell[(row["problem"], row["solver"])].append(row)

    stats = {}
    for problem in complete:
        stats[problem] = {}
        for solver in solvers:
            ordered = sorted(by_cell[(problem, solver)], key=lambda r: r["rep"])
            data = np.array([r["best_value"] for r in ordered])
            stats[problem][solver] = {
                "median": float(np.median(data)),
                "mean": float(np.mean(data)),
                "best": float(np.min(data)),
                "data": data,
            }

    best_incumbent = {}
    for problem in complete:
        best_incumbent[problem] = min(
            incumbents,
            key=lambda s: (stats[problem][s]["median"], stats[problem][s]["mean"],
                           incumbents.index(s)),
        )

    ranks = {}
    for problem in complete:
        medians = [stats[problem][s]["median"] for s in solvers]
        ranks[problem] = dict(zip(solvers, rankdata(medians, method="average")))
    mean_rank = {
        s: (float(np.mean([ranks[c][s] for c in complete])) if complete else math.nan)
        for s in solvers
    }

    tests = {}
    for candidate in candidates:
        entries = []
        for problem in complete:
            incumbent = best_incumbent[problem]
            x = stats[problem][candidate]["data"]
            y = stats[problem][incumbent]["data"]
            u, p = mannwhitney_p(x, y)
            diff = stats[problem][candidate]["median"] - stats[problem][incumbent]["median"]
            if diff != 0:
                direction = -1 if diff < 0 else 1
            else:
                half = len(x) * len(y) / 2.0
                direction = -1 if u < half else (1 if u > half else 0)
            entries.append({"cell": problem, "incumbent": incumbent, "u": u, "p": p,
                            "direction": direction})
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

    violations = {}
    for solver in sorted({row["solver"] for row in rows}):
        mine = [row for row in rows if row["solver"] == solver]
        violations[solver] = {
            "rows": len(mine),
            "over_budget": sum(r["over_budget"] for r in mine),
            "out_of_bounds": sum(r["out_of_bounds"] for r in mine),
            "violating_rows": sum(1 for r in mine if r["over_budget"] or r["out_of_bounds"]),
            "runs": [r["runs"] for r in mine],
            "evaluations": sum(r["evaluations"] for r in mine),
            "total": sum(r["total"] for r in mine),
            # Restarting ends normally only below min_run, so a larger rest
            # means a run made no evaluation (zero-evaluation stop).
            "empty_stops": sum(1 for r in mine if r["total"] - r["evaluations"] >= r["min_run"]),
            "wall_ms": sum(r["wall_ms"] for r in mine),
        }

    best_incumbent_rank = min(mean_rank[s] for s in incumbents) if complete else math.nan
    best_rank_incumbents = [s for s in incumbents if mean_rank[s] == best_incumbent_rank]
    decisions = {}
    for candidate in candidates:
        has_rows = candidate in violations
        wins = [e for e in tests[candidate] if e["outcome"] == "win"]
        g1 = len(wins) >= G1_WINS_R2
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
        decisions[candidate] = {
            "wins": len(wins),
            "losses": sum(1 for e in tests[candidate] if e["outcome"] == "loss"),
            "ties": sum(1 for e in tests[candidate] if e["outcome"] == "tie"),
            "g1": g1, "g2": g2, "clean": clean, "decision": decision,
        }

    # ---------------------------------------------------------------- report
    out = []
    w = out.append
    w("# Candidate optimizers, round 2: campaign report\n")
    w(f"Raw data: `{raw_path.name}` ({len(rows)} rows). Generated by `analyze.py "
      f"--round 2` with Python {platform.python_version()}, NumPy {np.__version__}, "
      f"SciPy {scipy.__version__}.\n")
    if not declared:
        w(f"**Protocol-T settings {settings} differ from the declared settings "
          f"{DECLARED_T}: this is a smoke test of the harness, not a campaign "
          "result. Its numbers must not be used for any decision.**\n")
    w("## Settings\n")
    w(f"- Protocol T: total {total} evaluations per repetition, runs of at most "
      f"{run_cap}, no run started with fewer than {min_run} remaining; root seed "
      f"{root_seed}")
    w(f"- Requested repetitions per cell: {args.reps}")
    w(f"- Expected solvers: {', '.join(solvers)}")
    w(f"- Incumbents: {', '.join(incumbents)}; candidates: {', '.join(candidates) or 'none'}")
    w(f"- Cells (problems) expected: {len(problems)}; complete: {len(complete)}")
    w(f"- Tests: two-sided Mann-Whitney U (`scipy.stats.mannwhitneyu`, default "
      f"method) against the best incumbent; a NaN p-value (all values identical) "
      f"counts as 1; Holm correction over each candidate's complete cells; "
      f"significant when the adjusted p-value is at most {ALPHA}")
    if extra:
        w(f"- Rows of solvers outside the expected set (ignored for statistics, "
          f"listed under violations): {', '.join(extra)}")
    if extra_problems:
        w(f"- Rows of problems outside the expected set (ignored): {', '.join(extra_problems)}")
    w("")

    w("## Decision per candidate\n")
    w(f"Best incumbent mean rank: {fmt(best_incumbent_rank)} "
      f"({', '.join(best_rank_incumbents) if complete else 'n/a'}).\n")
    w("| Candidate | Wins | Losses | Ties | Mean rank | G1 | G2 | No violation | Decision |")
    w("|---|---:|---:|---:|---:|:-:|:-:|:-:|---|")
    for candidate in candidates:
        d = decisions[candidate]
        w(f"| {candidate} | {d['wins']} | {d['losses']} | {d['ties']} | "
          f"{fmt(mean_rank[candidate])} | {'yes' if d['g1'] else 'no'} | "
          f"{'yes' if d['g2'] else 'no'} | {'yes' if d['clean'] else 'no'} | {d['decision']} |")
    w("")
    w(f"G1: at least {G1_WINS_R2} winning cells. G2: mean rank strictly better than "
      "the best incumbent's mean rank. Include: G1 or G2 and no contract violation "
      "(for a sequence arm: implement its first stage in core as a sequence stage). "
      "Watch: at least one winning cell but neither gate. Reject: no winning cell.\n")

    w("## Mean ranks\n")
    w(f"Rank by median among the {len(solvers)} solvers in each complete cell "
      "(average ranks for ties), averaged over the complete cells.\n")
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
        for problem in complete:
            entries = []
            for candidate in candidates:
                entry = next(e for e in tests[candidate] if e["cell"] == problem)
                entries.append(f"{entry['outcome']} (p={entry['p_holm']:.3g})")
            w(f"| {problem} | {best_incumbent[problem]} | " + " | ".join(entries) + " |")
    else:
        w("No complete cell or no candidate.")
    w("")
    w("p-values in this table are Holm-adjusted over the candidate's cells.\n")

    w("## Per-cell medians\n")
    w("Median of the guard-recorded best values over the repetitions; the best "
      "incumbent of each cell is marked with `*`.\n")
    if complete:
        w("| Cell | D | " + " | ".join(solvers) + " |")
        w("|---|---:|" + "---:|" * len(solvers))
        for problem in complete:
            cells = []
            for solver in solvers:
                mark = "*" if solver == best_incumbent[problem] else ""
                cells.append(f"{fmt(stats[problem][solver]['median'])}{mark}")
            w(f"| {problem} | {dims.get(problem, '?')} | " + " | ".join(cells) + " |")
    else:
        w("No complete cell.")
    w("")

    w("## Runs and evaluations\n")
    w("Runs per repetition (minimum / median / maximum of the `runs` column) and "
      "evaluations counted by the guards as a share of the total, over all rows of "
      "the solver. Empty stops: repetitions in which a run made no evaluation, so "
      "restarting stopped with at least the minimum run left.\n")
    w("| Solver | Rows | Runs min | Runs median | Runs max | Evaluations / total | Empty stops |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for solver, v in violations.items():
        runs = np.array(v["runs"])
        share = v["evaluations"] / v["total"] if v["total"] else math.nan
        w(f"| {solver} | {v['rows']} | {int(runs.min())} | {float(np.median(runs)):g} | "
          f"{int(runs.max())} | {share:.4f} | {v['empty_stops']} |")
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
        for problem, counts in incomplete:
            listing = ", ".join(f"{s} {n}" for s, n in counts.items() if n < args.reps)
            w(f"| {problem} | short: {listing} |")
    else:
        w("None.")
    w("")

    w("## Timing\n")
    w("| Solver | Rows | Wall time (s) |")
    w("|---|---:|---:|")
    total_ms = 0.0
    for solver, v in violations.items():
        total_ms += v["wall_ms"]
        w(f"| {solver} | {v['rows']} | {v['wall_ms'] / 1e3:.1f} |")
    w(f"| total | {len(rows)} | {total_ms / 1e3:.1f} |")
    w("")

    w("## Limits\n")
    w(f"- {args.reps} repetitions per cell; small effects can stay undetected and "
      "Holm correction is conservative.")
    w(f"- One budget shape (restarts of at most {run_cap} evaluations up to a total "
      f"of {total}); {len(problems)} problems, eight of them GTOP problems, on which "
      "the fcmaes defaults were developed.")
    w("- One host; wall times depend on it.")
    w("- Published defaults only; no solver was tuned, so the comparison says "
      "nothing about tuned performance.")
    w("- The decisions follow the declared rules mechanically; this report draws "
      "no other conclusion.")
    w("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out))
    print(f"wrote {out_path} ({len(complete)} complete cells, "
          f"{len(incomplete)} incomplete)")
    return 0


# ===================================================================== round 3
# ROUND3.md and DEVIATIONS.md (round 3): seven incumbents, one candidate, six
# problems, cells (problem, total), protocol T with run cap equal to the total.
INCUMBENTS_R3 = ["bite", "de", "cma", "crfmnes", "da", "de_cma", "jso_cma"]
CANDIDATE_R3 = "catcmawm"
PROBLEMS_R3 = ["truss_sizing", "phased_array", "field_service", "jobshop",
               "uav_assignment", "material_flow"]
# Problems with a categorical coordinate (descriptive split, ROUND3.md section 6).
CATEGORICAL_R3 = ["field_service", "jobshop"]
DECLARED_R3 = {"root_seed": 3, "totals": [2_000, 20_000], "min_run": 100}
G1_WINS_R3 = 3
G1_PROBLEMS_R3 = 2
G2_RESAMPLES = 10_000
G2_SHARE = 0.95
# Seed of the resampling generator (the round's root seed).
G2_SEED = 3


def resample_share(data: np.ndarray, candidate: int, incumbents: list[int],
                   resamples: int, seed: int) -> float:
    """Share of resamples in which the candidate's mean rank is strictly
    below every incumbent's.

    `data` has shape (cells, solvers, repetitions). Each resample draws, for
    every cell and solver independently, the repetitions with replacement,
    recomputes the medians, ranks the solvers per cell by median (average
    ranks for ties) and averages the ranks over the cells."""
    cells, solvers, reps = data.shape
    rng = np.random.default_rng(seed)
    rank_sum = np.zeros((resamples, solvers))
    for cell in range(cells):
        index = rng.integers(0, reps, size=(resamples, solvers, reps))
        drawn = np.take_along_axis(
            np.broadcast_to(data[cell], (resamples, solvers, reps)), index, axis=2)
        medians = np.median(drawn, axis=2)
        rank_sum += rankdata(medians, method="average", axis=1)
    mean_ranks = rank_sum / cells
    better = np.all(mean_ranks[:, [candidate]] < mean_ranks[:, incumbents], axis=1)
    return float(np.mean(better))


def main_round3(args: argparse.Namespace) -> int:
    """Round-3 analysis (ROUND3.md section 6): cells are (problem, total);
    best incumbent per cell by median (ties: lower mean); two-sided
    Mann-Whitney U against it, Holm over the candidate's cells; ranks of all
    solvers per cell by median; G1 (at least 3 winning cells on at least 2
    problems) and G2 (mean rank strictly better than the best incumbent's,
    and strictly better than every incumbent's in at least 95% of 10,000
    seeded resamples of the repetitions)."""
    if args.reps <= 0:
        raise AnalysisError("--reps must be positive")
    if args.budgets:
        raise AnalysisError("--budgets applies to round 1 only")
    solvers = [s for s in args.solvers.split(",") if s]
    problems = [p for p in args.problems.split(",") if p]
    if len(set(solvers)) != len(solvers) or len(set(problems)) != len(problems):
        raise AnalysisError("duplicate solver or problem in the expected sets")
    incumbents = [s for s in solvers if s in INCUMBENTS_R3]
    unknown = [s for s in solvers if s not in INCUMBENTS_R3 + [CANDIDATE_R3]]
    if unknown:
        raise AnalysisError(f"unknown round-3 solvers {unknown}")
    if not incumbents:
        raise AnalysisError("no incumbent among the expected solvers")
    if CANDIDATE_R3 not in solvers:
        raise AnalysisError(f"the candidate {CANDIDATE_R3} is not among the expected solvers")
    candidate = CANDIDATE_R3

    raw_path = Path(args.raw)
    rows = load_rows_t(raw_path)
    if not rows:
        raise AnalysisError(f"{raw_path}: no rows")
    for row in rows:
        if row["run_cap"] != row["total"]:
            raise AnalysisError(f"row {row['solver']} {row['problem']} {row['rep']}: run cap "
                                "differs from the total (round 3: run cap = total)")
        if row["budget_scale"] != 1.0:
            raise AnalysisError("protocol-T rows must have budget_scale 1")
    shared = sorted({(r["root_seed"], r["min_run"]) for r in rows})
    if len(shared) != 1:
        raise AnalysisError(f"mixed root seeds or minimum runs {shared}")
    root_seed, min_run = shared[0]
    totals = sorted({r["total"] for r in rows})
    declared = (root_seed == DECLARED_R3["root_seed"] and min_run == DECLARED_R3["min_run"]
                and totals == DECLARED_R3["totals"])

    seen = set()
    domains: dict[str, str] = {}
    dims: dict[str, int] = {}
    seeds: dict = {}
    for row in rows:
        key = (row["solver"], row["problem"], row["total"], row["rep"])
        if key in seen:
            raise AnalysisError(f"duplicate row {key}")
        seen.add(key)
        seed_key = (row["problem"], row["total"], row["rep"])
        if seeds.setdefault(seed_key, row["seed"]) != row["seed"]:
            raise AnalysisError(f"solvers saw different seeds for {seed_key}")
        if domains.setdefault(row["problem"], row["domain"]) != row["domain"]:
            raise AnalysisError(f"inconsistent domain for {row['problem']}")
        if dims.setdefault(row["problem"], row["dim"]) != row["dim"]:
            raise AnalysisError(f"inconsistent dimension for {row['problem']}")

    extra = sorted({row["solver"] for row in rows} - set(solvers))
    extra_problems = sorted({row["problem"] for row in rows} - set(problems))

    cells = [(p, t) for p in problems for t in totals]
    counts_by_key: dict = defaultdict(int)
    for (s, p, t, r) in seen:
        if r < args.reps:
            counts_by_key[(s, p, t)] += 1
    complete, incomplete = [], []
    for cell in cells:
        counts = {solver: counts_by_key[(solver, *cell)] for solver in solvers}
        if all(count >= args.reps for count in counts.values()):
            complete.append(cell)
        else:
            incomplete.append((cell, counts))

    by_cell: dict = defaultdict(list)
    for row in rows:
        if row["rep"] < args.reps:
            by_cell[(row["problem"], row["total"], row["solver"])].append(row)

    stats = {}
    for cell in complete:
        stats[cell] = {}
        for solver in solvers:
            ordered = sorted(by_cell[(*cell, solver)], key=lambda r: r["rep"])
            data = np.array([r["best_value"] for r in ordered])
            stats[cell][solver] = {
                "median": float(np.median(data)),
                "mean": float(np.mean(data)),
                "best": float(np.min(data)),
                "data": data,
            }

    best_incumbent = {}
    for cell in complete:
        best_incumbent[cell] = min(
            incumbents,
            key=lambda s: (stats[cell][s]["median"], stats[cell][s]["mean"],
                           incumbents.index(s)),
        )

    ranks = {}
    for cell in complete:
        medians = [stats[cell][s]["median"] for s in solvers]
        ranks[cell] = dict(zip(solvers, rankdata(medians, method="average")))

    def mean_rank_over(subset) -> dict:
        return {s: (float(np.mean([ranks[c][s] for c in subset])) if subset else math.nan)
                for s in solvers}

    mean_rank = mean_rank_over(complete)

    entries = []
    for cell in complete:
        incumbent = best_incumbent[cell]
        x = stats[cell][candidate]["data"]
        y = stats[cell][incumbent]["data"]
        u, p = mannwhitney_p(x, y)
        diff = stats[cell][candidate]["median"] - stats[cell][incumbent]["median"]
        if diff != 0:
            direction = -1 if diff < 0 else 1
        else:
            half = len(x) * len(y) / 2.0
            direction = -1 if u < half else (1 if u > half else 0)
        entries.append({"cell": cell, "incumbent": incumbent, "u": u, "p": p,
                        "direction": direction})
    adjusted = holm([e["p"] for e in entries])
    for entry, p_adj in zip(entries, adjusted):
        entry["p_holm"] = p_adj
        if p_adj <= ALPHA and entry["direction"] < 0:
            entry["outcome"] = "win"
        elif p_adj <= ALPHA and entry["direction"] > 0:
            entry["outcome"] = "loss"
        else:
            entry["outcome"] = "tie"

    violations = {}
    for solver in sorted({row["solver"] for row in rows}):
        mine = [row for row in rows if row["solver"] == solver]
        violations[solver] = {
            "rows": len(mine),
            "over_budget": sum(r["over_budget"] for r in mine),
            "out_of_bounds": sum(r["out_of_bounds"] for r in mine),
            "violating_rows": sum(1 for r in mine if r["over_budget"] or r["out_of_bounds"]),
            "wall_ms": sum(r["wall_ms"] for r in mine),
        }
    usage = {}
    for solver in sorted({row["solver"] for row in rows}):
        for total in totals:
            mine = [r for r in rows if r["solver"] == solver and r["total"] == total]
            if not mine:
                continue
            usage[(solver, total)] = {
                "rows": len(mine),
                "runs": np.array([r["runs"] for r in mine]),
                "evaluations": sum(r["evaluations"] for r in mine),
                "total": sum(r["total"] for r in mine),
                "empty_stops": sum(1 for r in mine
                                   if r["total"] - r["evaluations"] >= r["min_run"]),
            }

    best_incumbent_rank = min(mean_rank[s] for s in incumbents) if complete else math.nan
    best_rank_incumbents = [s for s in incumbents if mean_rank[s] == best_incumbent_rank]
    wins = [e for e in entries if e["outcome"] == "win"]
    win_problems = len({e["cell"][0] for e in wins})
    g1 = len(wins) >= G1_WINS_R3 and win_problems >= G1_PROBLEMS_R3
    g2_point = bool(complete) and mean_rank[candidate] < best_incumbent_rank
    if complete:
        data = np.array([[stats[cell][s]["data"] for s in solvers] for cell in complete])
        share = resample_share(data, solvers.index(candidate),
                               [solvers.index(s) for s in incumbents], G2_RESAMPLES, G2_SEED)
    else:
        share = math.nan
    g2 = g2_point and share >= G2_SHARE
    has_rows = candidate in violations
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
    losses = sum(1 for e in entries if e["outcome"] == "loss")
    ties = sum(1 for e in entries if e["outcome"] == "tie")

    # ---------------------------------------------------------------- report
    out = []
    w = out.append
    w("# Candidate optimizers, round 3: mixed-variable screening report\n")
    w(f"Raw data: `{raw_path.name}` ({len(rows)} rows). Generated by `analyze.py "
      f"--round 3` with Python {platform.python_version()}, NumPy {np.__version__}, "
      f"SciPy {scipy.__version__}.\n")
    if not declared:
        w(f"**Settings (root seed {root_seed}, totals {', '.join(map(str, totals))}, "
          f"minimum run {min_run}) differ from the declared settings (root seed "
          f"{DECLARED_R3['root_seed']}, totals "
          f"{', '.join(map(str, DECLARED_R3['totals']))}, minimum run "
          f"{DECLARED_R3['min_run']}): this is a smoke test of the harness, not a "
          "campaign result. Its numbers must not be used for any decision.**\n")
    w("## Settings\n")
    w(f"- Protocol T per cell: total of {', '.join(map(str, totals))} evaluations per "
      f"repetition, runs of at most the total, no run started with fewer than {min_run} "
      f"remaining; root seed {root_seed}")
    w(f"- Requested repetitions per cell: {args.reps}")
    w(f"- Incumbents: {', '.join(incumbents)}; candidate: {candidate} (`cmaes.CatCMAwM`, "
      "Python reference implementation, default parameters)")
    w(f"- Cells (problem, total) expected: {len(cells)}; complete: {len(complete)}")
    w(f"- Tests: two-sided Mann-Whitney U (`scipy.stats.mannwhitneyu`, default "
      f"method) against the best incumbent of the cell (lowest median, ties by lower "
      f"mean); a NaN p-value (all values identical) counts as 1; Holm correction over "
      f"the candidate's complete cells; significant when the adjusted p-value is at "
      f"most {ALPHA}")
    w(f"- Ranks: by median among the {len(solvers)} solvers in each complete cell "
      "(average ranks for ties), averaged over the complete cells")
    w(f"- Resampling for G2: {G2_RESAMPLES:,} resamples (NumPy `default_rng({G2_SEED})`); "
      "in each, the repetitions of every cell and solver are drawn with replacement, "
      "all mean ranks are recomputed, and the resample counts when the candidate's "
      "mean rank is strictly below every incumbent's")
    if extra:
        w(f"- Rows of solvers outside the expected set (ignored for statistics, "
          f"listed under violations): {', '.join(extra)}")
    if extra_problems:
        w(f"- Rows of problems outside the expected set (ignored): {', '.join(extra_problems)}")
    w("")

    w("## Decision\n")
    w(f"Best incumbent mean rank: {fmt(best_incumbent_rank)} "
      f"({', '.join(best_rank_incumbents) if complete else 'n/a'}).\n")
    w("| Candidate | Wins | Losses | Ties | Problems won | Mean rank | Resample share | "
      "G1 | G2 | No violation | Decision |")
    w("|---|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|---|")
    share_text = "n/a" if math.isnan(share) else f"{share:.4f}"
    w(f"| {candidate} | {len(wins)} | {losses} | {ties} | {win_problems} | "
      f"{fmt(mean_rank[candidate])} | {share_text} | {'yes' if g1 else 'no'} | "
      f"{'yes' if g2 else 'no'} | {'yes' if clean else 'no'} | {decision} |")
    w("")
    w(f"G1: at least {G1_WINS_R3} winning cells on at least {G1_PROBLEMS_R3} different "
      "problems. G2: mean rank strictly better than the best incumbent's mean rank, and "
      f"strictly better than every incumbent's in at least {G2_SHARE:.0%} of the "
      f"{G2_RESAMPLES:,} resamples (resample share). Include: G1 or G2 and no contract "
      "violation (implement CatCMAwM in core). Watch: at least one winning cell but "
      "neither gate. Reject for now: no winning cell.\n")

    w("## Mean ranks\n")
    w("| Solver | Role | Mean rank |")
    w("|---|---|---:|")
    for solver in sorted(solvers, key=lambda s: (math.isnan(mean_rank[s]), mean_rank[s])):
        role = "incumbent" if solver in INCUMBENTS_R3 else "candidate"
        w(f"| {solver} | {role} | {fmt(mean_rank[solver])} |")
    w("")

    w("## Win / loss / tie against the best incumbent\n")
    if complete:
        w(f"| Cell | Best incumbent | {candidate} | Adjusted p | Unadjusted p |")
        w("|---|---|---|---:|---:|")
        for entry in entries:
            problem, total = entry["cell"]
            w(f"| {problem} {total} | {entry['incumbent']} | {entry['outcome']} | "
              f"{entry['p_holm']:.3g} | {entry['p']:.3g} |")
    else:
        w("No complete cell.")
    w("")
    w("Adjusted p-values are Holm-adjusted over the candidate's cells.\n")

    w("## Per-cell medians\n")
    w("Median of the guard-recorded best values over the repetitions; the best "
      "incumbent of each cell is marked with `*`.\n")
    if complete:
        w("| Cell | D | " + " | ".join(solvers) + " |")
        w("|---|---:|" + "---:|" * len(solvers))
        for cell in complete:
            values = []
            for solver in solvers:
                mark = "*" if solver == best_incumbent[cell] else ""
                values.append(f"{fmt(stats[cell][solver]['median'])}{mark}")
            w(f"| {cell[0]} {cell[1]} | {dims.get(cell[0], '?')} | " + " | ".join(values) + " |")
    else:
        w("No complete cell.")
    w("")

    w("## Descriptive split: problems with and without a categorical coordinate\n")
    w("Descriptive only; this split decides nothing (ROUND3.md section 6). "
      f"Categorical: cells of {', '.join(CATEGORICAL_R3)}; other: cells of the "
      "remaining problems. Mean ranks are the full-cell ranks averaged over the "
      "group's complete cells.\n")
    groups = [
        ("categorical", [c for c in complete if c[0] in CATEGORICAL_R3]),
        ("other", [c for c in complete if c[0] not in CATEGORICAL_R3]),
    ]
    w(f"| Group | Complete cells | {candidate} wins | Losses | Ties | " +
      " | ".join(f"Mean rank {s}" for s in solvers) + " |")
    w("|---|---:|---:|---:|---:|" + "---:|" * len(solvers))
    for name, subset in groups:
        group_entries = [e for e in entries if e["cell"] in subset]
        group_rank = mean_rank_over(subset)
        w(f"| {name} | {len(subset)} | "
          f"{sum(1 for e in group_entries if e['outcome'] == 'win')} | "
          f"{sum(1 for e in group_entries if e['outcome'] == 'loss')} | "
          f"{sum(1 for e in group_entries if e['outcome'] == 'tie')} | " +
          " | ".join(fmt(group_rank[s]) for s in solvers) + " |")
    w("")

    w("## Runs and evaluations\n")
    w("Runs per repetition (minimum / median / maximum of the `runs` column) and "
      "evaluations counted by the guards as a share of the total, per solver and "
      "total, over all rows. Empty stops: repetitions in which a run made no "
      "evaluation, so restarting stopped with at least the minimum run left.\n")
    w("| Solver | Total | Rows | Runs min | Runs median | Runs max | Evaluations / total | "
      "Empty stops |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for (solver, total), v in usage.items():
        share_used = v["evaluations"] / v["total"] if v["total"] else math.nan
        w(f"| {solver} | {total} | {v['rows']} | {int(v['runs'].min())} | "
          f"{float(np.median(v['runs'])):g} | {int(v['runs'].max())} | {share_used:.4f} | "
          f"{v['empty_stops']} |")
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
          "ranks, resampling and gates.\n")
        w("| Cell | Repetitions per solver |")
        w("|---|---|")
        for (problem, total), counts in incomplete:
            listing = ", ".join(f"{s} {n}" for s, n in counts.items() if n < args.reps)
            w(f"| {problem} {total} | short: {listing} |")
    else:
        w("None.")
    w("")

    w("## Timing\n")
    w("Run wall times summed per solver. The candidate is Python and the incumbents "
      "are Rust; solvers are compared by evaluations only.\n")
    w("| Solver | Rows | Wall time (s) |")
    w("|---|---:|---:|")
    total_ms = 0.0
    for solver, v in violations.items():
        total_ms += v["wall_ms"]
        w(f"| {solver} | {v['rows']} | {v['wall_ms'] / 1e3:.1f} |")
    w(f"| total | {len(rows)} | {total_ms / 1e3:.1f} |")
    w("")

    w("## Limits\n")
    w("- Narrower question than declared (DEVIATIONS.md, round 3, entry 1): only two "
      "problems under the entry rule have an unordered categorical coordinate "
      "(field-service routing and the job shop), so the question is limited to "
      "mixed-integer problems with some categorical structure. A statement about "
      "categorical variables rests on those two problems alone and is not a verdict "
      "on CatCMAwM for categorical design problems in general.")
    w("- Six problems (DEVIATIONS.md, round 3, entry 2): the two network-coverage "
      "problems (4,000 variables) are left out by a dimension cap of 200, and the "
      "costlier problems with the richest mixed structure (energy hub, water network, "
      "ML tuning) are outside the 2 ms entry rule on this host.")
    w("- Binary switches are typed integer, as declared, although the `cmaes` library "
      "recommends categorical for them (DEVIATIONS.md, round 3, entry 3).")
    w("- The candidate is the authors' Python reference implementation; it is "
      "compared by objective evaluations only, never by time. Its seed is the low 32 "
      "bits of the shared run seed (entry 4).")
    w(f"- {args.reps} repetitions per cell; small effects can stay undetected and "
      "Holm correction is conservative.")
    w("- Published defaults only; no solver was tuned.")
    w("- The decision follows the declared rules mechanically; this report draws no "
      "other conclusion.")
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
