#!/usr/bin/env python3
"""Decision of round 4, ARRDE on the truss (ROUND4.md section 4).

Reads the raw files of results/round4 and writes results/round4/report.md:

    early.tsv                                  100,000 evaluations, one run, seed 9
    stage1-cap1000000.tsv, stage1-cap125000.tsv  1,000,000 evaluations, seed 7
    stage2-cap4000000.tsv, stage2-cap500000.tsv  4,000,000 evaluations, seed 8

An arm is a solver with a shape (one run: run cap = total; eight runs: run
cap = total / 8). At a stage ARRDE is superior if the median of its better
arm is lower than the median of every core arm of the stage and a two-sided
Mann-Whitney U test (scipy defaults) of that arm against the core arm with
the lowest median gives p <= 0.05 after Holm correction over the two stages.
Ties in the median are broken by the lower mean. Stage 2 uses the better
ARRDE shape and the two core arms with the lowest stage-1 medians, each in
its own shape, scaled. With stage 2 absent it is reported as pending and the
stage-1 decision is provisional (the adjusted p-value is bounded by twice the
unadjusted one).

A run ends infeasible when its best cost is at least 1e4 (DEVIATIONS.md,
round 4, entry 5); the count is informational. Contract violations are
points out of the box, and, for the core solvers, evaluations beyond the
total. ARRDE overshoots its own budget by a few evaluations, which the driver
cuts (DEVIATIONS.md, round 4, entry 3); those cut points are reported, they
are not violations.

Usage: python3 decide_round4.py [--dir results/round4] [--reps 10]

The script never uses `assert`, so it behaves identically under `python -O`.
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

from scipy.stats import mannwhitneyu

CORE = ["bite", "de", "cma", "crfmnes", "da", "de_cma", "jso_cma"]
CANDIDATE = "arrde"
SOLVERS = CORE + [CANDIDATE]
PROBLEM = "truss_sizing"
INFEASIBLE_AT = 1e4
ALPHA = 0.05
MIN_RUN = 1000
# stage name -> (total, root seed, {shape: file name})
STAGES = {
    "early": (100_000, 9, {"one": "early.tsv"}),
    "stage1": (1_000_000, 7, {"one": "stage1-cap1000000.tsv", "eight": "stage1-cap125000.tsv"}),
    "stage2": (4_000_000, 8, {"one": "stage2-cap4000000.tsv", "eight": "stage2-cap500000.tsv"}),
}


class Failure(Exception):
    pass


class Incomplete(Failure):
    """A raw file lacks repetitions (a run in progress)."""


def check(condition, message):
    if not condition:
        raise Failure(message)


def cap_of(total, shape):
    return total if shape == "one" else total // 8


def load(path, stage, shape, reps, solvers=None):
    """Rows of one raw file: {solver: [best values by repetition]} and
    per-solver counters. Every solver in `solvers` (default all eight) must
    have exactly the repetitions 0..reps-1."""
    total, root, _ = STAGES[stage]
    cap = cap_of(total, shape)
    solvers = SOLVERS if solvers is None else solvers
    values = {s: {} for s in solvers}
    info = {s: {"violations": 0, "cut": 0, "evaluations": 0, "wall_ms": 0.0, "runs": 0}
            for s in solvers}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            where = f"{path}: {row['solver']} rep {row['rep']}"
            check(row["problem"] == PROBLEM and row["protocol"] == "T", f"{where}: problem")
            check(int(row["root_seed"]) == root, f"{where}: root seed {row['root_seed']}, "
                  f"expected {root}")
            check(int(row["total"]) == total, f"{where}: total {row['total']}")
            check(int(row["run_cap"]) == cap, f"{where}: run cap {row['run_cap']}, "
                  f"expected {cap}")
            check(int(row["min_run"]) == MIN_RUN, f"{where}: min run {row['min_run']}")
            solver = row["solver"]
            check(solver in values, f"{where}: unexpected solver")
            rep = int(row["rep"])
            check(rep not in values[solver], f"{where}: duplicate row")
            check(int(row["evaluations"]) <= total, f"{where}: exceeds the total")
            value = float(row["best_value"])
            check(not math.isnan(value), f"{where}: NaN best value")
            values[solver][rep] = value
            d = info[solver]
            d["evaluations"] += int(row["evaluations"])
            d["wall_ms"] += float(row["wall_ms"])
            d["runs"] += int(row["runs"])
            d["violations"] += int(row["out_of_bounds"])
            if solver == CANDIDATE:
                d["cut"] += int(row["over_budget"])
            else:
                d["violations"] += int(row["over_budget"])
    for solver, by_rep in values.items():
        if sorted(by_rep) != list(range(reps)):
            raise Incomplete(f"{path}: {solver} has repetitions {sorted(by_rep)}")
    return {s: [v[r] for r in range(reps)] for s, v in values.items()}, info


def holm(pvalues):
    """Holm adjusted p-values of a {key: p} dict."""
    order = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for index, (key, p) in enumerate(order):
        running = max(running, min(1.0, (len(order) - index) * p))
        adjusted[key] = running
    return adjusted


def rank_key(values):
    return (statistics.median(values), statistics.mean(values))


def fmt(value):
    return f"{value:.6g}"


def pvalue(a, b):
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    return 1.0 if math.isnan(p) else float(p)


def infeasible(values):
    return sum(1 for v in values if v >= INFEASIBLE_AT)


def arms_of(data, solvers):
    """{(solver, shape): values} for the given solvers over all shapes of a stage."""
    return {(s, shape): data[shape][s] for shape in data for s in solvers if s in data[shape]}


def table(arms, info, lines):
    lines += ["| Arm | Median | Mean | Best | Worst | Infeasible runs | Contract violations "
              "| Evaluations per run | Seconds per million evaluations |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in sorted(arms, key=lambda a: rank_key(arms[a])):
        v = arms[arm]
        solver, shape = arm
        d = info[arm]
        n = len(v)
        per_run = d["evaluations"] / n
        seconds = d["wall_ms"] / 1e3 / (d["evaluations"] / 1e6) if d["evaluations"] else math.nan
        lines.append(f"| {solver}/{shape} | {fmt(statistics.median(v))} | {fmt(statistics.mean(v))} "
                     f"| {fmt(min(v))} | {fmt(max(v))} | {infeasible(v)} / {n} | "
                     f"{d['violations']} | {per_run:,.0f} | {seconds:.1f} |")
    lines.append("")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="results/round4")
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    directory = Path(args.dir)
    reps = args.reps

    lines = ["# Round 4, ARRDE on the truss", "",
             "Generated by `decide_round4.py` from the raw files of this directory under "
             "[ROUND4.md](../../ROUND4.md). Problem `truss_sizing`, protocol T, "
             f"{reps} repetitions per arm. Costs are masses in kg plus penalties; a run is "
             f"called infeasible at a best cost of at least {INFEASIBLE_AT:g}.", ""]

    # ---- early stage (informational)
    early_path = directory / STAGES["early"][2]["one"]
    early_lines = ["## Early stage (informational)", "",
                   "100,000 evaluations, one run, root seed 9. The early stage does not enter "
                   "the decision.", ""]
    if early_path.exists():
        data, info = load(early_path, "early", "one", reps)
        arms = {(s, "one"): data[s] for s in SOLVERS}
        table(arms, {(s, "one"): info[s] for s in SOLVERS}, early_lines)
        ranking = sorted(SOLVERS, key=lambda s: rank_key(data[s]))
        early_lines += ["Ranking by median: " + ", ".join(ranking) + ".", ""]
    else:
        early_lines += ["Pending (no raw file).", ""]

    # ---- stage 1
    stage1 = {}
    info1 = {}
    for shape, name in STAGES["stage1"][2].items():
        path = directory / name
        check(path.exists(), f"{path} is missing; stage 1 must be complete")
        stage1[shape], per = load(path, "stage1", shape, reps)
        for s in SOLVERS:
            info1[(s, shape)] = per[s]
    core1 = {(s, shape): stage1[shape][s] for shape in stage1 for s in CORE}
    arrde1 = {(CANDIDATE, shape): stage1[shape][CANDIDATE] for shape in stage1}
    best_arrde1 = min(arrde1, key=lambda a: rank_key(arrde1[a]))
    best_core1 = min(core1, key=lambda a: rank_key(core1[a]))
    lower1 = all(statistics.median(arrde1[best_arrde1]) < statistics.median(v)
                 for v in core1.values())
    p1 = pvalue(arrde1[best_arrde1], core1[best_core1])
    violations1 = sum(info1[a]["violations"] for a in info1)
    selected_core = sorted(core1, key=lambda a: rank_key(core1[a]))[:2]

    # ---- stage 2
    stage2, info2 = {}, {}
    pending2 = False
    s2_arms = [best_arrde1] + selected_core
    for shape, name in STAGES["stage2"][2].items():
        path = directory / name
        needed = [a[0] for a in s2_arms if a[1] == shape]
        if not needed:
            continue
        if not path.exists():
            pending2 = True
            continue
        try:
            stage2[shape], per = load(path, "stage2", shape, reps, sorted(set(needed)))
        except Incomplete as error:
            print(f"note: stage 2 incomplete ({error})", file=sys.stderr)
            pending2 = True
            continue
        for s in set(needed):
            info2[(s, shape)] = per[s]
    for arm in s2_arms:
        if arm[1] not in stage2:
            pending2 = True
    p2 = lower2 = best_core2 = None
    violations2 = 0
    if not pending2:
        values2 = {arm: stage2[arm[1]][arm[0]] for arm in s2_arms}
        arrde2 = values2[best_arrde1]
        core2 = {a: values2[a] for a in selected_core}
        best_core2 = min(core2, key=lambda a: rank_key(core2[a]))
        lower2 = all(statistics.median(arrde2) < statistics.median(v) for v in core2.values())
        p2 = pvalue(arrde2, core2[best_core2])
        violations2 = sum(info2[a]["violations"] for a in info2)

    if pending2:
        adjusted1 = min(1.0, 2 * p1)
        adjusted2 = None
    else:
        adjusted = holm({"stage1": p1, "stage2": p2})
        adjusted1, adjusted2 = adjusted["stage1"], adjusted["stage2"]
    superior1 = lower1 and adjusted1 <= ALPHA and violations1 == 0
    superior2 = bool(not pending2 and lower2 and adjusted2 <= ALPHA and violations2 == 0)
    if superior1 or superior2:
        outcome = "accepted: ARRDE is superior at " + (
            "stage 1" if superior1 else "stage 2") + (
            " (provisional until stage 2 is complete)" if pending2 else "") + "."
    elif pending2:
        outcome = ("pending: ARRDE is not superior at stage 1 (provisional adjusted p-value "
                   f"{min(1.0, 2 * p1):.3g}, upper bound); stage 2 has not been run.")
    else:
        outcome = "not accepted: ARRDE is superior at neither stage."

    def name(arm):
        return f"{arm[0]}/{arm[1]}"

    lines += ["## Decision", "",
              "| Stage | Total | ARRDE arm | Median | Best core arm | Median | "
              "Lower than every core arm | Unadjusted p | Adjusted p | Violations | Superior |",
              "|---|---:|---|---:|---|---:|:-:|---:|---:|---:|:-:|",
              f"| 1 | 1,000,000 | {name(best_arrde1)} | {fmt(statistics.median(arrde1[best_arrde1]))} "
              f"| {name(best_core1)} | {fmt(statistics.median(core1[best_core1]))} | "
              f"{'yes' if lower1 else 'no'} | {p1:.3g} | "
              f"{adjusted1:.3g}{' (bound)' if pending2 else ''} | {violations1} | "
              f"{'yes' if superior1 else 'no'} |"]
    if pending2:
        lines.append(f"| 2 | 4,000,000 | {name(best_arrde1)} | pending | "
                     f"{', '.join(name(a) for a in selected_core)} | pending | - | - | - | - | - |")
    else:
        lines.append(f"| 2 | 4,000,000 | {name(best_arrde1)} | "
                     f"{fmt(statistics.median(values2[best_arrde1]))} | {name(best_core2)} | "
                     f"{fmt(statistics.median(values2[best_core2]))} | "
                     f"{'yes' if lower2 else 'no'} | {p2:.3g} | {adjusted2:.3g} | {violations2} | "
                     f"{'yes' if superior2 else 'no'} |")
    lines += ["", f"**Outcome: {outcome}**", "",
              "Holm correction over the two stages"
              + (" (stage 2 pending: the stage-1 value is the Bonferroni bound 2 p, "
                 "an upper bound of the Holm value)." if pending2 else "."), ""]

    lines += early_lines
    lines += ["## Stage 1: 1,000,000 evaluations", "",
              "Root seed 7. One run: run cap 1,000,000. Eight runs: run cap 125,000. "
              f"Better ARRDE shape: {best_arrde1[1]}. The two core arms with the lowest "
              f"medians (stage-2 arms): {', '.join(name(a) for a in selected_core)}.", ""]
    all1 = {**core1, **arrde1}
    table(all1, info1, lines)
    cut = {a: info1[a]["cut"] for a in arrde1}
    lines += ["ARRDE points refused beyond the run allowance (library overshoot, cut by the "
              "driver): " + ", ".join(f"{name(a)} {n}" for a, n in cut.items()) + ".", ""]

    lines += ["## Stage 2: 4,000,000 evaluations", ""]
    if pending2:
        lines += ["Pending. Arms: " + ", ".join(name(a) for a in s2_arms)
                  + "; root seed 8; one run has run cap 4,000,000, eight runs 500,000.", ""]
    else:
        lines += ["Root seed 8. One run: run cap 4,000,000. Eight runs: run cap 500,000.", ""]
        arms2 = {a: values2[a] for a in s2_arms}
        table(arms2, {a: info2[a] for a in s2_arms}, lines)
        lines += ["ARRDE points refused beyond the run allowance: "
                  f"{info2[best_arrde1]['cut']}.", ""]

    lines += ["## Limits", "",
              f"- {reps} repetitions per arm, one problem, one host.",
              "- Equal evaluations, not equal time; the candidate is a library build, the "
              "core solvers are Rust.",
              "- The infeasibility count uses a cost threshold; a marginally infeasible "
              "design with a small penalty counts as feasible.", ""]
    out = directory / "report.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    print(f"stage 1: ARRDE arm {name(best_arrde1)} median "
          f"{fmt(statistics.median(arrde1[best_arrde1]))}, best core arm {name(best_core1)} "
          f"median {fmt(statistics.median(core1[best_core1]))}, p {p1:.4g}")
    print("stage2-arms: " + " ".join(name(a) for a in s2_arms))
    print("outcome: " + outcome)


if __name__ == "__main__":
    try:
        main()
    except Failure as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
