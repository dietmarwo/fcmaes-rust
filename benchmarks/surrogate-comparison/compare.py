#!/usr/bin/env python3
"""Tabulate final regret and proposal time per problem and method.

Reads every benchmark JSON in the given directory (fcmaes-surrogate's
`examples/benchmark.rs` output and `run_external.py` output) and writes a
Markdown report. fcmaes-surrogate's own runs are labelled `fcmaes-qQ`; its
in-run baselines (random, CMA-ES, BiteOpt) are kept from the q=1 file.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import PROBLEMS, check_against  # noqa: E402

PROPOSING = ("fcmaes", "botorch", "egobox", "skopt", "optuna")
ORDER = ["branin", "hartmann3", "hartmann6", "styblinski10", "rosenbrock8", "oscillator"]


def load(directory):
    runs = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        if "parameters" in data:
            # Written by examples/benchmark.rs: the Python objectives must agree.
            check_against(path)
        q = data.get("concurrent", 1)
        for run in data["runs"]:
            method = run["method"]
            if method == "gp":
                method = f"fcmaes-q{q}"
            elif method in ("random", "cmaes", "biteopt"):
                if q != 1:
                    continue
            run = dict(run, method=method)
            history = run.pop("history")
            values = [o["value"].get("value", math.inf) if isinstance(o["value"], dict) else math.inf
                      for o in history]
            # Regret against the corrected known minima, whatever the file says.
            optimum = PROBLEMS[run["problem"]][3]
            half = len(values) // 2
            run["regret"] = max(min(values) - optimum, 0.0)
            run["regret_half"] = max(min(values[:half]) - optimum, 0.0)
            runs.append(run)
    return runs


def geometric_mean(values, floor=1e-12):
    return math.exp(statistics.mean(math.log(max(v, floor)) for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, help="Markdown report")
    parser.add_argument("--csv", type=Path, help="one row per run")
    args = parser.parse_args()
    runs = load(args.directory)
    if args.csv:
        fields = ["problem", "dimension", "method", "seed", "evaluations", "best", "regret",
                  "regret_half", "proposal_seconds", "wall_seconds"]
        with args.csv.open("w") as out:
            out.write(",".join(fields) + "\n")
            for r in sorted(runs, key=lambda r: (r["problem"], r["method"], r["seed"])):
                out.write(",".join(repr(r[f]) if isinstance(r[f], float) else str(r[f]) for f in fields) + "\n")
    problems = [p for p in ORDER if any(r["problem"] == p for r in runs)]
    methods = sorted({r["method"] for r in runs}, key=lambda m: (not m.startswith("fcmaes"), m))
    lines = ["# Surrogate optimizers on equal call budgets", ""]
    for problem in problems:
        rows = [r for r in runs if r["problem"] == problem]
        budget = rows[0]["evaluations"]
        lines += [
            f"## {problem} (d={rows[0]['dimension']}, {budget} calls)", "",
            "| Method | Seeds | Median regret at half budget | Median regret | Geometric mean regret | Worst | Median proposal s/run |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        table = []
        for method in methods:
            mine = [r for r in rows if r["method"] == method]
            if not mine:
                continue
            regrets = [r["regret"] for r in mine]
            table.append((statistics.median(regrets), method, len(mine),
                          statistics.median(r["regret_half"] for r in mine), geometric_mean(regrets),
                          max(regrets), statistics.median(r["proposal_seconds"] for r in mine)))
        for median, method, seeds, half, geo, worst, proposal in sorted(table):
            shown = f"{proposal:.2f}" if method.startswith(PROPOSING) else "–"
            lines.append(f"| {method} | {seeds} | {half:.3g} | {median:.3g} | {geo:.3g} | {worst:.3g} | {shown} |")
        lines.append("")
    text = "\n".join(lines)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
