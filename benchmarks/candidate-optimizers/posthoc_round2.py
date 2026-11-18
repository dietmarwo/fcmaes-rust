#!/usr/bin/env python3
"""Post-hoc robustness checks for round 2. Not part of the declared rules.

Reads results/round2/raw.tsv and prints two checks that ROUND2.md does not
define and that therefore decide nothing:

1. A bootstrap of the mean-rank comparison behind gate G2: the repetitions
   of every solver are resampled with replacement within each problem, and
   the mean ranks are recomputed.
2. Head-to-head two-sided Mann-Whitney U tests of the candidate sequences
   against the core DE->CMA sequence, Holm-corrected over the problems.

Usage: python3 posthoc_round2.py [--raw results/round2/raw.tsv]
"""

import argparse
import collections
import csv
import math
import random
import statistics

from scipy.stats import mannwhitneyu, rankdata

INCUMBENTS = ["bite", "de", "cma", "crfmnes", "da", "de_cma"]
CANDIDATES = ["lshade", "jso", "lsrtde_b", "bipop", "jso_cma", "lsrtde_cma"]
SOLVERS = INCUMBENTS + CANDIDATES
SEQUENCES = ["jso_cma", "lsrtde_cma"]
RESAMPLES = 10000
SEED = 20261118


def mean_ranks(data, problems):
    ranks = collections.defaultdict(float)
    for problem in problems:
        medians = [statistics.median(data[(problem, s)]) for s in SOLVERS]
        for solver, rank in zip(SOLVERS, rankdata(medians)):
            ranks[solver] += float(rank) / len(problems)
    return ranks


def holm(pvalues):
    order = sorted(pvalues, key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for index, (key, p) in enumerate(order):
        running = max(running, min(1.0, (len(order) - index) * p))
        adjusted[key] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default="results/round2/raw.tsv")
    args = parser.parse_args()

    data = collections.defaultdict(list)
    with open(args.raw, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            data[(row["problem"], row["solver"])].append(float(row["best_value"]))
    problems = sorted({problem for problem, _ in data})
    for problem in problems:
        for solver in SOLVERS:
            if not data[(problem, solver)]:
                raise SystemExit(f"no rows for {solver} on {problem}")

    observed = mean_ranks(data, problems)
    best_incumbent = min(INCUMBENTS, key=lambda s: observed[s])
    print(f"problems: {len(problems)}; best incumbent by mean rank: "
          f"{best_incumbent} ({observed[best_incumbent]:.2f})")

    rng = random.Random(SEED)
    below = {c: 0 for c in CANDIDATES}
    differences = {c: [] for c in SEQUENCES}
    for _ in range(RESAMPLES):
        sample = {key: [rng.choice(values) for _ in values] for key, values in data.items()}
        ranks = mean_ranks(sample, problems)
        bar = min(ranks[s] for s in INCUMBENTS)
        for candidate in CANDIDATES:
            below[candidate] += ranks[candidate] < bar
        for candidate in SEQUENCES:
            differences[candidate].append(ranks[candidate] - ranks["de_cma"])
    print(f"\nbootstrap, {RESAMPLES} resamples, seed {SEED}")
    print("candidate    mean rank  share of resamples with G2 met")
    for candidate in CANDIDATES:
        print(f"{candidate:12s} {observed[candidate]:8.2f}  {below[candidate] / RESAMPLES:.3f}")
    for candidate in SEQUENCES:
        values = sorted(differences[candidate])
        low = values[int(0.025 * RESAMPLES)]
        high = values[int(0.975 * RESAMPLES) - 1]
        print(f"mean rank of {candidate} minus de_cma, 95% interval: [{low:.2f}, {high:.2f}]")

    print("\nhead to head against de_cma (Holm over the problems, alpha 0.05)")
    for candidate in SEQUENCES:
        tests = []
        for problem in problems:
            p = mannwhitneyu(data[(problem, candidate)], data[(problem, "de_cma")],
                             alternative="two-sided").pvalue
            tests.append((problem, 1.0 if math.isnan(p) else float(p)))
        adjusted = holm(tests)
        better, worse, lower_median = [], [], 0
        for problem in problems:
            delta = (statistics.median(data[(problem, candidate)])
                     - statistics.median(data[(problem, "de_cma")]))
            lower_median += delta < 0
            if adjusted[problem] <= 0.05:
                (better if delta < 0 else worse).append(problem)
        print(f"{candidate}: significantly better on {better or 'none'}, "
              f"significantly worse on {worse or 'none'}, "
              f"lower median on {lower_median} of {len(problems)} problems")


if __name__ == "__main__":
    main()
