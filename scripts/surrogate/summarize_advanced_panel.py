#!/usr/bin/env python3
"""Pack and summarize an advanced-surrogate panel from its raw cells.

Reads the cell files written by run_advanced_panel.py (or a packed
JSONL.GZ), and reports per problem and variant the median final loss, the
median seconds and the paired geometric-mean loss ratio against `legacy`,
with a percentile bootstrap interval over seeds. Each ratio is
(loss + 1e-12) / (legacy loss + 1e-12), as in the delivered summary, so
that exact optima do not divide by zero. The bootstrap uses a fixed seed,
so the same records give the same summary.

    summarize_advanced_panel.py pack CELL_DIR OUTPUT.jsonl.gz
    summarize_advanced_panel.py summary RECORDS [--reference ARM] [--bootstrap N] [--seed S]

The reference arm defaults to legacy; panels above legacy's 256 calls use
another one, such as advanced or biteopt.
"""

import argparse
import gzip
import json
import math
import random
import statistics
from pathlib import Path

OFFSET = 1e-12
PROBLEMS = ["sphere", "ackley", "styblinski", "oscillator"]
VARIANTS = ["legacy", "advanced", "thompson", "handoff", "governor"]


def ordered(names, known):
    """Known names in their order, then the others alphabetically."""
    return [n for n in known if n in names] + sorted(set(names) - set(known))


def load(path):
    """Records of a cell directory (in file-name order) or a JSONL.GZ."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.glob("*.json") if p.name != "manifest.json")
        return [json.loads(p.read_text()) for p in files]
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def pack(cells, output):
    """Write the cells as JSONL.GZ in file-name order, with no timestamp."""
    records = load(cells)
    lines = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    with open(output, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
            f.write(lines.encode())
    return len(records)


def geometric_ratio(pairs):
    return math.exp(
        statistics.fmean(math.log(value / legacy) for value, legacy in pairs)
    )


def summary(records, bootstrap, seed, reference="legacy"):
    failed = [r for r in records if "status" in r]
    usable = [r for r in records if "status" not in r]
    best = {(r["problem"], r["variant"], r["seed"]): r["best"] for r in usable}
    seconds = {(r["problem"], r["variant"], r["seed"]): r["seconds"] for r in usable}
    rng = random.Random(seed)
    rows = []
    for problem in ordered({p for p, _, _ in best}, PROBLEMS):
        for variant in ordered({v for _, v, _ in best}, VARIANTS):
            seeds = sorted(s for p, v, s in best if p == problem and v == variant)
            if not seeds:
                continue
            pairs = [
                (
                    best[problem, variant, s] + OFFSET,
                    best[problem, reference, s] + OFFSET,
                )
                for s in seeds
                if (problem, reference, s) in best
            ]
            draws = sorted(
                geometric_ratio([pairs[rng.randrange(len(pairs))] for _ in pairs])
                for _ in range(bootstrap)
            )
            rows.append(
                {
                    "problem": problem,
                    "variant": variant,
                    "median_loss": statistics.median(
                        best[problem, variant, s] for s in seeds
                    ),
                    "median_seconds": statistics.median(
                        seconds[problem, variant, s] for s in seeds
                    ),
                    "pairs": len(pairs),
                    "geometric_ratio": geometric_ratio(pairs),
                    "ratio_95_interval": [
                        draws[int(0.025 * (bootstrap - 1))],
                        draws[int(0.975 * (bootstrap - 1))],
                    ],
                }
            )
    result = {
        "records": len(records),
        "failed": len(failed),
        "loss_offset": OFFSET,
        "bootstrap": {
            "resamples": bootstrap,
            "seed": seed,
            "interval": "percentile, paired over seeds",
        },
        "rows": rows,
    }
    if reference != "legacy":
        result["reference"] = reference
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("pack")
    p.add_argument("cells", type=Path)
    p.add_argument("output", type=Path)
    s = commands.add_parser("summary")
    s.add_argument("records", type=Path)
    s.add_argument("--reference", default="legacy")
    s.add_argument("--bootstrap", type=int, default=10000)
    s.add_argument("--seed", type=int, default=20261116)
    a = parser.parse_args()
    if a.command == "pack":
        print(json.dumps({"records": pack(a.cells, a.output), "output": str(a.output)}))
    else:
        print(
            json.dumps(
                summary(load(a.records), a.bootstrap, a.seed, a.reference), indent=2
            )
        )


if __name__ == "__main__":
    main()
