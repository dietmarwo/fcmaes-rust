#!/usr/bin/env python3
"""Apply the predeclared handoff selection rule to development panels.

For each fraction f and each (problem, budget), take the larger of the
paired geometric loss ratios of handoff-f against `advanced` and against
`biteopt`; choose the f whose largest value over all (problem, budget) is
smallest, the smaller f on a tie (see handoff/protocol.json).

    select_handoff_fraction.py PANEL [PANEL ...]

Each PANEL is a cell directory or packed JSONL.GZ of one budget.
"""

import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "summarize", Path(__file__).with_name("summarize_advanced_panel.py")
)
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)
REFERENCES = ["advanced", "biteopt"]


def select(panels):
    ratios = {}
    for panel in panels:
        records = summarize.load(panel)
        budget = records[0]["budget"]
        for reference in REFERENCES:
            for row in summarize.summary(records, 1, 0, reference)["rows"]:
                key = (budget, row["problem"], row["variant"])
                ratios.setdefault(key, {})[reference] = row["geometric_ratio"]
    fractions = sorted(
        {v for _, _, v in ratios if v.startswith("handoff-")}, key=lambda v: int(v[8:])
    )
    cells = sorted({(b, p) for b, p, _ in ratios})
    candidates = []
    for f in fractions:
        rows = [
            {
                "budget": b,
                "problem": p,
                **{f"against_{r}": ratios[b, p, f][r] for r in REFERENCES},
            }
            for b, p in cells
        ]
        worst = max(max(row[f"against_{r}"] for r in REFERENCES) for row in rows)
        candidates.append({"variant": f, "worst": worst, "rows": rows})
    chosen = min(candidates, key=lambda c: (c["worst"], fractions.index(c["variant"])))
    return {"chosen": chosen["variant"], "candidates": candidates}


if __name__ == "__main__":
    print(json.dumps(select(sys.argv[1:]), indent=2))
