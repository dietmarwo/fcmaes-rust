#!/usr/bin/env python3
"""Collect results/runs/*.txt into results/stage1.tsv (standard library only)."""
import glob, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for path in sorted(glob.glob(os.path.join(HERE, "results", "runs", "*__*__*.txt"))):
    problem, arm, seed = os.path.basename(path)[:-4].split("__")
    text = open(path).read()
    m = re.search(r"value=(\S+) evaluations=(\d+) runs=(\d+)", text)
    w = re.search(r"wall_s=([\d.]+)", text)
    if not (m and w):
        continue
    value = float(m.group(1))  # "inf" when no result was retained
    retained = "yes" if value == value and abs(value) != float("inf") else "no"
    rows.append((problem, arm, int(seed), m.group(1), m.group(2), m.group(3), w.group(1), retained))
order = ["cassini1", "cassini2", "rosetta", "tandem", "messenger", "gtoc1", "sagas", "messenger-full"]
rows.sort(key=lambda r: (order.index(r[0]), r[1], r[2]))
out = os.path.join(HERE, "results", "stage1.tsv")
with open(out, "w") as f:
    f.write("problem\tarm\tseed\tvalue\tevaluations\tretries\twall_s\tretained\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print(len(rows), "rows ->", os.path.relpath(out))
