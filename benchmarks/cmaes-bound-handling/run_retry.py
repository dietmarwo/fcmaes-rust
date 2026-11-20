#!/usr/bin/env python3
"""Comparisons C and D: coordinated DE->CMA retry of gtop-advexamples.

Runs `gtop-advexamples --problem P --retries 300 --evaluations 1500
--workers 1 --seed S --max-eval-fac 50 --check-interval 100` with no flag
(clip), --repair-z (repair) or --bound-transform (transform) and appends one
row per run to results/<problem>.tsv: mode, seed, value, evaluations,
retries (the binary's `runs`). Finished rows are skipped, so the script
resumes after an interruption.

Usage: python3 run_retry.py --binary PATH --problem messenger|cassini2
           [--modes clip,repair,transform] [--seeds 1-12]
"""

import argparse
import re
import subprocess
from pathlib import Path

FLAGS = {"clip": [], "repair": ["--repair-z"], "transform": ["--bound-transform"]}
HEADER = "mode\tseed\tvalue\tevaluations\tretries"
LINE = re.compile(r": value=(\S+) evaluations=(\d+) runs=(\d+) ")


def seeds(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--problem", required=True, choices=["messenger", "cassini2"])
    ap.add_argument("--modes", default="clip,repair,transform")
    ap.add_argument("--seeds", default="1-12")
    a = ap.parse_args()
    out = Path(__file__).resolve().parent / "results" / f"{a.problem}.tsv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines()[1:]:
            f = line.split("\t")
            if len(f) == 5:
                done.add((f[0], int(f[1])))
    else:
        out.write_text(HEADER + "\n")
    for seed in seeds(a.seeds):
        for mode in a.modes.split(","):
            if (mode, seed) in done:
                continue
            cmd = [a.binary, "--problem", a.problem, "--retries", "300", "--evaluations", "1500",
                   "--workers", "1", "--seed", str(seed), "--max-eval-fac", "50",
                   "--check-interval", "100"] + FLAGS[mode]
            text = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
            m = LINE.search(text)
            if not m:
                raise SystemExit(f"no summary line in output of {cmd}:\n{text}")
            row = f"{mode}\t{seed}\t{m.group(1)}\t{m.group(2)}\t{m.group(3)}"
            with out.open("a") as f:
                f.write(row + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
