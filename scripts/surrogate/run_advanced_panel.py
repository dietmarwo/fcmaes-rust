#!/usr/bin/env python3
"""Bounded/resumable cells: failures and censoring are retained, never retried silently."""

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--seed-start", type=int, default=101)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "confirmation" if a.confirm else "development",
        "seeds": list(range(a.seed_start, a.seed_start + 30))
        if a.confirm
        else list(range(10)),
        "problems": ["sphere", "ackley", "styblinski", "oscillator"],
        "variants": ["legacy", "advanced", "thompson", "handoff", "governor"],
        "budget": 48,
        "watchdog_seconds": 90,
        "workers": 2,
        "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
    }
    mp = a.output / "manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest
    else:
        mp.write_text(json.dumps(manifest, indent=2) + "\n")

    def run(cell):
        v, p, s = cell
        path = a.output / f"{v}-{p}-{s}.json"
        if path.exists():
            return
        begin = time.monotonic()
        try:
            result = subprocess.run(
                [str(a.binary), v, p, str(s), "48"],
                capture_output=True,
                text=True,
                timeout=90,
            )
            value = (
                json.loads(result.stdout)
                if result.returncode == 0
                else {
                    "status": "failed",
                    "stderr": result.stderr[-4000:],
                    "code": result.returncode,
                }
            )
        except subprocess.TimeoutExpired:
            value = {"status": "censored", "reason": "watchdog"}
        value.update(
            cell={"variant": v, "problem": p, "seed": s},
            wall_seconds=time.monotonic() - begin,
        )
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        temp.replace(path)

    cells = [
        (v, p, s)
        for s in manifest["seeds"]
        for p in manifest["problems"]
        for v in manifest["variants"]
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, cells))
    print(json.dumps({"cells": len(cells), "output": str(a.output)}))


if __name__ == "__main__":
    main()
