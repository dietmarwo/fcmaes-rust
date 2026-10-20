#!/usr/bin/env python3
"""Summarize surrogate benchmark and overhead runs into REPORT.md.

Input directory (all files optional except one new-* benchmark):
  {old,new}-SUITE-{T}.json                              examples/benchmark.rs
  overhead-{old,new}-{T}[.rK].jsonl                      examples/overhead.rs
`old` is the delivered M0 build, `new` the current build, T the thread count.
Repeated overhead files (.r1, .r2, ...; run old and new alternately) are
reduced to the per-configuration minimum time.

Checks: within a version, histories are identical for every thread count;
baseline methods (random, CMA-ES, BiteOpt) are identical between versions.
Writes REPORT.md and summary.json (no histories) to --output.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["gp", "random", "cmaes", "biteopt"]


def key(run):
    return run["problem"], run["method"], run["seed"]


def digest(history):
    text = json.dumps(history, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def load_benchmarks(directory):
    found = {}
    for path in sorted(directory.glob("*.json")):
        match = re.fullmatch(r"(old|new)-(\w+)-(\d+)\.json", path.name)
        if match:
            version, suite, threads = match.group(1), match.group(2), int(match.group(3))
            found[(version, suite, threads)] = json.loads(path.read_text())
    return found


def load_overhead(directory):
    found = {}
    for path in sorted(directory.glob("overhead-*.jsonl")):
        match = re.fullmatch(r"overhead-(old|new)-(\d+)(\.r\d+)?\.jsonl", path.name)
        if not match:
            continue
        table = found.setdefault((match.group(1), int(match.group(2))), {})
        for line in path.read_text().splitlines():
            row = json.loads(line)
            config = (row["d"], row["n"], row["count"])
            if config in table:
                assert table[config]["nlp"] == row["nlp"], config
                for field in ["fit_ms", "carried_ms"]:
                    table[config][field] = min(table[config][field], row[field])
            else:
                table[config] = row
    return found


def median(values):
    return statistics.median(values) if values else float("nan")


def check_determinism(benchmarks, lines):
    lines += ["## Thread-count determinism", ""]
    for (version, suite) in sorted({(v, s) for v, s, _ in benchmarks}):
        counts = sorted(t for v, s, t in benchmarks if (v, s) == (version, suite))
        if len(counts) < 2:
            continue
        reference = {key(r): r for r in benchmarks[(version, suite, counts[0])]["runs"]}
        for threads in counts[1:]:
            for run in benchmarks[(version, suite, threads)]["runs"]:
                other = reference[key(run)]
                assert run["history"] == other["history"], (version, suite, threads, key(run))
        lines.append(
            f"- `{version}` {suite}: all {len(reference)} runs have identical observation "
            f"histories on {', '.join(map(str, counts))} installed threads."
        )
    lines.append("")


def check_baselines(benchmarks, lines):
    pairs = 0
    for (version, suite, threads), data in benchmarks.items():
        if version != "new" or ("old", suite, threads) not in benchmarks:
            continue
        old = {key(r): r for r in benchmarks[("old", suite, threads)]["runs"]}
        for run in data["runs"]:
            if run["method"] != "gp" and key(run) in old:
                assert run["history"] == old[key(run)]["history"], key(run)
                pairs += 1
    if pairs:
        lines.append(
            f"Baseline methods are unchanged code: all {pairs} paired random/CMA-ES/BiteOpt "
            "histories are identical between the two builds.\n"
        )


def overhead_section(overhead, lines):
    if ("new", 1) not in overhead:
        return
    lines += [
        "## Proposal overhead",
        "",
        "`examples/overhead.rs`: one `propose` call on a Levy-like objective over an LHS",
        "history of n points in d dimensions, default configuration, median of three seeds.",
        "*Fit* is a round with a fresh MAP fit (two restarts) plus the acquisition for",
        "q points; *carried* reuses the hyperparameters (no fit). NLP is the fitted negative",
        "log posterior (lower is better). Times are the minimum over alternating M0/now",
        "repetitions where several were recorded.",
        "",
    ]
    for threads in sorted({t for _, t in overhead}):
        new = overhead.get(("new", threads))
        old = overhead.get(("old", threads))
        if not new:
            continue
        lines += [
            f"### {threads} installed thread{'s' if threads > 1 else ''}",
            "",
            "| d | n | q | fit ms M0 | fit ms now | speedup | carried ms M0 | carried ms now | speedup | NLP M0 | NLP now |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for (d, n, q), row in sorted(new.items()):
            if n not in (32, 128, 248):
                continue
            base = old.get((d, n, q)) if old else None
            if base:
                lines.append(
                    f"| {d} | {n} | {q} | {base['fit_ms']:.1f} | {row['fit_ms']:.1f} | "
                    f"{base['fit_ms'] / row['fit_ms']:.1f}x | {base['carried_ms']:.1f} | "
                    f"{row['carried_ms']:.1f} | {base['carried_ms'] / row['carried_ms']:.1f}x | "
                    f"{base['nlp']:.3f} | {row['nlp']:.3f} |"
                )
            else:
                lines.append(
                    f"| {d} | {n} | {q} | | {row['fit_ms']:.1f} | | | {row['carried_ms']:.1f} | | | {row['nlp']:.3f} |"
                )
        if old:
            better = sum(new[k]["nlp"] <= old[k]["nlp"] + 1e-9 for k in new if k in old)
            lines += [
                "",
                f"The current fit reaches an equal or lower NLP in {better} of {len(new)} "
                "configurations (all sizes, including those not shown).",
            ]
        lines.append("")


def quality_section(benchmarks, lines, summary):
    suites = sorted({s for _, s, _ in benchmarks}, key=lambda s: (s != "small", s != "medium", s))
    for suite in suites:
        candidates = sorted(t for v, s, t in benchmarks if v == "new" and s == suite)
        if not candidates:
            continue
        threads = 2 if 2 in candidates else candidates[-1]
        new = benchmarks[("new", suite, threads)]
        old_data = benchmarks.get(("old", suite, threads))
        old = {key(r): r for r in old_data["runs"]} if old_data else {}
        seeds = new["seeds"]
        problems = list(dict.fromkeys(r["problem"] for r in new["runs"]))
        budgets = {r["problem"]: r["evaluations"] for r in new["runs"]}
        lines += [
            f"## Quality, {', '.join(problems)}",
            "",
            f"{len(seeds)} seeds, batch size {new['concurrent']}, {threads} installed threads; "
            + ", ".join(f"{p} {budgets[p]} calls" for p in problems)
            + ". All methods share the eight-point centre/LHS start. Simple regret = best "
            "found minus known minimum (lower is better). Proposal seconds are the GP's total "
            "time in `propose` per run.",
            "",
            "| Problem | Method | Median regret | Mean regret | Median regret M0 | Mean regret M0 | Seeds where now < M0 | Proposal s/run | Proposal s/run M0 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for problem in problems:
            for method in METHODS:
                rows = [r for r in new["runs"] if r["problem"] == problem and r["method"] == method]
                regrets = [r["regret"] for r in rows]
                old_rows = [old[key(r)] for r in rows if key(r) in old]
                old_regrets = [r["regret"] for r in old_rows]
                if method == "gp":
                    proposal = f"{median([r['proposal_seconds'] for r in rows]):.3f}"
                    old_proposal = (
                        f"{median([r['proposal_seconds'] for r in old_rows]):.3f}" if old_rows else ""
                    )
                else:
                    proposal = old_proposal = ""
                if old_regrets and method == "gp":
                    wins = sum(r["regret"] < old[key(r)]["regret"] for r in rows if key(r) in old)
                    old_cells = (
                        f"{median(old_regrets):.4g} | {statistics.mean(old_regrets):.4g} | "
                        f"{wins} of {len(old_rows)}"
                    )
                else:
                    old_cells = " |  | "
                lines.append(
                    f"| {problem} | {method} | {median(regrets):.4g} | {statistics.mean(regrets):.4g} | "
                    f"{old_cells} | {proposal} | {old_proposal} |"
                )
        lines.append("")
        for version, data in [("new", new), ("old", old_data)]:
            if not data:
                continue
            for run in data["runs"]:
                summary["runs"].append(
                    {
                        "build": "current" if version == "new" else "m0",
                        "suite": suite,
                        "threads": threads,
                        **{k: run[k] for k in ["problem", "method", "seed", "evaluations", "best",
                                               "regret", "initial_best", "proposal_seconds",
                                               "fit_evaluations"]},
                        "history_sha256": digest(run["history"]),
                    }
                )


def environment():
    def run(command):
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "cpus_visible": os.cpu_count(),
        "cpus_usable": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "machine": platform.machine(),
        "system": platform.system(),
        "rustc": run(["rustc", "--version"]),
        "commit": run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="directory with benchmark/overhead outputs")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/surrogate/validation")
    args = parser.parse_args()
    benchmarks = load_benchmarks(args.input)
    overhead = load_overhead(args.input)
    if not any(v == "new" for v, _, _ in benchmarks):
        raise SystemExit("no new-*.json benchmark output found")
    env = environment()
    summary = {"schema_version": "fcmaes.surrogate.validation-summary/1", "environment": env, "runs": []}
    lines = [
        "# Surrogate validation: delivered M0 versus reviewed build",
        "",
        "Generated by `scripts/surrogate/analyze.py` from `examples/benchmark.rs` and",
        "`examples/overhead.rs` outputs. *M0* is the delivered code (`4dd693b`); *now* is",
        "the reviewed build (bounded L-BFGS MAP fit with analytic gradients, packed",
        "row-major Cholesky). Acquisition refinement and batch rule are unchanged.",
        f"Host: {env['machine']} {env['system']}, {env['cpus_usable']} usable CPUs, {env['rustc']}.",
        "Timings come from a shared host and are observations, not portable guarantees.",
        "",
    ]
    overhead_section(overhead, lines)
    quality_section(benchmarks, lines, summary)
    check_baselines(benchmarks, lines)
    check_determinism(benchmarks, lines)
    lines += [
        "## Scope",
        "",
        "Small deterministic test functions with known minima (the oscillator is the",
        "tutorial's RK4 calibration), not the expensive Rust simulations the profile targets.",
        "A different fit changes later proposals, so per-seed results diverge; the per-seed",
        "counts are close to even and no significance is claimed. Read the tables as: no",
        "systematic quality change from the faster fit, plus how proposal time scales.",
        "",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "REPORT.md").write_text("\n".join(lines))
    runs = ",\n".join(json.dumps(r, separators=(",", ":")) for r in summary["runs"])
    head = json.dumps({k: v for k, v in summary.items() if k != "runs"})[:-1]
    (args.output / "summary.json").write_text(f'{head}, "runs": [\n{runs}\n]}}\n')
    for (version, threads), rows in overhead.items():
        target = args.output / f"overhead-{version}-{threads}.jsonl"
        target.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    print("Wrote", args.output / "REPORT.md")


if __name__ == "__main__":
    main()
