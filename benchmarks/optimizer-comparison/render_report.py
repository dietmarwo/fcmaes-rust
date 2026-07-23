#!/usr/bin/env python3
"""Render the optimizer comparison from dependency-free TSV measurements."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
EXPECTED_RUNS = 100
COMMON_BUDGET = 240_000
COORDINATED_RAW = [
    ROOT.parent / "benchmark_gtop_100_raw.tsv",
    ROOT.parent / "benchmark_gtop_tandem_100_raw.tsv",
]
PROBLEM_ORDER = [
    "Cassini1",
    "Cassini2",
    "Gtoc1",
    "Messenger",
    "Rosetta",
    "Tandem",
    "Sagas",
]
OPTIMIZER_ORDER = [
    ("fcmaes", "BiteOpt", "independent-retries"),
    ("fcmaes", "DE→CMA", "coordinated-retries"),
    ("fcmaes", "BiteOpt", "ask-tell-batch"),
    ("cmaes", "CMA-ES", "parallel-population"),
    ("cmaes", "BIPOP-CMA-ES", "parallel-bipop"),
    ("genetic_algorithms", "L-SHADE", "native-serial"),
    ("math-optimisation", "DE/best/1/bin", "parallel-population"),
    ("argmin", "PSO", "parallel-population"),
]
COORDINATED_RETRY_CAPS = {
    "cassini1": 4_000,
    "cassini2": 6_000,
    "gtoc1": 10_000,
    "messenger": 8_000,
    "rosetta": 4_000,
    "tandem": 20_000,
    "sagas": 4_000,
}


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(RAW.glob("*.tsv")):
        if path.name == "all_results.tsv" or path.name.startswith("tandem_stress_"):
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            required = {
                "library",
                "version",
                "algorithm",
                "parallel_mode",
                "problem",
                "run",
                "workers",
                "population_or_batch",
                "optimizer_runs",
                "configured_evaluations",
                "actual_evaluations",
                "success",
                "value",
                "wall_seconds",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise SystemExit(f"{path} has an incompatible header")
            rows.extend(reader)
    if not rows:
        raise SystemExit(f"no TSV files found in {RAW}")
    return rows


def write_combined(rows: list[dict[str, str]]) -> None:
    path = RAW / "all_results.tsv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def pstd(values: list[float]) -> float:
    return statistics.pstdev(values)


def read_coordinated() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in COORDINATED_RAW:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                grouped[row["problem"]].append(row)
    return grouped


def coordinated_ceiling(retries: int) -> int:
    # Positive-value equivalent of Rust f64::round used by advanced_retry.
    return sum(
        int(1_500 * (1.0 + 49.0 * run / (retries - 1)) + 0.5)
        for run in range(retries)
    )


def render_tandem_stress() -> list[str]:
    metadata_path = ROOT / "tandem_stress_metadata.json"
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_path = ROOT / metadata["raw_output"]
    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected = int(metadata["runs"])
    if len(rows) != expected:
        raise SystemExit(
            f"{raw_path} has {len(rows)} stress rows, expected {expected}"
        )
    values = [float(row["value"]) for row in rows]
    evaluations = [float(row["actual_evaluations"]) for row in rows]
    seconds = [float(row["wall_seconds"]) for row in rows]
    successes = sum(row["success"] == "true" for row in rows)
    selected = metadata["selected"]
    candidates = metadata["candidate_mean_optima"]
    lines = [
        "## Tandem long-retry stress test",
        "",
        "The alternative was selected *before this stress test* by the lowest "
        "mean Tandem optimum in the 100-run, 240,000-evaluation table:",
        "",
        "| Selection candidate | Mean Tandem optimum |",
        "|---|---:|",
    ]
    for name, value in sorted(candidates.items(), key=lambda item: item[1]):
        lines.append(f"| {name} | {value:.9g} |")
    lines.extend(
        [
            "",
            f"Selected: **{selected['library']} / {selected['algorithm']}**.",
            "",
            f"The stress test ran {expected:,} independent retries with "
            f"{int(metadata['evaluations_per_retry']):,} evaluations allowed per "
            "retry. It used 24 concurrent optimizer processes and one optimizer "
            "thread per process, avoiding nested parallelism.",
            "",
            "| Retries | Configured total evals | Actual total evals | Successes "
            "| Best optimum | Mean optimum | Sdev optimum | Total wall time |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {expected:,} | {int(metadata['configured_total_evaluations']):,} "
            f"| {sum(evaluations):,.0f} | {successes} "
            f"| {min(values):.9g} | {mean(values):.9g} | {pstd(values):.6g} "
            f"| {float(metadata['invocation_wall_seconds']):.3f} s |",
            "",
            f"Mean single-retry optimizer time was {mean(seconds):.3f} s "
            f"(population sdev {pstd(seconds):.3f} s).",
            "",
        ]
    )
    if successes == 0:
        lines.extend(
            [
                "None of the 1,000 long retries reached the Tandem stop value "
                "`-1493`. This is strong empirical evidence for this implementation "
                "and configuration, not a mathematical impossibility result.",
                "",
            ]
        )
    return lines


def render(rows: list[dict[str, str]]) -> str:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["library"],
            row["algorithm"],
            row["parallel_mode"],
            row["problem"],
        )
        grouped[key].append(row)

    expected_keys = {
        (*optimizer, problem)
        for optimizer in OPTIMIZER_ORDER
        for problem in PROBLEM_ORDER
    }
    missing = expected_keys.difference(grouped)
    if missing:
        formatted = ", ".join("/".join(key) for key in sorted(missing))
        raise SystemExit(f"missing benchmark groups: {formatted}")

    for key in expected_keys:
        group = grouped[key]
        if len(group) != EXPECTED_RUNS:
            raise SystemExit(f"{'/'.join(key)} has {len(group)} rows, expected {EXPECTED_RUNS}")
        run_ids = {int(row["run"]) for row in group}
        if run_ids != set(range(EXPECTED_RUNS)):
            raise SystemExit(f"{'/'.join(key)} has an incomplete run index set")

    lines = [
        "# GTOP optimizer comparison",
        "",
        "The seven problems, absolute-best values, and stop targets are identical to "
        "the [native GTOP report](../benchmark_gtop.md). Each entry is based on 100 "
        "independent experiments, a maximum of 240,000 objective evaluations, "
        "root seed 1, and at most 24 evaluation workers.",
        "",
        "Problem definitions and putative best solutions come from ESA's "
        "[GTOP database](https://www.esa.int/gsp/ACT/projects/gtop/). See ESA's "
        "[TandEM page](https://www.esa.int/gsp/ACT/projects/gtop/tandem/) for "
        "the mission model, bounds, and problem instances used by that case.",
        "",
        "The `workers` column is the number actually available to the optimizer. "
        "`genetic_algorithms` 3.0.0 L-SHADE is correctly shown as one worker "
        "because its DE engine has no parallel population-evaluation API. "
        "BIPOP population sizes vary between restarts.",
        "",
        "Standard deviations use the population definition (`ddof=0`). Wall time "
        "covers only the optimizer call; compilation and process startup are excluded.",
        "",
        "## Main results",
        "",
        "- fcmaes produces the best mean final optimum on six of seven problems "
        "and the lowest mean optimizer wall time on five of seven.",
        "- BIPOP-CMA-ES produces the best equal-budget Tandem mean "
        "(-495.388325), but no equal-budget method reaches the `-1493` target.",
        "- The pre-registered BIPOP-CMA-ES stress test also reaches 0/1,000 "
        "targets with 10,000,000 configured evaluations per retry; its best "
        "result is -1410.050665 after 9,466,290,846 actual evaluations.",
        "- fcmaes coordinated DE→CMA retry reaches the Tandem target in 85/100 "
        "experiments. It uses a much larger adaptive budget—230,727,025 actual "
        "evaluations on average—so this is evidence for adaptive coordination, "
        "not an equal-budget comparison.",
        "",
    ]

    for problem in PROBLEM_ORDER:
        lines.extend(
            [
                f"## {problem}",
                "",
                "| Library / algorithm | Parallel mode | Version | Workers | Pop/batch "
                "| Success | Mean optimum | Sdev optimum | Mean evals | Mean wall ms "
                "| Sdev wall ms |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for library, algorithm, mode in OPTIMIZER_ORDER:
            group = grouped[(library, algorithm, mode, problem)]
            values = [float(row["value"]) for row in group]
            seconds = [float(row["wall_seconds"]) for row in group]
            evaluations = [float(row["actual_evaluations"]) for row in group]
            successes = sum(row["success"] == "true" for row in group)
            workers = {int(row["workers"]) for row in group}
            populations = {int(row["population_or_batch"]) for row in group}
            versions = {row["version"] for row in group}
            if len(workers) != 1 or len(versions) != 1:
                raise SystemExit(f"inconsistent configuration in {library}/{algorithm}/{problem}")
            population = "varies" if populations == {0} else str(next(iter(populations)))
            lines.append(
                f"| {library} / {algorithm} | {mode} | {next(iter(versions))} "
                f"| {next(iter(workers))} | {population} | {successes}% "
                f"| {mean(values):.9g} | {pstd(values):.6g} "
                f"| {mean(evaluations):.0f} | {1000.0 * mean(seconds):.3f} "
                f"| {1000.0 * pstd(seconds):.3f} |"
            )
        lines.append("")

    coordinated = read_coordinated()
    lines.extend(
        [
            "## Relation to coordinated retry",
            "",
            "The native coordinated DE→CMA retry results use adaptive budgets "
            "that are much larger than the common 240,000-evaluation allowance "
            "above. Six problems reached the stop value in all 100 experiments; "
            "Tandem reached it in 85:",
            "",
            "| Problem | Coordinated success | Mean actual evals | Multiple of 240k "
            "| Exact configured ceiling |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, retry_cap in COORDINATED_RETRY_CAPS.items():
        group = coordinated[key]
        if len(group) != EXPECTED_RUNS:
            raise SystemExit(
                f"coordinated {key} has {len(group)} rows, expected {EXPECTED_RUNS}"
            )
        actual = [float(row["evaluations"]) for row in group]
        successes = sum(row["success"] == "true" for row in group)
        display = next(
            problem for problem in PROBLEM_ORDER if problem.lower() == key
        )
        lines.append(
            f"| {display} | {successes}% | {mean(actual):,.0f} "
            f"| {mean(actual) / COMMON_BUDGET:.1f}× "
            f"| {coordinated_ceiling(retry_cap):,} |"
        )
    lines.extend(
        [
            "",
            "The coordinated ceilings are the exact sums of retry limits growing "
            "linearly from 1,500 to 75,000 evaluations over each problem's retry "
            "cap. Every recorded run consumed less than its theoretical ceiling. "
            "These results demonstrate the quality available from fcmaes "
            "coordination at a larger budget; they are not an equal-budget "
            "wall-time comparison.",
            "",
            "For context, the original Python/C++ fcmaes "
            "[Performance report](https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Performance.adoc) "
            "records 81/100 Tandem successes, 166.92 s mean wall time, and "
            "147.87 s wall-time sdev with 32 parallel Python processes. The "
            "native Rust run records 85/100, 40.207 s, and 39.105 s using 32 "
            "native retry threads. This is a historical implementation comparison, "
            "not the equal-budget crate comparison above.",
            "",
        ]
    )
    lines.extend(render_tandem_stress())

    lines.extend(
        [
            "## Interpretation constraints",
            "",
            "- BiteOpt retry uses 24 independent 10,000-evaluation searches. BiteOpt "
            "ask/tell instead uses one 240,000-evaluation state with batches of 24.",
            "- The equal-budget fcmaes coordinated row uses 48 DE→CMA retries on "
            "24 workers. Retry limits grow from 2,500 to 7,500 evaluations and "
            "sum to exactly 240,000; later retries can adapt their sigma and local "
            "box from retained solutions.",
            "- BIPOP-CMA-ES is itself an adaptive restart strategy, not plain "
            "single-population CMA-ES: it dynamically allocates restarts between "
            "small and increasingly large populations. Its conceptual counterpart "
            "is coordinated retry, while BiteOpt retry remains the fixed-budget "
            "independent-restart baseline.",
            "- BiteOpt also adapts selectors inside each optimizer state. This is "
            "different from BIPOP population-size adaptation and coordinated "
            "retry's cross-run sigma/box adaptation; the table exposes all three "
            "rather than treating every method as straight retry.",
            "- Population optimizers use their native population evaluation. Equal "
            "evaluation budgets and worker caps do not make their search topology identical.",
            "- `cmaes` is unconstrained, so its adapter searches normalized coordinates "
            "and reflects out-of-range coordinates into `[0,1]` before decoding the "
            "original GTOP bounds.",
            "- `genetic_algorithms` DE does not enforce `RangeGene` bounds during "
            "mutation. Its adapter reflects trials into the declared box before "
            "objective evaluation.",
            "- `math-optimisation` is GPL-3.0-or-later. It is built only in the external "
            "comparison workspace and is not a dependency of fcmaes-rust.",
            "",
            "The individual raw files and the combined `raw/all_results.tsv` contain "
            "every seed, final objective, actual evaluation count, success flag, and "
            "wall-time measurement.",
            "",
        ]
    )
    return "\n".join(lines)


def write_checksums() -> None:
    paths = []
    for relative in ["sources", "raw"]:
        paths.extend(path for path in (ROOT / relative).rglob("*") if path.is_file())
    for name in [
        "Cargo.lock.reference",
        "README.md",
        "comparison.md",
        "environment.txt",
        "prepare_external.sh",
        "render_report.py",
        "run_all_external.sh",
        "run_tandem_stress_external.py",
        "tandem_stress_metadata.json",
    ]:
        path = ROOT / name
        if path.exists():
            paths.append(path)
    lines = []
    for path in sorted(set(paths)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT)}")
    (ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_rows()
    write_combined(rows)
    report = render(rows)
    (ROOT / "comparison.md").write_text(report, encoding="utf-8")
    write_checksums()


if __name__ == "__main__":
    main()
