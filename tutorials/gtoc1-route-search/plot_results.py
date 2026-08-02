#!/usr/bin/env python3
"""Render deterministic SVG summaries from route-search artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


ARMS = ("agent", "random", "evolutionary")
COLORS = {"agent": "#2774ae", "random": "#c99118", "evolutionary": "#7656b5"}
MGA_ARMS = ("random", "evolutionary", "gemma4", "gemma4-assisted")
MGA_LABELS = {
    "random": "random",
    "evolutionary": "evolutionary",
    "gemma4": "cold Gemma",
    "gemma4-assisted": "Gemma assisted",
}
MGA_COLORS = {
    "random": "#c99118",
    "evolutionary": "#7656b5",
    "gemma4": "#2774ae",
    "gemma4-assisted": "#16856b",
}
BODY = {"1": "Me", "2": "V", "3": "E", "4": "Ma", "5": "J", "6": "S", "10": "A"}
L0_CONSTRAINT_THRESHOLD = 1.0e-8


def arm_directory(root: Path, arm: str) -> Path:
    """Prefer the completed repaired evolutionary arm when it is present."""

    repaired = root / "evolutionary-repaired"
    if arm == "evolutionary" and repaired.is_dir():
        return repaired
    return root / arm


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_runs(root: Path) -> dict[str, dict[str, object]]:
    runs = {}
    for arm in ARMS:
        with (arm_directory(root, arm) / "run.json").open(encoding="utf-8") as stream:
            runs[arm] = json.load(stream)
    return runs


def svg_document(body: str, title: str, description: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="520" '
        'viewBox="0 0 1000 520" role="img" aria-labelledby="title desc">\n'
        f'  <title id="title">{html.escape(title)}</title>\n'
        f'  <desc id="desc">{html.escape(description)}</desc>\n'
        '  <rect width="1000" height="520" fill="#f7f9fc"/>\n'
        f"{body}</svg>\n"
    )


def feasibility_key(row: dict[str, str]) -> tuple[float, float, float]:
    """Return the same feasibility-first ordering used by the Rust archive."""

    constraint = float(row["constraint_l0"])
    score = float(row["estimated_score_l0"])
    if constraint <= L0_CONSTRAINT_THRESHOLD:
        return (0.0, -score, constraint)
    return (1.0, constraint, -score)


def convergence(root: Path) -> str:
    archives = {
        arm: read_csv(arm_directory(root, arm) / "archive.csv") for arm in ARMS
    }
    runs = read_runs(root)
    series = {}
    for arm, rows in archives.items():
        best = None
        values = []
        for row in rows:
            if best is None or feasibility_key(row) < feasibility_key(best):
                best = row
            values.append(float(best["constraint_l0"]))
        series[arm] = values
    transformed = [
        math.log10(1.0 + max(value, 0.0))
        for values in series.values()
        for value in values
    ]
    maximum = max(transformed, default=1.0)
    minimum = min(transformed, default=0.0)
    span = max(maximum - minimum, 1.0)
    maximum_candidates = max((len(values) for values in series.values()), default=1)
    body = [
        '  <text x="500" y="42" text-anchor="middle" '
        'font-family="system-ui" font-size="24" font-weight="700" fill="#132238">'
        "Lowest L0 constraint violation under the protocol budget</text>\n",
        '  <path d="M90 450 H950 M90 450 V80" stroke="#40556f" stroke-width="2"/>\n',
    ]
    for arm, values in series.items():
        points = []
        for index, value in enumerate(values):
            x = 90 + 860 * index / max(maximum_candidates - 1, 1)
            scaled = math.log10(1.0 + max(value, 0.0))
            y = 450 - 350 * (scaled - minimum) / span
            points.append(f"{x:.2f},{y:.2f}")
        if len(points) > 1:
            body.append(
                f'  <polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{COLORS[arm]}" stroke-width="4"/>\n'
            )
        if points:
            x, y = points[-1].split(",")
            body.append(
                f'  <circle cx="{x}" cy="{y}" r="6" fill="{COLORS[arm]}"/>\n'
            )
    for index, arm in enumerate(ARMS):
        x = 260 + index * 230
        body.append(
            f'  <rect x="{x}" y="475" width="24" height="5" fill="{COLORS[arm]}"/>'
            f'<text x="{x + 34}" y="484" font-family="system-ui" font-size="15" '
            f'fill="#26384f">{arm} ({runs[arm]["status"]})</text>\n'
        )
    body.append(
        '  <text x="520" y="510" text-anchor="middle" font-family="system-ui" '
        'font-size="14" fill="#40556f">accepted candidates</text>\n'
    )
    body.append(
        '  <text x="24" y="270" text-anchor="middle" font-family="system-ui" '
        'font-size="13" fill="#40556f" transform="rotate(-90 24 270)">'
        "log10(1 + violation), lower is better</text>\n"
    )
    return svg_document(
        "".join(body),
        "Route-search constraint convergence",
        "Lowest feasibility-first L0 constraint violation by accepted candidate "
        "for three configured campaign arms; failed arms retain their partial "
        "archive.",
    )


def niche_coverage(root: Path) -> str:
    runs = read_runs(root)
    body = [
        '  <text x="500" y="45" text-anchor="middle" font-family="system-ui" '
        'font-size="24" font-weight="700" fill="#132238">Occupied route niches</text>\n',
        '  <path d="M100 440 H940" stroke="#40556f" stroke-width="2"/>\n',
    ]
    maximum = max(run["budget"]["niches"] for run in runs.values()) or 1
    for index, arm in enumerate(ARMS):
        niches = runs[arm]["budget"]["niches"]
        height = 310 * niches / maximum
        x = 180 + index * 270
        body.extend(
            [
                f'  <rect x="{x}" y="{440 - height:.2f}" width="130" '
                f'height="{height:.2f}" rx="8" fill="{COLORS[arm]}"/>\n',
                f'  <text x="{x + 65}" y="{425 - height:.2f}" text-anchor="middle" '
                f'font-family="system-ui" font-size="22" font-weight="700" '
                f'fill="#132238">{niches}</text>\n',
                f'  <text x="{x + 65}" y="475" text-anchor="middle" '
                f'font-family="system-ui" font-size="16" fill="#26384f">{arm}</text>\n',
            ]
        )
    return svg_document(
        "".join(body),
        "Route niche coverage",
        "Number of structural route niches occupied by each protocol arm.",
    )


def surrogate_gap(root: Path) -> str:
    points = []
    for arm in ARMS:
        for row in read_csv(arm_directory(root, arm) / "promotions.csv"):
            if row["l1_score"] and row["surrogate_gap"]:
                points.append(
                    (
                        arm,
                        float(row["l0_estimated_score"]),
                        float(row["l1_score"]),
                        row["l1_threshold_passed"] == "true",
                    )
                )
    body = [
        '  <text x="500" y="45" text-anchor="middle" font-family="system-ui" '
        'font-size="24" font-weight="700" fill="#132238">L0–L1 surrogate gap</text>\n',
    ]
    if not points:
        body.append(
            '  <text x="500" y="260" text-anchor="middle" font-family="system-ui" '
            'font-size="19" fill="#40556f">Status placeholder: zero L1 promotions '
            "and zero measured gaps.</text>\n"
        )
    else:
        xs = [point[1] for point in points]
        ys = [point[2] for point in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xspan, yspan = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
        body.append(
            '  <path d="M90 450 H950 M90 450 V80" stroke="#40556f" stroke-width="2"/>\n'
        )
        for arm, xvalue, yvalue, passed in points:
            x = 90 + 860 * (xvalue - xmin) / xspan
            y = 450 - 350 * (yvalue - ymin) / yspan
            body.append(
                f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="8" '
                f'fill="{COLORS[arm]}" stroke="{"#183" if passed else "#b33"}" '
                'stroke-width="3"/>\n'
            )
    return svg_document(
        "".join(body),
        "L0 and L1 surrogate gap",
        "Promoted route L0 estimates versus impulsive Sims-Flanagan L1 scores.",
    )


def best_structure(root: Path) -> str:
    candidates = [
        (arm, row)
        for arm in ARMS
        for row in read_csv(arm_directory(root, arm) / "archive.csv")
    ]
    arm, best = min(candidates, key=lambda item: feasibility_key(item[1]))
    bodies = best["structure_key"].split("-")
    labels = [BODY.get(body, body) for body in bodies]
    body = [
        '  <text x="500" y="45" text-anchor="middle" font-family="system-ui" '
        'font-size="24" font-weight="700" fill="#132238">'
        "Feasibility-first leading L0 route structure</text>\n",
        '  <text x="500" y="83" text-anchor="middle" font-family="system-ui" '
        'font-size="15" fill="#a14d00">structure diagram, not a propagated trajectory</text>\n',
    ]
    for index, label in enumerate(labels):
        x = 80 + 840 * index / max(len(labels) - 1, 1)
        if index:
            prior = 80 + 840 * (index - 1) / max(len(labels) - 1, 1)
            body.append(
                f'  <path d="M{prior + 22:.2f} 270 H{x - 22:.2f}" '
                'stroke="#40556f" stroke-width="3"/>\n'
            )
        body.extend(
            [
                f'  <circle cx="{x:.2f}" cy="270" r="24" fill="#e7f2ff" '
                'stroke="#2774ae" stroke-width="2"/>\n',
                f'  <text x="{x:.2f}" y="277" text-anchor="middle" '
                'font-family="system-ui" font-size="15" font-weight="700" '
                f'fill="#132238">{html.escape(label)}</text>\n',
            ]
        )
    body.append(
        f'  <text x="500" y="355" text-anchor="middle" font-family="system-ui" '
        f'font-size="17" fill="#26384f">{arm} arm · variant '
        f'{html.escape(best["variant_key"])}</text>\n'
    )
    body.append(
        f'  <text x="500" y="390" text-anchor="middle" font-family="system-ui" '
        f'font-size="17" fill="#26384f">L0 constraint violation '
        f'{float(best["constraint_l0"]):.6g}; diagnostic score '
        f'{float(best["estimated_score_l0"]):.3f}</text>\n'
    )
    return svg_document(
        "".join(body),
        "Leading L0 route structure",
        "Body-order diagram of the feasibility-first leading L0 route across "
        "the supplied campaign arms.",
    )


def render_all(root: Path) -> dict[str, str]:
    return {
        "convergence.svg": convergence(root),
        "niche-coverage.svg": niche_coverage(root),
        "surrogate-gap.svg": surrogate_gap(root),
        "best-route-structure.svg": best_structure(root),
    }


def read_mga(root: Path) -> dict[str, dict[str, object]]:
    """Load the compact blind and assisted MGA publication evidence."""

    evidence = {}
    for arm in MGA_ARMS:
        directory = root / arm
        with (directory / "run.json").open(encoding="utf-8") as stream:
            run = json.load(stream)
        rows = [
            row
            for row in read_csv(directory / "archive.csv")
            if row["evaluation_found"].lower() == "true"
            and math.isfinite(float(row["mga_score"]))
            and float(row["mga_score"]) > 0.0
        ]
        ranked = sorted(rows, key=lambda row: float(row["mga_score"]), reverse=True)
        portfolio_size = int(run["configuration"]["portfolio_size"])
        evidence[arm] = {
            "run": run,
            "rows": rows,
            "best": float(ranked[0]["mga_score"]),
            "portfolio": sum(
                float(row["mga_score"]) for row in ranked[:portfolio_size]
            ),
        }
    return evidence


def mga_portfolio_results(root: Path) -> str:
    """Compare the declared best-20 metric and secondary best-route score."""

    evidence = read_mga(root)
    maximum_portfolio = max(float(row["portfolio"]) for row in evidence.values())
    maximum_best = max(float(row["best"]) for row in evidence.values())
    body = [
        '  <text x="500" y="42" text-anchor="middle" font-family="system-ui" '
        'font-size="24" font-weight="700" fill="#132238">Seed-42 MGA portfolio results</text>\n',
        '  <text x="270" y="82" text-anchor="middle" font-family="system-ui" '
        'font-size="17" font-weight="700" fill="#26384f">Declared best-20 sum</text>\n',
        '  <text x="755" y="82" text-anchor="middle" font-family="system-ui" '
        'font-size="17" font-weight="700" fill="#26384f">Best single route</text>\n',
    ]
    for index, arm in enumerate(MGA_ARMS):
        y = 120 + index * 82
        portfolio = float(evidence[arm]["portfolio"])
        best = float(evidence[arm]["best"])
        portfolio_width = 300.0 * portfolio / maximum_portfolio
        best_width = 260.0 * best / maximum_best
        color = MGA_COLORS[arm]
        label = MGA_LABELS[arm]
        body.extend(
            [
                f'  <text x="18" y="{y + 24}" font-family="system-ui" font-size="15" '
                f'fill="#26384f">{html.escape(label)}</text>\n',
                f'  <rect x="140" y="{y}" width="{portfolio_width:.2f}" height="32" '
                f'rx="5" fill="{color}"/>\n',
                f'  <text x="{148 + portfolio_width:.2f}" y="{y + 23}" '
                f'font-family="system-ui" font-size="14" fill="#132238">{portfolio / 1.0e6:.3f} M</text>\n',
                f'  <rect x="625" y="{y}" width="{best_width:.2f}" height="32" '
                f'rx="5" fill="{color}"/>\n',
                f'  <text x="{633 + best_width:.2f}" y="{y + 23}" '
                f'font-family="system-ui" font-size="14" fill="#132238">{best / 1.0e6:.3f} M</text>\n',
            ]
        )
    body.extend(
        [
            '  <path d="M500 72 V445" stroke="#ccd5e0" stroke-width="1"/>\n',
            '  <text x="500" y="480" text-anchor="middle" font-family="system-ui" '
            'font-size="14" fill="#40556f">Higher is better · assisted uses prior baseline evidence</text>\n',
        ]
    )
    return svg_document(
        "".join(body),
        "Seed-42 MGA portfolio comparison",
        "Best-20 portfolio sum and best single impulsive MGA score for random, "
        "evolutionary, cold Gemma, and the separately identified prior-informed "
        "Gemma-assisted follow-up.",
    )


def mga_length_mix(root: Path) -> str:
    """Show the route-length collapse and its assisted repair."""

    evidence = read_mga(root)
    bands = ((3, 6, "3–6"), (7, 9, "7–9"), (10, 11, "10–11"), (12, 14, "12–14"))
    band_colors = ("#9ecae1", "#56b4a9", "#f3ba63", "#d66b6b")
    body = [
        '  <text x="500" y="42" text-anchor="middle" font-family="system-ui" '
        'font-size="24" font-weight="700" fill="#132238">Accepted body orders by encounter count</text>\n',
    ]
    for index, arm in enumerate(MGA_ARMS):
        y = 105 + index * 78
        rows = evidence[arm]["rows"]
        counts = []
        for lower, upper, _ in bands:
            counts.append(
                sum(lower <= len(row["structure_key"].split("-")) <= upper for row in rows)
            )
        body.append(
            f'  <text x="22" y="{y + 27}" font-family="system-ui" font-size="15" '
            f'fill="#26384f">{html.escape(MGA_LABELS[arm])}</text>\n'
        )
        x = 160.0
        for count, color in zip(counts, band_colors, strict=True):
            width = 7.4 * count
            body.append(
                f'  <rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="38" '
                f'fill="{color}" stroke="#f7f9fc" stroke-width="1"/>\n'
            )
            if count >= 4:
                body.append(
                    f'  <text x="{x + width / 2:.2f}" y="{y + 26}" text-anchor="middle" '
                    f'font-family="system-ui" font-size="14" font-weight="700" '
                    f'fill="#132238">{count}</text>\n'
                )
            x += width
    for index, ((_, _, label), color) in enumerate(zip(bands, band_colors, strict=True)):
        x = 190 + index * 190
        body.extend(
            [
                f'  <rect x="{x}" y="438" width="22" height="16" fill="{color}"/>\n',
                f'  <text x="{x + 30}" y="452" font-family="system-ui" font-size="14" '
                f'fill="#26384f">{label} encounters</text>\n',
            ]
        )
    body.append(
        '  <text x="530" y="493" text-anchor="middle" font-family="system-ui" '
        'font-size="14" fill="#40556f">Each bar contains 100 accepted, unique body orders</text>\n'
    )
    return svg_document(
        "".join(body),
        "MGA route-length distribution",
        "Encounter-count bands for 100 accepted routes per strategy. Cold Gemma "
        "concentrates in the longest band while the assisted policy concentrates "
        "in the 7–9 encounter band.",
    )


def render_mga_all(root: Path) -> dict[str, str]:
    return {
        "mga-portfolio-results.svg": mga_portfolio_results(root),
        "mga-length-mix.svg": mga_length_mix(root),
    }


def write_or_check(rendered: dict[str, str], output: Path, check: bool) -> list[Path]:
    """Write one renderer group or return its stale paths."""

    stale = []
    for name, content in rendered.items():
        path = output / name
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(path)
    return stale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path
    )
    parser.add_argument("--output", type=Path, default=Path("images"))
    parser.add_argument("--mga-results", type=Path)
    parser.add_argument("--mga-output", type=Path, default=Path("images"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    default_invocation = args.results is None
    result_root = args.results or Path("results/protocol-evidence")
    stale = write_or_check(render_all(result_root), args.output, args.check)
    if default_invocation or args.mga_results is not None:
        mga_root = args.mga_results or Path("results/mga-matched-seed42")
        stale.extend(
            write_or_check(render_mga_all(mga_root), args.mga_output, args.check)
        )
    if stale:
        raise SystemExit("missing or stale figures:\n" + "\n".join(map(str, stale)))
    if args.check:
        print("route-search figures are current")


if __name__ == "__main__":
    main()
