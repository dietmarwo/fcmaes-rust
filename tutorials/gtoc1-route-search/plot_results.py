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
BODY = {"1": "Me", "2": "V", "3": "E", "4": "Ma", "5": "J", "6": "S", "10": "A"}
L0_CONSTRAINT_THRESHOLD = 1.0e-8


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


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
    archives = {arm: read_csv(root / arm / "archive.csv") for arm in ARMS}
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
    body = [
        '  <text x="500" y="42" text-anchor="middle" '
        'font-family="system-ui" font-size="24" font-weight="700" fill="#132238">'
        "Lowest L0 constraint violation under the protocol budget</text>\n",
        '  <path d="M90 450 H950 M90 450 V80" stroke="#40556f" stroke-width="2"/>\n',
    ]
    for arm, values in series.items():
        points = []
        for index, value in enumerate(values):
            x = 90 + 860 * index / max(len(values) - 1, 1)
            scaled = math.log10(1.0 + max(value, 0.0))
            y = 450 - 350 * (scaled - minimum) / span
            points.append(f"{x:.2f},{y:.2f}")
        body.append(
            f'  <polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{COLORS[arm]}" stroke-width="4"/>\n'
        )
    for index, arm in enumerate(ARMS):
        x = 260 + index * 230
        body.append(
            f'  <rect x="{x}" y="475" width="24" height="5" fill="{COLORS[arm]}"/>'
            f'<text x="{x + 34}" y="484" font-family="system-ui" font-size="15" '
            f'fill="#26384f">{arm}</text>\n'
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
        "for three protocol-fixture arms.",
    )


def niche_coverage(root: Path) -> str:
    runs = {}
    for arm in ARMS:
        with (root / arm / "run.json").open(encoding="utf-8") as stream:
            runs[arm] = json.load(stream)
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
        for row in read_csv(root / arm / "promotions.csv"):
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
    rows = read_csv(root / "agent" / "archive.csv")
    best = min(rows, key=feasibility_key)
    bodies = best["structure_key"].split("-")
    labels = [BODY.get(body, body) for body in bodies]
    body = [
        '  <text x="500" y="45" text-anchor="middle" font-family="system-ui" '
        'font-size="24" font-weight="700" fill="#132238">'
        "Lowest-violation mock-arm route structure</text>\n",
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
        f'font-size="17" fill="#26384f">variant {html.escape(best["variant_key"])}</text>\n'
    )
    body.append(
        f'  <text x="500" y="390" text-anchor="middle" font-family="system-ui" '
        f'font-size="17" fill="#26384f">L0 constraint violation '
        f'{float(best["constraint_l0"]):.6g}; diagnostic score '
        f'{float(best["estimated_score_l0"]):.3f}</text>\n'
    )
    return svg_document(
        "".join(body),
        "Closest protocol route structure",
        "Body-order diagram of the lowest-violation L0 route in the mock protocol fixture.",
    )


def render_all(root: Path) -> dict[str, str]:
    return {
        "convergence.svg": convergence(root),
        "niche-coverage.svg": niche_coverage(root),
        "surrogate-gap.svg": surrogate_gap(root),
        "best-route-structure.svg": best_structure(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=Path("results/protocol-evidence")
    )
    parser.add_argument("--output", type=Path, default=Path("images"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_all(args.results)
    stale = []
    for name, content in rendered.items():
        path = args.output / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(path)
    if stale:
        raise SystemExit("missing or stale figures:\n" + "\n".join(map(str, stale)))
    if args.check:
        print("route-search figures are current")


if __name__ == "__main__":
    main()
