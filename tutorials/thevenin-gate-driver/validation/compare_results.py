#!/usr/bin/env python3
"""Compare thevenin and ngspice measurements and enforce publication gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATES = {
    "rise_time_abs_ns": 0.01,
    "overshoot_abs_percentage_points": 0.01,
    "peak_current_abs_a": 0.01,
    "settling_time_abs_ns": 0.1,
    "final_voltage_abs_v": 0.01,
}


def read_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    indexed = {int(row["point_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{path} contains duplicate point identifiers")
    return indexed


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty comparison")
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thevenin",
        type=Path,
        default=ROOT / "results" / "publication" / "validation" / "thevenin.csv",
    )
    parser.add_argument(
        "--ngspice",
        type=Path,
        default=ROOT / "results" / "publication" / "validation" / "ngspice.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "publication" / "validation",
    )
    arguments = parser.parse_args()

    thevenin = read_rows(arguments.thevenin)
    ngspice = read_rows(arguments.ngspice)
    if thevenin.keys() != ngspice.keys():
        missing_thevenin = sorted(ngspice.keys() - thevenin.keys())
        missing_ngspice = sorted(thevenin.keys() - ngspice.keys())
        raise ValueError(
            "comparison identifiers differ: "
            f"missing_thevenin={missing_thevenin}, "
            f"missing_ngspice={missing_ngspice}"
        )
    fields = [
        "point_id",
        "u_resistance",
        "u_snubber",
        "resistance_ohm",
        "snubber_resistance_ohm",
        "thevenin_rise_time_ns",
        "ngspice_rise_time_ns",
        "rise_time_abs_ns",
        "thevenin_overshoot_percent",
        "ngspice_overshoot_percent",
        "overshoot_abs_percentage_points",
        "thevenin_peak_current_a",
        "ngspice_peak_current_a",
        "peak_current_abs_a",
        "thevenin_settling_time_ns",
        "ngspice_settling_time_ns",
        "settling_time_abs_ns",
        "thevenin_final_voltage_v",
        "ngspice_final_voltage_v",
        "final_voltage_abs_v",
    ]
    comparisons: list[dict[str, float | int]] = []
    for point_id in sorted(thevenin):
        first = thevenin[point_id]
        second = ngspice[point_id]
        row: dict[str, float | int] = {
            "point_id": point_id,
            "u_resistance": float(first["u_resistance"]),
            "u_snubber": float(first["u_snubber"]),
            "resistance_ohm": float(first["resistance_ohm"]),
            "snubber_resistance_ohm": float(
                first["snubber_resistance_ohm"]
            ),
            "thevenin_rise_time_ns": float(first["rise_time_ns"]),
            "ngspice_rise_time_ns": float(second["rise_time_ns"]),
            "thevenin_overshoot_percent": float(
                first["overshoot_percent"]
            ),
            "ngspice_overshoot_percent": float(
                second["overshoot_percent"]
            ),
            "thevenin_peak_current_a": float(
                first["peak_driver_current_a"]
            ),
            "ngspice_peak_current_a": float(
                second["peak_driver_current_a"]
            ),
            "thevenin_settling_time_ns": float(
                first["settling_time_ns"]
            ),
            "ngspice_settling_time_ns": float(
                second["settling_time_ns"]
            ),
            "thevenin_final_voltage_v": float(
                first["final_gate_voltage_v"]
            ),
            "ngspice_final_voltage_v": float(
                second["final_gate_voltage_v"]
            ),
        }
        row["rise_time_abs_ns"] = abs(
            float(row["thevenin_rise_time_ns"])
            - float(row["ngspice_rise_time_ns"])
        )
        row["overshoot_abs_percentage_points"] = abs(
            float(row["thevenin_overshoot_percent"])
            - float(row["ngspice_overshoot_percent"])
        )
        row["peak_current_abs_a"] = abs(
            float(row["thevenin_peak_current_a"])
            - float(row["ngspice_peak_current_a"])
        )
        row["settling_time_abs_ns"] = abs(
            float(row["thevenin_settling_time_ns"])
            - float(row["ngspice_settling_time_ns"])
        )
        row["final_voltage_abs_v"] = abs(
            float(row["thevenin_final_voltage_v"])
            - float(row["ngspice_final_voltage_v"])
        )
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError(f"non-finite comparison at point {point_id}")
        comparisons.append(row)

    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)

    metrics = {}
    passed = True
    for name, limit in GATES.items():
        values = [float(row[name]) for row in comparisons]
        maximum = max(values)
        metrics[name] = {
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "maximum": maximum,
            "limit": limit,
            "passed": maximum <= limit,
        }
        passed &= maximum <= limit
    summary = {
        "schema_version": 1,
        "rows": len(comparisons),
        "passed": passed,
        "gate_semantics": "every maximum absolute difference must be <= its limit",
        "metrics": metrics,
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for name, values in metrics.items():
        print(
            f"{name}: max={values['maximum']:.9g}, "
            f"limit={values['limit']:.9g}, passed={values['passed']}"
        )
    if not passed:
        print("cross-simulator validation failed")
        return 1
    print(f"cross-simulator validation passed for {len(comparisons)} designs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
