#!/usr/bin/env python3
"""Replace one problem block in a GTOP result CSV without losing raw rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return reader.fieldnames, list(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fields, base = read_rows(args.base)
    replacement_fields, replacement = read_rows(args.replacement)
    if fields != replacement_fields:
        raise ValueError("base and replacement CSV schemas differ")
    if not replacement or any(row["problem"] != args.problem for row in replacement):
        raise ValueError("replacement must contain only the selected problem")

    kept = [row for row in base if row["problem"] != args.problem]
    removed = len(base) - len(kept)
    if removed != len(replacement):
        raise ValueError(
            f"replacing {removed} rows with {len(replacement)} rows changes the protocol"
        )
    merged = kept + replacement
    keys = {
        (row["phase"], row["arm"], row["problem"], row["run"], row["seed"], row["workers"])
        for row in merged
    }
    if len(keys) != len(merged):
        raise ValueError("merged result contains duplicate protocol rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    print(f"replaced {removed} {args.problem} rows; wrote {len(merged)} rows")


if __name__ == "__main__":
    main()
