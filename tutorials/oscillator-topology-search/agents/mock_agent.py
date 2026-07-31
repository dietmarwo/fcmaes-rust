#!/usr/bin/env python3
"""Deterministic transport fixture; never use in a scientific comparison."""

from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.load(sys.stdin)
    # Three self edges connect all genes but form no checked-in reference or
    # cross-gene motif. Eight sign variants exercise deduplication.
    value = max(0, request.get("proposal_attempt", 1) - 1)
    edges = [1 + ((value >> bit) & 1) for bit in range(3)] + [0] * 6
    json.dump(
        {"edges": edges, "input_tokens": 0, "output_tokens": 0},
        sys.stdout,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
