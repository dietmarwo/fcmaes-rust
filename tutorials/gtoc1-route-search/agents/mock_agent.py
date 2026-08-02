#!/usr/bin/env python3
"""Deterministic no-dependency agent for CI and protocol demonstrations."""

import json
import sys


ROUTES = [
    (
        ["Earth", "Venus", "Earth", "Earth", "Earth",
         "Jupiter", "Saturn", "Jupiter", "TW229"],
        [False, False, False, False, False, False, True, True],
        "Historical JPL control route.",
    ),
    (
        ["Earth", "Venus", "Venus", "Earth", "Earth", "Earth", "Earth",
         "Jupiter", "Saturn", "Jupiter", "TW229"],
        [False, False, False, False, False, False, False, False, True, True],
        "Historical JPL2 regression route.",
    ),
    (
        ["Earth", "Venus", "Venus", "Earth", "Venus", "Venus", "Earth",
         "Earth", "Saturn", "Jupiter", "TW229"],
        [False, False, False, False, False, False, False, False, True, True],
        "Historical Jena regression route.",
    ),
    (
        ["Earth", "Venus", "Venus", "Earth", "Earth", "Venus", "Venus",
         "Earth", "Venus", "Earth", "Jupiter", "Saturn", "Jupiter", "TW229"],
        [False, False, False, False, False, False, False, False, False,
         False, False, True, True],
        "Historical Deimos regression route.",
    ),
    (
        ["Earth", "Venus", "Earth", "Venus", "Earth",
         "Jupiter", "Earth", "Saturn", "TW229"],
        [False, False, False, False, False, False, True, True],
        "Deterministic diverse mock route.",
    ),
]


def main() -> None:
    request = json.load(sys.stdin)
    start = (int(request["proposal_attempt"]) - 1) % len(ROUTES)
    count = int(request.get("batch_size", 1))
    candidates = []
    for offset in range(count):
        bodies, _clockwise, rationale = ROUTES[(start + offset) % len(ROUTES)]
        candidates.append({
            "bodies": bodies,
            "rationale": rationale,
        })
    json.dump({
        "candidates": candidates,
        "usage": {
            "provider": "mock",
            "model": "deterministic-python-v1",
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        },
    }, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
