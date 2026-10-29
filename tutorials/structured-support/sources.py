"""Locate the sibling checkouts that the structured workshops import from.

X3 and X4 reuse tutorial support code from LP-IP-craft and SQP-craft. Clone
those repositories next to fcmaes-rust, or set ``STRUCTURED_SOURCES`` to the
directory that holds all checkouts (the optimization-jobs driver sets it).
"""

import os
from pathlib import Path


class CheckFailure(AssertionError):
    """An independent tutorial check failed."""


def require(condition, message="independent check failed"):
    """Raise ``CheckFailure`` unless ``condition`` holds.

    Tutorial checks use this instead of ``assert`` so that ``python -O`` and
    ``PYTHONOPTIMIZE`` cannot remove them.
    """
    if not condition:
        raise CheckFailure(message)


def sources_root():
    configured = os.environ.get("STRUCTURED_SOURCES")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[3]


def sibling(repo, *parts):
    """Path inside the sibling checkout ``repo``, matched case-insensitively."""
    root = sources_root()
    path = root / repo
    if not path.exists() and root.is_dir():
        matches = [p for p in root.iterdir() if p.name.casefold() == repo.casefold()]
        if len(matches) == 1:
            path = matches[0]
    path = path.joinpath(*parts)
    if not path.exists():
        raise SystemExit(
            f"{path} not found: clone {repo} next to fcmaes-rust, or set "
            "STRUCTURED_SOURCES to the directory that holds the checkouts"
        )
    return path
