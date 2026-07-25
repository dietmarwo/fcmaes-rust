#!/usr/bin/env python3
"""Assemble and optionally build the mdBook from canonical repository files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "target" / "mdbook-src"
SOURCE = STAGING / "src"
OUTPUT = ROOT / "target" / "book"

TOP_LEVEL_FILES = ("README.md", "CHANGELOG.md", "RELEASING.md", "ai-context.md")
SOURCE_DIRECTORIES = ("docs", "tutorials", "benchmarks", "examples", "crates")
IGNORED_NAMES = {
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".pyright",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "htmlcov",
    "target",
    "venv",
}


def ignored(_directory: str, names: list[str]) -> set[str]:
    """Return generated and environment-specific entries to omit."""

    return {
        name
        for name in names
        if name in IGNORED_NAMES
        or name.endswith((".egg-info", ".profraw", ".pyc", ".pyo"))
    }


def prepare() -> None:
    """Create a clean mdBook source tree without changing canonical files."""

    if STAGING.exists():
        shutil.rmtree(STAGING)
    SOURCE.mkdir(parents=True)

    shutil.copy2(ROOT / "book" / "book.toml", STAGING / "book.toml")
    shutil.copy2(ROOT / "book" / "SUMMARY.md", SOURCE / "SUMMARY.md")
    shutil.copy2(ROOT / "book" / "introduction.md", SOURCE / "introduction.md")
    shutil.copy2(ROOT / "book" / "tutorials.md", SOURCE / "tutorials.md")

    for filename in TOP_LEVEL_FILES:
        shutil.copy2(ROOT / filename, SOURCE / filename)
    for directory in SOURCE_DIRECTORIES:
        shutil.copytree(
            ROOT / directory,
            SOURCE / directory,
            ignore=ignored,
            copy_function=shutil.copy2,
        )


def main() -> None:
    """Prepare the source tree and invoke mdBook unless only staging is wanted."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="assemble target/mdbook-src but do not invoke mdbook",
    )
    args = parser.parse_args()

    prepare()
    if not args.prepare_only:
        subprocess.run(
            [
                "mdbook",
                "build",
                str(STAGING),
                "--dest-dir",
                str(OUTPUT),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
