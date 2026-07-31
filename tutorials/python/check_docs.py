"""Validate local documentation links and their public-repository status."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def local_targets(readme: Path) -> list[tuple[str, Path]]:
    targets: list[tuple[str, Path]] = []
    for raw in LINK.findall(readme.read_text(encoding="utf-8")):
        target = raw.strip().strip("<>")
        if not target or target.startswith(SKIP_PREFIXES):
            continue
        path_text = unquote(target.split("#", 1)[0])
        if not path_text:
            continue
        targets.append((raw, (readme.parent / path_text).resolve()))
    return targets


def git_root(start: Path) -> Path | None:
    completed = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return Path(completed.stdout.strip()).resolve()


def tracked_paths(repository: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return {
        entry.decode("utf-8")
        for entry in completed.stdout.split(b"\0")
        if entry
    }


def is_tracked(target: Path, repository: Path, tracked: set[str]) -> bool:
    try:
        relative = target.relative_to(repository).as_posix()
    except ValueError:
        return False
    return relative in tracked or (
        target.is_dir()
        and any(path.startswith(f"{relative}/") for path in tracked)
    )


def main() -> int:
    tutorial_root = Path(__file__).resolve().parents[1]
    public_root = tutorial_root.parent
    repository = git_root(public_root)
    # The development tree currently stages `public/` inside a larger private
    # repository. Enforce tracked-file status only after `public/` is the
    # standalone repository root, as it is in CI and after publication.
    enforce_tracked = repository == public_root
    tracked = tracked_paths(repository) if enforce_tracked and repository else set()
    missing: list[str] = []
    untracked: list[str] = []
    documents = sorted(
        {
            *public_root.glob("*.md"),
            *(public_root / "docs").rglob("*.md"),
            *(public_root / "foundations").rglob("*.md"),
            *(public_root / "benchmarks").rglob("*.md"),
            *(public_root / "crates").rglob("*.md"),
            *(public_root / "examples" / "data").rglob("*.md"),
            *(public_root / "python").rglob("*.md"),
            *tutorial_root.rglob("*.md"),
        }
    )
    for document in documents:
        for raw, target in local_targets(document):
            if not target.exists():
                missing.append(f"{document.relative_to(public_root)}: {raw}")
            elif enforce_tracked and not is_tracked(target, public_root, tracked):
                untracked.append(f"{document.relative_to(public_root)}: {raw}")
    if missing:
        print("missing local documentation links:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
    if untracked:
        print("local links absent from the public git index:", file=sys.stderr)
        for item in untracked:
            print(f"  {item}", file=sys.stderr)
    if missing or untracked:
        return 1
    status = "resolve and are tracked" if enforce_tracked else "resolve"
    print(f"all local documentation links {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
