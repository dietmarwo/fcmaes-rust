#!/usr/bin/env python3
"""Check documentation inventories and version references against the tree."""

from __future__ import annotations

import re
import sys
import tomllib
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]


def number_word(value: int) -> str:
    """Return the lower-case English spelling needed by the documentation."""

    units = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    )
    if value < len(units):
        return units[value]
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty")
    if value < 70:
        quotient, remainder = divmod(value, 10)
        return tens[quotient] if remainder == 0 else f"{tens[quotient]}-{units[remainder]}"
    raise ValueError(f"unsupported documentation count: {value}")


def tutorial_rows(document: str) -> list[str]:
    """Extract directory links from the canonical tutorial overview table."""

    overview = document.split("Each directory is a standalone Cargo workspace.", 1)[0]
    return re.findall(r"^\| \[[^]]+\]\(([^/#]+)/\) \|", overview, re.MULTILINE)


def named_table_rows(document: str, heading: str, trailer: str) -> int:
    """Count data rows in one Markdown table delimited by prose."""

    section = document.split(heading, 1)[1].split(trailer, 1)[0]
    return sum(
        line.startswith("| ")
        and not line.startswith("| Tutorial ")
        and not line.startswith("|---")
        for line in section.splitlines()
    )


def main() -> int:
    """Report all consistency failures, returning nonzero when any exist."""

    failures: list[str] = []
    tutorial_dirs = sorted(
        path.name
        for path in (ROOT / "tutorials").iterdir()
        if (path / "Cargo.toml").is_file()
    )
    expected = set(tutorial_dirs)
    count = len(tutorial_dirs)
    count_word = number_word(count)

    tutorial_index = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    rows = tutorial_rows(tutorial_index)
    if Counter(rows) != Counter(tutorial_dirs):
        failures.append(
            "tutorials/README.md overview does not contain each Cargo tutorial exactly once"
        )

    summary = (ROOT / "book" / "SUMMARY.md").read_text(encoding="utf-8")
    summary_links = re.findall(r"\]\(tutorials/([^/#]+)/README\.md", summary)
    summary_primary = [name for name in summary_links if name in expected]
    if Counter(summary_primary) != Counter(tutorial_dirs):
        failures.append("book/SUMMARY.md does not contain each Cargo tutorial exactly once")

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    root_links = re.findall(r"\]\(tutorials/([^/#]+)/README\.md", root_readme)
    if Counter(root_links) != Counter(tutorial_dirs):
        failures.append("README.md does not contain each Cargo tutorial exactly once")

    book_overview = (ROOT / "book" / "tutorials.md").read_text(encoding="utf-8")
    overview_rows = named_table_rows(
        book_overview,
        "The tutorials put native Rust objective functions",
        "The canonical, detailed",
    )
    if overview_rows != count:
        failures.append(
            f"book/tutorials.md has {overview_rows} overview rows for {count} tutorials"
        )

    ai_context = (ROOT / "ai-context.md").read_text(encoding="utf-8")
    lesson_rows = named_table_rows(
        ai_context,
        "## Lessons from the application tutorials",
        "MODE and MAP-Elites are complementary",
    )
    if lesson_rows != count:
        failures.append(
            f"ai-context.md has {lesson_rows} lesson rows for {count} tutorials"
        )

    count_claims = {
        "README.md": f"{count_word}.",
        "book/tutorials.md": f"{count_word} application tutorials cover:",
        "docs/README.md": f"{count_word} native optimization applications",
        "docs/architecture.md": f"The {count_word} application directories",
        "docs/choosing-an-optimizer.md": f"The {count_word} [application tutorials]",
        "docs/getting-started.md": f"all {count_word} application",
        "tutorials/README.md": f"The {count_word} applications",
        "ai-context.md": f"The {count_word} standalone tutorials",
    }
    for filename, claim in count_claims.items():
        text = (ROOT / filename).read_text(encoding="utf-8")
        if claim.casefold() not in text.casefold():
            failures.append(f"{filename} is missing derived tutorial-count claim: {claim}")

    pinned = sum(
        'fcmaes-core = "=0.1.3"' in (ROOT / "tutorials" / name / "Cargo.toml").read_text()
        for name in tutorial_dirs
    )
    pinned_claim = f"{number_word(pinned)} tutorials"
    if pinned_claim.casefold() not in tutorial_index.casefold():
        failures.append(
            f"tutorials/README.md is missing the derived pinned-count claim: {pinned_claim}"
        )

    multi_objective = sum(
        bool(
            re.search(
                r"\bMODE\b",
                (ROOT / "tutorials" / name / "README.md").read_text(encoding="utf-8"),
            )
        )
        for name in tutorial_dirs
    )
    mo_claim = f"{number_word(multi_objective)} application tutorials retain multi-objective"
    if mo_claim.casefold() not in root_readme.casefold():
        failures.append(f"README.md is missing the derived MODE-count claim: {mo_claim}")

    with (ROOT / "Cargo.toml").open("rb") as source:
        version = tomllib.load(source)["workspace"]["package"]["version"]
    with (ROOT / "Cargo.lock").open("rb") as source:
        packages = tomllib.load(source)["package"]
    locked = {package["name"]: package["version"] for package in packages}
    for package in ("fcmaes-core", "fcmaes-gtop", "fcmaes-py", "fcmaes-examples"):
        if locked.get(package) != version:
            failures.append(
                f"Cargo.lock has {package} {locked.get(package)!r}, expected {version}"
            )

    releasing = (ROOT / "RELEASING.md").read_text(encoding="utf-8")
    tag_section = releasing.split("## Tag and publish", 1)[1].split(
        "## Post-release verification", 1
    )[0]
    if "scripts/package_version.py" not in tag_section or '"v${release_version}"' not in tag_section:
        failures.append("RELEASING.md tag commands are not derived from the package version")

    relative_links = 0
    for source in ROOT.rglob("*.md"):
        relative = source.relative_to(ROOT)
        if relative.parts[0] in {"book", "target"} or any(
            part.startswith(".") for part in relative.parts
        ):
            continue
        document = source.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", document):
            raw = match.group(1).strip()
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1]
            elif " " in raw:
                raw = raw.split(maxsplit=1)[0]
            if not raw or raw.startswith(
                ("#", "http://", "https://", "mailto:", "tel:", "data:", "javascript:", "/")
            ):
                continue
            path = unquote(raw.split("#", 1)[0].split("?", 1)[0])
            if not path:
                continue
            relative_links += 1
            target = (source.parent / path).resolve()
            try:
                target.relative_to(ROOT)
            except ValueError:
                continue
            if not target.exists():
                failures.append(
                    f"{relative.as_posix()} links to missing relative target {raw}"
                )

    if failures:
        print("documentation consistency failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(
        f"documentation inventories agree: {count} tutorials, "
        f"{multi_objective} with MODE, {pinned} registry-pinned; "
        f"workspace packages are {version}; {relative_links} source links resolve"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
