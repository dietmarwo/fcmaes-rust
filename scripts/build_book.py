#!/usr/bin/env python3
"""Assemble and optionally build the mdBook from canonical repository files."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "target" / "mdbook-src"
SOURCE = STAGING / "src"
OUTPUT = ROOT / "target" / "book"
PRINT_OUTPUT = ROOT / "target" / "book-pdf"

TOP_LEVEL_FILES = (
    "README.md",
    "CHANGELOG.md",
    "RELEASING.md",
    "SECURITY.md",
    "ai-context.md",
    "AGENTS.md",
    "LICENSE",
)
SOURCE_DIRECTORIES = (
    "docs",
    "foundations",
    "tutorials",
    "benchmarks",
    "examples",
    "crates",
)
README_LINK = re.compile(
    r"(?<=\()(?P<target>[^)\s]+README\.md)(?P<fragment>#[^)\s]*)?(?=\))"
)
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


def rewrite_readme_links() -> None:
    """Point prose links at the index pages emitted for directory READMEs.

    mdBook correctly treats a ``README.md`` listed in ``SUMMARY.md`` as the
    directory's ``index.html`` chapter, but prose links to the same source are
    otherwise rendered as nonexistent ``README.html`` URLs. Keep the canonical
    Markdown convenient on GitHub and normalize only the staged book sources.
    """

    def replacement(match: re.Match[str]) -> str:
        target = match.group("target")
        if "://" in target:
            return match.group(0)
        directory = target[: -len("README.md")] or "./"
        return f"{directory}{match.group('fragment') or ''}"

    for document in SOURCE.rglob("*.md"):
        if document.name == "SUMMARY.md":
            continue
        text = document.read_text(encoding="utf-8")
        rendered = README_LINK.sub(replacement, text)
        if rendered != text:
            document.write_text(rendered, encoding="utf-8")


FENCE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})")
MARKDOWN_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def escape_math(tex: str) -> str:
    """Backslash-escape TeX so CommonMark passes it to MathJax unchanged.

    pulldown-cmark treats a backslash before ASCII punctuation as an escape,
    reads ``_``/``*`` as emphasis and applies smart punctuation. Escaping
    every punctuation character makes the rendered text equal the TeX source.
    Blank lines would split the formula across paragraphs, and trailing
    spaces would become ``<br>``, so both are removed.
    """

    lines = [line.rstrip() for line in tex.split("\n")]
    return "\n".join(
        "".join("\\" + c if c in MARKDOWN_PUNCTUATION else c for c in line)
        for line in lines
        if line.strip()
    )


def inline_dollar_end(text: str, start: int) -> int | None:
    """Return the closing ``$`` of inline math opened at ``start``, if any.

    Pandoc's rule: the opening ``$`` is followed by a non-space, the closing
    one is preceded by a non-space and not followed by a digit, and both lie
    on the same line. This keeps prices such as ``$5 and $10`` as text.
    """

    if start + 1 >= len(text) or text[start + 1] in " \t\n$":
        return None
    position = start + 1
    while position < len(text) and text[position] != "\n":
        character = text[position]
        if character == "\\":
            position += 2
            continue
        if character == "$":
            if text[position - 1] in " \t":
                return None
            if position + 1 < len(text) and text[position + 1].isdigit():
                return None
            return position
        position += 1
    return None


def convert_math_prose(text: str) -> tuple[str, int]:
    """Rewrite GitHub/LaTeX math delimiters in prose as mdBook MathJax spans.

    Sources keep the delimiters GitHub renders (``$...$``, ``$$...$$``) or
    plain LaTeX (``\\(...\\)``, ``\\[...\\]``). mdBook's MathJax expects
    ``\\\\(...\\\\)`` and ``\\\\[...\\\\]`` with Markdown-escaped contents.
    Inline code spans are copied unchanged.
    """

    output: list[str] = []
    count = 0
    position = 0
    while position < len(text):
        character = text[position]
        if character == "`":
            run = len(text[position:]) - len(text[position:].lstrip("`"))
            closing = re.compile(r"(?<!`)" + "`" * run + r"(?!`)")
            match = closing.search(text, position + run)
            end = match.end() if match else position + run
            output.append(text[position:end])
            position = end
            continue
        display = None
        if text.startswith("$$", position):
            end = text.find("$$", position + 2)
            if end != -1:
                display = (text[position + 2 : end], end + 2, "[", "]")
        elif text.startswith("\\[", position):
            end = text.find("\\]", position + 2)
            if end != -1:
                display = (text[position + 2 : end], end + 2, "[", "]")
        elif text.startswith("\\(", position):
            end = text.find("\\)", position + 2)
            if end != -1 and "\n\n" not in text[position:end]:
                display = (text[position + 2 : end], end + 2, "(", ")")
        elif character == "$":
            end = inline_dollar_end(text, position)
            if end is not None:
                display = (text[position + 1 : end], end + 1, "(", ")")
        if display is not None:
            tex, end, opening, closing_delimiter = display
            output.append(
                f"\\\\{opening}{escape_math(tex.strip())}\\\\{closing_delimiter}"
            )
            count += 1
            position = end
            continue
        if character == "\\":
            output.append(text[position : position + 2])
            position += 2
            continue
        output.append(character)
        position += 1
    return "".join(output), count


def convert_math(document: str) -> tuple[str, int]:
    """Convert math outside fenced code blocks; return text and span count."""

    pieces: list[str] = []
    prose: list[str] = []
    count = 0
    fence: str | None = None

    def flush() -> None:
        nonlocal count
        if prose:
            converted, found = convert_math_prose("".join(prose))
            pieces.append(converted)
            count += found
            prose.clear()

    for line in document.splitlines(keepends=True):
        match = FENCE.match(line)
        if fence is None:
            if match:
                flush()
                fence = match.group("fence")
                pieces.append(line)
            else:
                prose.append(line)
        else:
            pieces.append(line)
            closes = (
                match is not None
                and match.group("fence")[0] == fence[0]
                and len(match.group("fence")) >= len(fence)
                and not line.strip(" \t\r\n" + fence[0])
            )
            if closes:
                fence = None
    flush()
    return "".join(pieces), count


def rewrite_math() -> None:
    """Make every staged chapter's math reach MathJax intact."""

    total = 0
    for document in SOURCE.rglob("*.md"):
        text = document.read_text(encoding="utf-8")
        rendered, count = convert_math(text)
        if count:
            document.write_text(rendered, encoding="utf-8")
            total += count
    print(f"converted {total} math spans for MathJax")


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
    rewrite_readme_links()
    rewrite_math()


def main() -> None:
    """Prepare the source tree and invoke mdBook unless only staging is wanted."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="assemble target/mdbook-src but do not invoke mdbook",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="enable print.html and write the PDF source book to target/book-pdf",
    )
    args = parser.parse_args()

    prepare()
    output = OUTPUT
    if args.print:
        configuration = STAGING / "book.toml"
        rendered = configuration.read_text(encoding="utf-8")
        disabled = "[output.html.print]\nenable = false"
        if disabled not in rendered:
            raise RuntimeError("book configuration has no disabled HTML print output")
        configuration.write_text(
            rendered.replace(disabled, "[output.html.print]\nenable = true"),
            encoding="utf-8",
        )
        output = PRINT_OUTPUT
    if not args.prepare_only:
        subprocess.run(
            [
                "mdbook",
                "build",
                str(STAGING),
                "--dest-dir",
                str(output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
