#!/usr/bin/env python3
"""Build a portable, bookmarked PDF from an mdBook print document."""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "pykep-core-rust": {
        "book_dir": "target/book",
        "output": "docs/pykep-rust-book.pdf",
        "slug": "dietmarwo/pykep-core-rust",
        "source_roots": ("", "docs"),
    },
    "fcmaes-rust": {
        "book_dir": "target/book-pdf",
        "output": "docs/fcmaes-rust-book.pdf",
        "slug": "dietmarwo/fcmaes-rust",
        "source_roots": ("",),
    },
}
PROJECT = PROJECTS.get(ROOT.name)
if PROJECT is None:
    raise SystemExit(f"unsupported repository: {ROOT}")

A_HREF = re.compile(
    r"(?P<prefix><a\b[^>]*?\bhref=)(?P<quote>[\"'])(?P<href>.*?)(?P=quote)",
    re.IGNORECASE,
)
ID = re.compile(r"\bid=[\"'](?P<id>[^\"']+)[\"']", re.IGNORECASE)
H1_ID = re.compile(
    r"<h1\b[^>]*?\bid=(?P<quote>[\"'])(?P<id>.*?)(?P=quote)",
    re.IGNORECASE,
)


def run(command: list[str]) -> str:
    """Run a command and return its standard output."""

    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def html_target(book_dir: Path, raw_path: str) -> Path | None:
    """Resolve an mdBook link to a rendered chapter, if it names one."""

    relative = PurePosixPath(raw_path or ".")
    if relative.is_absolute() or ".." in relative.parts:
        return None
    base = book_dir.joinpath(*relative.parts)
    candidates = [base]
    if base.suffix != ".html":
        candidates.append(base.with_suffix(".html"))
    candidates.append(base / "index.html")
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix == ".html":
            return candidate
    return None


def first_heading_id(document: Path) -> str:
    """Return the destination used for a rendered chapter's first heading."""

    match = H1_ID.search(document.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"chapter has no h1 destination: {document}")
    return html.unescape(match.group("id"))


def source_target(raw_path: str) -> Path | None:
    """Map a staged-book path back to a version-controlled source path."""

    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    for source_root in PROJECT["source_roots"]:
        candidate = ROOT.joinpath(source_root, *relative.parts).resolve()
        if candidate == ROOT or ROOT in candidate.parents:
            if candidate.exists():
                return candidate
    return None


def github_url(target: Path, fragment: str, query: str) -> str:
    """Create a portable repository link for a non-chapter artifact."""

    relative = target.relative_to(ROOT).as_posix()
    kind = "tree" if target.is_dir() else "blob"
    url = (
        f"https://github.com/{PROJECT['slug']}/{kind}/main/"
        f"{quote(relative, safe='/')}"
    )
    if query:
        url += f"?{query}"
    if fragment:
        url += f"#{quote(fragment, safe='-._~:')}"
    return url


def portable_print_html(book_dir: Path) -> tuple[str, int, int]:
    """Rewrite chapter and repository links for a single portable PDF."""

    print_path = book_dir / "print.html"
    if not print_path.is_file():
        raise RuntimeError(f"missing mdBook print document: {print_path}")
    document = print_path.read_text(encoding="utf-8")
    destinations = {html.unescape(value) for value in ID.findall(document)}
    chapter_links = 0
    repository_links = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal chapter_links, repository_links
        raw_href = html.unescape(match.group("href"))
        parsed = urlsplit(raw_href)
        if parsed.scheme or parsed.netloc or raw_href.startswith("#"):
            return match.group(0)

        raw_path = unquote(parsed.path)
        chapter = html_target(book_dir, raw_path)
        if chapter is not None:
            destination = unquote(parsed.fragment) or first_heading_id(chapter)
            if destination not in destinations:
                raise RuntimeError(
                    f"missing print destination #{destination} for {raw_href}"
                )
            rewritten = f"#{destination}"
            chapter_links += 1
        else:
            target = source_target(raw_path)
            if target is None:
                raise RuntimeError(f"cannot make local link portable: {raw_href}")
            rewritten = github_url(target, unquote(parsed.fragment), parsed.query)
            repository_links += 1
        return f"{match.group('prefix')}{match.group('quote')}{html.escape(rewritten, quote=True)}{match.group('quote')}"

    return A_HREF.sub(replace, document), chapter_links, repository_links


def verify(pdf: Path) -> None:
    """Check bookmarks, named destinations, and local-machine link leakage."""

    mutool = shutil.which("mutool")
    if mutool is None:
        raise RuntimeError("mutool is required to verify PDF navigation")
    root = run([mutool, "show", str(pdf), "trailer/Root"])
    outline = run([mutool, "show", str(pdf), "outline"])
    objects = run([mutool, "show", str(pdf), "grep"])
    destinations = run([mutool, "show", str(pdf), "trailer/Root/Dests"])

    if "/Outlines" not in root or not outline.strip():
        raise RuntimeError("PDF has no navigable outline/bookmark tree")
    if re.search(r"(?:file:/{2,}|file%3A)", objects, re.IGNORECASE):
        raise RuntimeError("PDF still contains a machine-local file link")

    defined = set(re.findall(r"^\s*/([^\s]+)\s+\[", destinations, re.MULTILINE))
    linked = set(re.findall(r"/Dest\s*/([^\s/<>()\[\]]+)", objects))
    missing = sorted(linked - defined)
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(f"PDF contains undefined internal destinations: {sample}")

    bookmarks = sum(1 for line in outline.splitlines() if line.lstrip().startswith("|"))
    print(
        f"verified {pdf}: {bookmarks} bookmarks, "
        f"{len(linked)} linked internal destinations, no local file URLs"
    )


def build(book_dir: Path, output: Path) -> None:
    """Print the rewritten book with Chrome's document outline enabled."""

    chrome = next(
        (
            executable
            for name in ("google-chrome", "chromium", "chromium-browser")
            if (executable := shutil.which(name)) is not None
        ),
        None,
    )
    if chrome is None:
        raise RuntimeError("Google Chrome or Chromium is required to build the PDF")

    rewritten, chapter_links, repository_links = portable_print_html(book_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    html_fd, html_name = tempfile.mkstemp(
        prefix=".book-pdf-", suffix=".html", dir=book_dir
    )
    pdf_fd, pdf_name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".pdf", dir=output.parent
    )
    os.close(pdf_fd)
    os.unlink(pdf_name)
    try:
        with os.fdopen(html_fd, "w", encoding="utf-8") as temporary_html:
            temporary_html.write(rewritten)
        subprocess.run(
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--generate-pdf-document-outline",
                f"--print-to-pdf={pdf_name}",
                Path(html_name).resolve().as_uri(),
            ],
            check=True,
        )
        os.replace(pdf_name, output)
    finally:
        Path(html_name).unlink(missing_ok=True)
        Path(pdf_name).unlink(missing_ok=True)

    print(
        f"rewrote {chapter_links} chapter links as PDF destinations and "
        f"{repository_links} artifact links as GitHub URLs"
    )
    verify(output)


def main() -> None:
    """Parse paths, build the PDF, and verify its navigation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--book-dir", type=Path, default=ROOT / PROJECT["book_dir"]
    )
    parser.add_argument("--output", type=Path, default=ROOT / PROJECT["output"])
    parser.add_argument(
        "--verify-only", action="store_true", help="verify the existing PDF"
    )
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output.resolve())
    else:
        build(args.book_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
