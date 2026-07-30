#!/usr/bin/env python3
"""Validate internal links and anchors in a rendered mdBook."""

from __future__ import annotations

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse


class DocumentParser(HTMLParser):
    """Collect linked resources and named anchors from one HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(
        self, _tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attributes)
        for name in ("id", "name"):
            if values.get(name):
                self.anchors.add(values[name] or "")
        for name in ("href", "src"):
            if values.get(name):
                self.references.append(values[name] or "")


def documents(book: Path) -> dict[Path, DocumentParser]:
    """Parse every rendered HTML document."""

    parsed: dict[Path, DocumentParser] = {}
    for document in sorted(book.rglob("*.html")):
        parser = DocumentParser()
        parser.feed(document.read_text(encoding="utf-8", errors="replace"))
        parsed[document.resolve()] = parser
    return parsed


def local_target(
    source: Path, raw: str, book: Path, site_prefix: str
) -> tuple[Path, str] | None:
    """Resolve one local URL to a rendered file and optional fragment."""

    if not raw or raw.startswith(("http://", "https://", "mailto:", "tel:")):
        return None
    if raw.startswith(("javascript:", "data:")):
        return None
    parsed = urlparse(raw)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    if path.startswith(site_prefix):
        target = book / path.removeprefix(site_prefix)
    elif path.startswith("/"):
        target = book / path.lstrip("/")
    else:
        base = f"file://{source.as_posix()}"
        target = Path(unquote(urlparse(urljoin(base, path)).path))
    if not path:
        target = source
    if raw.endswith("/") or target.is_dir():
        target /= "index.html"
    return target.resolve(), parsed.fragment


def validate(book: Path, site_prefix: str) -> list[str]:
    """Return rendered-link failures."""

    parsed = documents(book)
    failures: list[str] = []
    for source, document in parsed.items():
        for raw in document.references:
            resolved = local_target(source, raw, book, site_prefix)
            if resolved is None:
                continue
            target, fragment = resolved
            location = source.relative_to(book).as_posix()
            if not target.exists():
                failures.append(f"{location}: {raw} -> missing {target}")
                continue
            if fragment and target.suffix.lower() == ".html":
                target_document = parsed.get(target)
                if target_document is not None and fragment not in target_document.anchors:
                    failures.append(
                        f"{location}: {raw} -> missing anchor #{fragment}"
                    )
    return failures


def main() -> int:
    """Run the rendered-book audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--book",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "target" / "book",
        help="rendered mdBook directory",
    )
    parser.add_argument(
        "--site-prefix",
        default="/fcmaes-rust/",
        help="absolute URL prefix configured in book.toml",
    )
    arguments = parser.parse_args()
    book = arguments.book.resolve()
    if not book.is_dir():
        parser.error(f"rendered book does not exist: {book}")
    failures = validate(book, arguments.site_prefix)
    if failures:
        print("broken rendered documentation links:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    count = len(list(book.rglob("*.html")))
    print(f"all internal links and anchors resolve across {count} rendered pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
