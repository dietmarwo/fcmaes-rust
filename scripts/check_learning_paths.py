#!/usr/bin/env python3
"""Check learning-path links against sibling checkouts or current default branches.

Only the selected page is traversed. Its targets are checked for existence,
anchors, default-branch source URLs and moved-page stubs, not recursively
crawled. Remote mode reads the other repositories from temporary shallow Git
checkouts of their default branches; it never changes or fetches into the
reader's checkouts. This repository is always read from the working tree, so
a pull request is checked against its own files. Requires Python 3.10+ and Git.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "fcmaes-rust"
OWNER = "dietmarwo"


def prose(text: str) -> str:
    """Remove fenced code while preserving line boundaries and inline links."""
    result = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            value = marker.group(1)
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            result.append("")
        else:
            result.append(line if fence is None else "")
    return "\n".join(result)


def links(text: str) -> list[str]:
    """Extract inline and explicit reference Markdown links outside fences."""
    source = prose(text)
    refs = {m.group(1).casefold(): m.group(2) for m in re.finditer(
        r"^\s{0,3}\[([^]]+)\]:\s*<?([^\s>]+)>?", source, re.MULTILINE)}
    found = re.findall(r"!?\[[^]\n]*\]\(\s*<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\s*\)", source)
    for match in re.finditer(r"!?\[([^]\n]+)\]\[([^]\n]*)\]", source):
        key = (match.group(2) or match.group(1)).casefold()
        if key not in refs:
            raise ValueError(f"undefined Markdown reference: {key}")
        found.append(refs[key])
    # Bare angle URLs are clickable Markdown too.
    found.extend(re.findall(r"<(https?://[^>\s]+)>", source))
    return list(dict.fromkeys(html.unescape(value) for value in found))


def anchors(text: str) -> set[str]:
    """GitHub-style Markdown heading slugs, duplicate suffixes and HTML IDs."""
    source = prose(text)
    found = set(re.findall(r"\b(?:id|name)=[\"']([^\"']+)[\"']", source))
    counts: dict[str, int] = {}
    heading_texts = []
    lines = source.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            heading_texts.append(match.group(1))
        elif index and re.match(r"^\s{0,3}(?:=+|-+)\s*$", line) and lines[index-1].strip():
            heading_texts.append(lines[index-1].strip())
    for value in heading_texts:
        value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
        value = html.unescape(re.sub(r"<[^>]+>", "", value)).lower()
        value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).replace(" ", "-")
        number = counts.get(value, 0)
        counts[value] = number + 1
        found.add(value if number == 0 else f"{value}-{number}")
    return found


def is_stub(text: str) -> bool:
    """Reject relocation notices in a page's opening, even without metadata."""
    opening = prose(text)[:1600].lower()
    opening = re.sub(r"\s+", " ", opening)
    return bool(re.search(
        r"(?:<!--\s*(?:redirect|stub|moved)\b|"
        r"(?:this (?:page|document|guide|tutorial|file)|the (?:tutorial|manual|guide|content|documentation))"
        r"\s+(?:has moved|moved to|is now (?:at|in|part of)|was moved|were merged)|"
        r"(?:permanent|compatibility|redirect) (?:stub|page))", opening))


def command(argv: list[str], timeout: int = 180) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        # Error identity is enough; avoid credential-bearing Git diagnostics.
        raise ValueError(f"Git command failed ({result.returncode}): {argv[1]}")
    return result.stdout.strip()


class Sources:
    def __init__(self, sources: Path, root: Path, remote: Path | None = None):
        self.sources = sources.resolve()
        self.root = root.resolve()
        self.remote = remote
        self.cache: dict[str, tuple[Path, str]] = {}

    def checkout(self, name: str) -> tuple[Path, str]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("invalid repository name")
        if name in self.cache:
            return self.cache[name]
        url = f"https://github.com/{OWNER}/{name}.git"
        if name == REPOSITORY:
            path = self.root
        elif self.remote is not None:
            path = self.remote / name
            command(["git", "clone", "--depth", "1", "--quiet", "--single-branch", url, str(path)])
        else:
            path = self.sources / name
        if not path.is_dir():
            raise ValueError(f"missing sibling checkout: {name}")
        if self.remote is not None and name == REPOSITORY:
            # A CI checkout of a pull request has no origin/HEAD; ask the remote.
            head = re.search(r"^ref: refs/heads/(\S+)\s+HEAD$",
                             command(["git", "ls-remote", "--symref", url, "HEAD"]), re.MULTILINE)
            if head is None:
                raise ValueError(f"default branch not reported: {name}")
            branch = head.group(1)
        else:
            branch = command(["git", "-C", str(path), "symbolic-ref", "--short",
                              "refs/remotes/origin/HEAD"]).removeprefix("origin/")
        self.cache[name] = (path, branch)
        return path, branch

    def check_file(self, name: str, relative: str, fragment: str, branch: str | None = None) -> None:
        checkout, default = self.checkout(name)
        if branch is not None and branch != default:
            raise ValueError(f"{name}: URL uses {branch}, default branch is {default}")
        target = (checkout / relative).resolve()
        if not target.is_relative_to(checkout.resolve()):
            raise ValueError("link escapes repository")
        if target.is_dir():
            target = next((target / candidate for candidate in ("README.md", "README.adoc", "index.md")
                           if (target / candidate).is_file()), target)
        if not target.is_file():
            raise ValueError(f"missing target: {name}/{relative}")
        if target.suffix.lower() not in {".md", ".markdown", ".adoc", ".html"}:
            if fragment:
                raise ValueError(f"unsupported fragment on non-document: {name}/{relative}")
            return
        text = target.read_text(encoding="utf-8")
        if is_stub(text):
            raise ValueError(f"moved-page stub: {name}/{relative}")
        if fragment and fragment not in anchors(text):
            raise ValueError(f"missing anchor: {name}/{relative}#{fragment}")

    def check(self, url: str, page: Path, page_text: str) -> None:
        parsed = urlsplit(url)
        fragment = unquote(parsed.fragment)
        if not parsed.scheme:
            if not parsed.path:
                if fragment not in anchors(page_text):
                    raise ValueError(f"missing page anchor: #{fragment}")
                return
            relative = (page.parent / unquote(parsed.path)).resolve()
            if not relative.is_relative_to(self.root):
                raise ValueError("relative link escapes repository")
            self.check_file(REPOSITORY, str(relative.relative_to(self.root)), fragment)
            return
        if parsed.scheme != "https":
            raise ValueError(f"unsupported link scheme: {parsed.scheme}")
        if parsed.netloc == "github.com":
            parts = unquote(parsed.path).strip("/").split("/")
            if len(parts) < 2 or parts[0] != OWNER:
                raise ValueError("GitHub target outside configured source owner")
            name = parts[1]
            if len(parts) == 2:
                self.check_file(name, "README.md", fragment)
            elif len(parts) >= 5 and parts[2] in {"blob", "tree"}:
                self.check_file(name, "/".join(parts[4:]), fragment, parts[3])
            else:
                raise ValueError("unsupported GitHub source URL")
            return
        # Future ordinary HTTPS targets are checked directly, including HTML IDs.
        request = Request(url, headers={"User-Agent": "fcmaes-learning-path-checker"})
        with urlopen(request, timeout=30) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("external target exceeds 4 MiB")
        text = data.decode("utf-8", errors="replace")
        if is_stub(text):
            raise ValueError("external target is a moved-page stub")
        if fragment and fragment not in anchors(text):
            raise ValueError(f"missing external anchor: #{fragment}")


def check_page(page: Path, sources: Sources) -> dict:
    text = page.read_text(encoding="utf-8")
    errors = []
    if len(text.splitlines()) > 250:
        errors.append({"link": "page", "error": "learning path exceeds 250 lines"})
    targets = links(text)
    for url in targets:
        try:
            sources.check(url, page, text)
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            errors.append({"link": url, "error": str(error)})
    return {"passed": not errors, "links_checked": len(targets), "errors": errors,
            "mode": "remote-default-branches" if sources.remote else "local-sibling-checkouts"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT.parent,
                        help="parent of sibling default-branch checkouts (default: repository parent)")
    parser.add_argument("--page", type=Path, default=ROOT / "docs/learning-paths.md")
    parser.add_argument("--remote", action="store_true",
                        help="read targets from fresh shallow default-branch checkouts")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="learning-path-links-") as temporary:
        sources = Sources(args.sources, ROOT, Path(temporary) if args.remote else None)
        report = check_page(args.page.resolve(), sources)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
