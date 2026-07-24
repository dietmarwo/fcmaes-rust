"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .io import load_run
from .render import render_run


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("run", type=Path, help="path to run.json")
    result.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory receiving generated SVG files",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    rendered = render_run(load_run(arguments.run), arguments.output_dir)
    for name, path in rendered.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
