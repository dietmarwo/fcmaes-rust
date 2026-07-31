#!/usr/bin/env python3
"""Render every tutorial run manifest below a directory."""

from __future__ import annotations

import argparse
import filecmp
import subprocess
import sys
import tempfile
from pathlib import Path

from fcmaes_tutorial_plots import load_run, render_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    manifests = sorted(arguments.root.glob("*/results/**/run.json"))
    if not manifests:
        raise SystemExit(f"no run.json manifests found below {arguments.root}")
    stale = []
    for manifest in manifests:
        relative = manifest.relative_to(arguments.root)
        tutorial = arguments.root / relative.parts[0]
        if (tutorial / ".custom-renderer").is_file():
            continue
        run = load_run(manifest)
        run_name = "-".join(relative.parts[2:-1])
        output = tutorial / "images" / run_name
        if arguments.write:
            render_run(run, output)
            print(f"rendered {manifest}")
            continue
        with tempfile.TemporaryDirectory() as temporary:
            generated = render_run(run, temporary)
            for path in generated.values():
                checked_in = output / path.name
                if not checked_in.is_file() or not filecmp.cmp(path, checked_in, shallow=False):
                    stale.append(checked_in)
    if stale:
        print("missing or stale tutorial figures:")
        for path in stale:
            print(path)
        return 1
    custom_scripts = sorted(arguments.root.glob("*/plot_results.py"))
    action = "--write" if arguments.write else "--check"
    for script in custom_scripts:
        completed = subprocess.run(
            [sys.executable, str(script), action],
            cwd=script.parent,
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
