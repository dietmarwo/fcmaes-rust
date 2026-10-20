#!/usr/bin/env python3
"""Prints the root of Cargo's checkout of github.com/dietmarwo/optimization-jobs
at the revision this workspace pins, which holds the shared job API guide,
schemas and test scripts:

    SHARED=$(python3 scripts/job-api/shared_root.py)
    JOB_API_TARGET=$PWD/target/release python3 "$SHARED/scripts/job-api/test_integration.py"
"""
import json
import subprocess
from pathlib import Path


def shared_root(workspace=Path(__file__).resolve().parents[2]):
    """The checkout that holds the pinned optimization-runner package."""
    metadata = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],
        cwd=workspace, capture_output=True, text=True, check=True).stdout)
    manifest = next(p["manifest_path"] for p in metadata["packages"] if p["name"] == "optimization-runner")
    return Path(manifest).parents[2]


if __name__ == "__main__":
    print(shared_root())
