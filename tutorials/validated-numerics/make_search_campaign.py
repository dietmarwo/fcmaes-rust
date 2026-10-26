#!/usr/bin/env python3
"""Freeze a search -> rigorous verification -> replay command campaign; never launch work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from make_campaign import ROOT, generate as reference_campaign


def generate(destination, julia, depot, binary, problem, protocol=None, wall_seconds=600):
    binary = Path(binary).resolve()
    protocol = Path(protocol or ROOT / "search-protocol.json").resolve()
    if not binary.is_file() or not protocol.is_file():
        raise ValueError("built search binary and protocol must exist")
    path = reference_campaign(destination, julia, depot, problem, wall_seconds)
    plan = json.loads(path.read_text())
    first = plan["steps"][0]
    first["id"] = "search"
    job = first["job"]
    job["argv"] = [str(binary), "--julia", str(Path(julia).resolve()), "--project", "project",
                   "--server", "project/bin/search_server.jl", "--problem", "problem.json",
                   "--protocol", "protocol.json", "--output", "out",
                   "--wall-seconds", str(wall_seconds - 10)]
    for source, staged in [(ROOT / "julia/src/CAPSearch.jl", "project/src/CAPSearch.jl"),
                           (ROOT / "julia/bin/search_server.jl", "project/bin/search_server.jl"),
                           (protocol, "protocol.json")]:
        relative = "inputs/" + staged
        target = path.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        data = target.read_bytes()
        job["inputs"].append({"path": staged, "file": relative, "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest(), "executable": False})
    for name, file, maximum in [("study", "study.json", 1048576),
                                ("candidate_bundle", "candidates.json", 1048576),
                                ("evaluations", "evaluations.jsonl", 8 * 1048576)]:
        job["outputs"].append({"name": name, "path": "out/" + file, "max_bytes": maximum, "required": True})
    for step in plan["steps"][1:]:
        for item in step["job"]["inputs"]:
            if item.get("from", {}).get("step") == "candidate":
                item["from"]["step"] = "search"
    plan["name"] = "Validated numerics search " + problem
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["destination", "julia", "depot", "binary"]:
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--problem", choices=["scalar-cuberoot", "cahn-hilliard"], default="cahn-hilliard")
    parser.add_argument("--wall-seconds", type=int, default=600)
    args = parser.parse_args()
    print(generate(args.destination, args.julia, args.depot, args.binary,
                   args.problem, args.protocol, args.wall_seconds))


if __name__ == "__main__":
    main()
