#!/usr/bin/env python3
"""Generate search -> all-arm certificate collection -> strict acceptance; never launch work."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

from audit import AUDIT_LIMIT, load_protocol, parse, read
from make_campaign import ROOT
from make_search_campaign import generate as search_campaign


def generate(destination, julia, depot, binary, problem, protocol=None, audit_protocol=None,
             python=None, wall_seconds=600):
    protocol = Path(protocol or ROOT / "search-protocol.json").resolve()
    audit_protocol = Path(audit_protocol or ROOT / "audit-protocol.json").resolve()
    audit, _ = load_protocol(audit_protocol)
    arm_count = len(parse(read(protocol))["arms"])
    if not 6 <= arm_count <= 16:
        raise ValueError("audit requires 6..16 predeclared search arms")
    # Preserve a virtualenv interpreter path: resolving its symlink can select
    # the system Python and lose jsonschema installed in that environment.
    python = Path(python or sys.executable).absolute()
    if not python.is_file():
        raise ValueError("Python executable must exist and provide jsonschema")
    path = search_campaign(destination, julia, depot, binary, problem, protocol, wall_seconds)
    plan = json.loads(path.read_text())
    first = plan["steps"][0]
    inputs = copy.deepcopy(first["job"]["inputs"])
    sources = [(ROOT / "audit.py", "app/audit.py"), (audit_protocol, "audit-protocol.json")]
    sources += [(ROOT / "schemas" / name, "app/schemas/" + name) for name in (
        "cap-audit-0.1.schema.json", "cap-artifacts-0.1.schema.json", "cap-search-0.1.schema.json")]
    for source, staged in sources:
        relative = "inputs/" + staged
        target = path.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        data = target.read_bytes()
        inputs.append({"path": staged, "file": relative, "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest(), "executable": False})

    def binding(step, name, staged):
        return {"path": staged, "from": {"step": step, "kind": "output", "output": name}}

    inputs += [binding("search", name, file) for name, file in (
        ("study", "study.json"), ("candidate_bundle", "candidates.json"), ("evaluations", "evaluations.jsonl"))]
    arguments = ["--problem", "problem.json", "--study", "study.json", "--candidates", "candidates.json",
                 "--evaluations", "evaluations.jsonl", "--search-protocol", "protocol.json",
                 "--project", "project", "--protocol", "audit-protocol.json"]
    audit_wall = arm_count * (audit["verification_wall_seconds"] + audit["replay_wall_seconds"]) + 60

    def step(name, command, derived, outputs, wall):
        return {"id": name, "job": {"schema_version": "execution.command/0.1",
            "argv": [str(python), "app/audit.py", command, *arguments, *derived],
            "env": first["job"]["env"].copy(), "resources": {"cpu_slots": 1},
            "limits": {"wall_seconds": wall, "grace_seconds": 5}, "replay": "never",
            "inputs": copy.deepcopy(inputs), "outputs": [
                {"name": key, "path": "out/" + file, "max_bytes": maximum, "required": True}
                for key, file, maximum in outputs]}}

    collect = step("audit", "run", ["--julia", str(Path(julia).resolve()), "--output", "out"], [
        ("audit", "audit.json", AUDIT_LIMIT), ("events", "events.jsonl", 1048576),
        ("report", "report.md", 65536)], audit_wall)
    acceptance = step("accept", "accept", ["--audit", "audit.json", "--events", "events.jsonl",
                                            "--output", "out/acceptance.json"],
                      [("acceptance", "acceptance.json", 1048576)], 60)
    acceptance["job"]["inputs"] += [binding("audit", "audit", "audit.json"), binding("audit", "events", "events.jsonl")]
    plan["steps"] = [first, collect, acceptance]
    plan["name"] = "Validated numerics all-arm certificate audit " + problem
    total = wall_seconds + audit_wall + 60
    plan["budget"] = {"max_jobs": 3, "max_wall_seconds": total + 300,
                      "max_reserved_slot_seconds": total + 15}
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("destination", "julia", "depot", "binary"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol", "audit-protocol", "python"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--problem", choices=["scalar-cuberoot", "cahn-hilliard"], default="cahn-hilliard")
    parser.add_argument("--wall-seconds", type=int, default=600)
    args = parser.parse_args()
    print(generate(args.destination, args.julia, args.depot, args.binary, args.problem,
                   args.protocol, args.audit_protocol, args.python, args.wall_seconds))


if __name__ == "__main__":
    main()
