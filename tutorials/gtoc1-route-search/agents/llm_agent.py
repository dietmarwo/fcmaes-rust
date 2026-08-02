#!/usr/bin/env python3
"""Provider adapter for the versioned GTOC1 MGA route protocol.

Anthropic-compatible providers return the campaign's route schema directly.
OpenAI-compatible providers, including a local llama.cpp server, select one
opaque ID from a deterministic menu of grammar-valid unseen route variants.
The latter mirrors the oscillator tutorial: malformed or duplicate local-model
output cannot consume an inner MGA optimization budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


PROTOCOL = "gtoc1-mga-route-proposal-v2"
ASSISTED_PROTOCOL = "gemma4-assisted-v1"
BODY_NAME = {2: "Venus", 3: "Earth", 5: "Jupiter", 6: "Saturn", 10: "TW229"}
BODY_SYMBOL = {2: "V", 3: "E", 5: "J", 6: "S", 10: "A"}
INTERIOR_BODIES = (2, 3, 5, 6)


@dataclass(frozen=True)
class Experience:
    """Declared offline evidence used only by the assisted menu policy."""

    records: tuple[dict[str, Any], ...] = ()
    digest: str = "none"
    sources: tuple[str, ...] = ()


def first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    while start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    raise ValueError("provider response contains no JSON object")


def response_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        ]
        if texts:
            return "".join(texts)
    raise ValueError("provider message has no textual content")


def structure_key(bodies: Iterable[int]) -> str:
    return "-".join(str(body) for body in bodies)


def variant_key(bodies: Iterable[int], clockwise: Iterable[bool]) -> str:
    return structure_key(bodies) + "|" + "".join("1" if bit else "0" for bit in clockwise)


def canonical_clockwise(bodies: Iterable[int]) -> tuple[bool, ...]:
    values = tuple(bodies)
    return tuple(pair in {(6, 5), (5, 10)} for pair in zip(values, values[1:]))


def parse_variant_key(value: Any) -> tuple[tuple[int, ...], tuple[bool, ...]] | None:
    if not isinstance(value, str) or "|" not in value:
        return None
    body_text, bits = value.split("|", 1)
    try:
        bodies = tuple(int(field) for field in body_text.split("-"))
    except ValueError:
        return None
    if len(bits) + 1 != len(bodies) or re.fullmatch(r"[01]+", bits) is None:
        return None
    return bodies, tuple(bit == "1" for bit in bits)


def valid_variant(
    bodies: tuple[int, ...], clockwise: tuple[bool, ...], constraints: dict[str, Any]
) -> bool:
    maximum = int(constraints["maximum_encounters"])
    if not 3 <= len(bodies) <= maximum or len(clockwise) + 1 != len(bodies):
        return False
    if bodies[0] != 3 or bodies[-1] != 10:
        return False
    if any(body not in INTERIOR_BODIES for body in bodies[1:-1]):
        return False
    maximum_run = int(constraints["maximum_same_body_run"])
    run = 1
    for left, right in zip(bodies, bodies[1:]):
        run = run + 1 if left == right else 1
        if run > maximum_run:
            return False
    return sum(body in {5, 6} for body in bodies[1:-1]) <= int(
        constraints["maximum_outer_encounters"]
    )


def stable_rng(request: dict[str, Any], namespace: str) -> random.Random:
    material = (
        f"{PROTOCOL}:{request.get('proposal_attempt', 0)}:{namespace}"
    ).encode("ascii")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


def compact_route(bodies: Iterable[int]) -> str:
    return "".join(BODY_SYMBOL.get(body, "?") for body in bodies)


def body_edit_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_body in enumerate(left, start=1):
        current = [row]
        for column, right_body in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_body != right_body),
                )
            )
        previous = current
    return previous[-1]


def result_record(value: Any, origin: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    structure = value.get("structure") or {}
    bodies_value = structure.get("bodies") if isinstance(structure, dict) else None
    if not isinstance(bodies_value, list):
        parsed = parse_variant_key(value.get("variant_key") or value.get("best_variant_key"))
        bodies = parsed[0] if parsed is not None else ()
    else:
        try:
            bodies = tuple(int(body) for body in bodies_value)
        except (TypeError, ValueError):
            return None
    if len(bodies) < 3:
        return None
    clockwise = canonical_clockwise(bodies)
    l0 = value.get("l0") or {}
    score = (
        l0.get("estimated_score")
        if isinstance(l0, dict) and "estimated_score" in l0
        else value.get("mga_score", value.get("best_mga_score"))
    )
    worker_seconds = (
        l0.get("worker_seconds")
        if isinstance(l0, dict) and "worker_seconds" in l0
        else value.get("worker_seconds", value.get("mean_worker_seconds", 0.0))
    )
    try:
        score = float(score)
        worker_seconds = float(worker_seconds or 0.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not math.isfinite(worker_seconds):
        return None
    return {
        "bodies": bodies,
        "clockwise": clockwise,
        "score": score,
        "worker_seconds": worker_seconds,
        "origin": origin,
    }


def load_experience(paths: Any) -> Experience:
    if paths in (None, []):
        return Experience()
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) and path for path in paths):
        raise ValueError("experience_archive_paths must be a non-empty string array")
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    sources: list[str] = []
    for value in paths:
        path = Path(value)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        decoded = json.loads(data)
        rows = decoded.get("results") if isinstance(decoded, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"experience archive {path} has no results array")
        source = path.name
        sources.append(source)
        records.extend(
            record
            for row in rows
            if (record := result_record(row, source)) is not None
        )
    if not records:
        raise ValueError("experience archives contain no finite MGA results")
    return Experience(tuple(records), digest.hexdigest(), tuple(sources))


def candidate_allowed(
    bodies: tuple[int, ...],
    clockwise: tuple[bool, ...],
    request: dict[str, Any],
    seen: set[str],
) -> bool:
    constraints = request["constraints"]
    if not valid_variant(bodies, clockwise, constraints):
        return False
    key = variant_key(bodies, clockwise)
    if key in seen:
        return False
    counts = request.get("archive", {}).get("structure_variant_counts", {})
    count = int(counts.get(structure_key(bodies), 0)) if isinstance(counts, dict) else 0
    return count < int(constraints["maximum_variants_per_structure"])


def random_variant(
    request: dict[str, Any], rng: random.Random
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    constraints = request["constraints"]
    maximum = int(constraints["maximum_encounters"])
    return random_variant_in_range(request, rng, 3, maximum)


def random_variant_in_range(
    request: dict[str, Any],
    rng: random.Random,
    minimum: int,
    maximum: int,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    constraints = request["constraints"]
    grammar_maximum = int(constraints["maximum_encounters"])
    minimum = max(3, minimum)
    maximum = min(grammar_maximum, maximum)
    if minimum > maximum:
        raise ValueError("candidate-menu length range is outside the route grammar")
    for _ in range(2000):
        length = rng.randint(minimum, maximum)
        bodies = (3,) + tuple(rng.choice(INTERIOR_BODIES) for _ in range(length - 2)) + (10,)
        clockwise = canonical_clockwise(bodies)
        if valid_variant(bodies, clockwise, constraints):
            return bodies, clockwise
    raise ValueError("candidate-menu sampler exhausted")


def mutate_variant(
    parent: tuple[tuple[int, ...], tuple[bool, ...]],
    request: dict[str, Any],
    rng: random.Random,
) -> tuple[tuple[int, ...], tuple[bool, ...]] | None:
    constraints = request["constraints"]
    maximum = int(constraints["maximum_encounters"])
    for _ in range(128):
        bodies = list(parent[0])
        bits = list(parent[1])
        operator = rng.randrange(5)
        if operator == 0 and len(bodies) > 2:
            bodies[rng.randrange(1, len(bodies) - 1)] = rng.choice(INTERIOR_BODIES)
        elif operator == 1 and len(bodies) < maximum:
            index = rng.randrange(1, len(bodies))
            bodies.insert(index, rng.choice(INTERIOR_BODIES))
            bits.insert(index, bits[max(index - 1, 0)])
        elif operator == 2 and len(bodies) > 3:
            index = rng.randrange(1, len(bodies) - 1)
            bodies.pop(index)
            bits.pop(index)
        elif operator == 3 and len(bodies) > 3:
            index = rng.randrange(1, len(bodies) - 2)
            bodies[index], bodies[index + 1] = bodies[index + 1], bodies[index]
        elif operator == 4:
            outer = [index for index, body in enumerate(bodies) if body in {5, 6}]
            if not outer:
                continue
            for index in outer:
                bodies[index] = rng.choice((5, 6))
        candidate_bodies = tuple(bodies)
        candidate = candidate_bodies, canonical_clockwise(candidate_bodies)
        if candidate != parent and valid_variant(*candidate, constraints):
            return candidate
    return None


def seen_variants(request: dict[str, Any]) -> set[str]:
    archive = request.get("archive", {})
    values = archive.get("already_evaluated_variants", []) if isinstance(archive, dict) else []
    return {value for value in values if parse_variant_key(value) is not None}


def elite_variants(request: dict[str, Any]) -> list[tuple[tuple[int, ...], tuple[bool, ...]]]:
    archive = request.get("archive", {})
    top = archive.get("top", []) if isinstance(archive, dict) else []
    elites = []
    for item in top:
        if isinstance(item, dict):
            parsed = parse_variant_key(item.get("variant_key"))
            if parsed is not None:
                elites.append(parsed)
    return elites[:8]


def live_elite_records(request: dict[str, Any]) -> list[dict[str, Any]]:
    archive = request.get("archive", {})
    values = []
    if isinstance(archive, dict):
        values.extend(archive.get("top", []))
        values.extend(archive.get("length_evidence", []))
    return [
        record
        for value in values
        if (record := result_record(value, "live-archive")) is not None
    ]


def assisted_elite_records(
    request: dict[str, Any], experience: Experience
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in [*experience.records, *live_elite_records(request)]:
        key = structure_key(record["bodies"])
        if key not in by_key or record["score"] > by_key[key]["score"]:
            by_key[key] = record
    ranked = sorted(by_key.values(), key=lambda record: record["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    by_length: dict[int, list[dict[str, Any]]] = {}
    for record in ranked:
        by_length.setdefault(len(record["bodies"]), []).append(record)
    for length in sorted(by_length):
        for record in by_length[length][:2]:
            key = structure_key(record["bodies"])
            if key not in selected_keys:
                selected.append(record)
                selected_keys.add(key)
    for record in ranked:
        if len(selected) >= 32:
            break
        key = structure_key(record["bodies"])
        if key not in selected_keys:
            selected.append(record)
            selected_keys.add(key)
    return selected


def assisted_length_bands(maximum: int) -> list[tuple[str, int, int, int]]:
    bands = [
        ("3-6", 3, 6, 1),
        ("7-9", 7, 9, 2),
        ("10-11", 10, 11, 2),
        ("12-14", 12, 14, 1),
    ]
    return [
        (name, lower, min(upper, maximum), weight)
        for name, lower, upper, weight in bands
        if lower <= maximum
    ]


def allocate_band_quotas(
    bands: list[tuple[str, int, int, int]], menu_size: int
) -> list[tuple[str, int, int, int]]:
    total_weight = sum(band[3] for band in bands)
    quotas = [menu_size * band[3] // total_weight for band in bands]
    for index in range(menu_size - sum(quotas)):
        quotas[index % len(quotas)] += 1
    return [
        (name, lower, upper, quota)
        for (name, lower, upper, _weight), quota in zip(bands, quotas)
        if quota > 0
    ]


def terminal_motif(bodies: tuple[int, ...]) -> str:
    return "-".join(BODY_NAME[body] for body in bodies[-min(4, len(bodies)) :])


def optimizer_cost_band(encounters: int) -> str:
    if encounters <= 6:
        return "low"
    if encounters <= 9:
        return "medium"
    if encounters <= 11:
        return "high"
    return "very-high"


def build_cold_candidate_menu(
    request: dict[str, Any], menu_size: int
) -> list[dict[str, Any]]:
    if not 1 <= menu_size <= 256:
        raise ValueError("candidate_menu_size must be in 1..=256")
    seen = seen_variants(request)
    selected: dict[str, tuple[tuple[int, ...], tuple[bool, ...], str]] = {}
    mutation_target = menu_size // 3
    rng = stable_rng(request, "elite-mutations")
    elites = elite_variants(request)
    mutation_attempts = 0
    while elites and len(selected) < mutation_target and mutation_attempts < menu_size * 100:
        parent = elites[mutation_attempts % len(elites)]
        mutation_attempts += 1
        candidate = mutate_variant(parent, request, rng)
        if candidate is None:
            break
        bodies, clockwise = candidate
        key = variant_key(bodies, clockwise)
        if candidate_allowed(bodies, clockwise, request, seen):
            selected.setdefault(key, (bodies, clockwise, "elite-mutation"))

    rng = stable_rng(request, "random-immigrants")
    attempts = 0
    while len(selected) < menu_size and attempts < menu_size * 500:
        attempts += 1
        bodies, clockwise = random_variant(request, rng)
        key = variant_key(bodies, clockwise)
        if candidate_allowed(bodies, clockwise, request, seen):
            selected.setdefault(key, (bodies, clockwise, "random-immigrant"))
    if not selected:
        raise ValueError("no unseen route variant is available for the candidate menu")

    order_rng = stable_rng(request, "menu-order")
    rows = list(selected.values())
    order_rng.shuffle(rows)
    menu = []
    for index, (bodies, clockwise, source) in enumerate(rows):
        menu.append(
            {
                "candidate_id": f"c{index:03d}",
                "bodies": [BODY_NAME[body] for body in bodies],
                "source": source,
                "encounters": len(bodies),
                "outer_encounters": sum(body in {5, 6} for body in bodies),
            }
        )
    return menu


def build_assisted_candidate_menu(
    request: dict[str, Any], menu_size: int, experience: Experience
) -> list[dict[str, Any]]:
    if not 4 <= menu_size <= 128:
        raise ValueError("assisted candidate_menu_size must be in 4..=128")
    maximum = int(request["constraints"]["maximum_encounters"])
    bands = assisted_length_bands(maximum)
    if not bands:
        raise ValueError("route grammar has no assisted length band")
    if request.get("phase") == "bootstrap":
        # Deliberately gather evidence across the full dimensional range before
        # score feedback is exposed. Rejections do not advance this schedule.
        order = ["7-9", "10-11", "3-6", "12-14"]
        available = {band[0]: band for band in bands}
        schedule = [available[name] for name in order if name in available]
        band = schedule[int(request.get("accepted_candidates", 0)) % len(schedule)]
        quotas = [(band[0], band[1], band[2], menu_size)]
    else:
        quotas = allocate_band_quotas(bands, menu_size)

    seen = seen_variants(request)
    seen.update(variant_key(record["bodies"], record["clockwise"]) for record in experience.records)
    parents = assisted_elite_records(request, experience)
    selected: dict[str, dict[str, Any]] = {}

    for name, lower, upper, quota in quotas:
        mutation_target = quota // 2 if parents else 0
        rng = stable_rng(request, f"{ASSISTED_PROTOCOL}:{name}:elite-mutations")
        attempts = 0
        mutations = 0
        while mutations < mutation_target and attempts < max(500, quota * 300):
            parent = parents[attempts % len(parents)]
            attempts += 1
            candidate = mutate_variant(
                (parent["bodies"], parent["clockwise"]), request, rng
            )
            if candidate is None:
                continue
            bodies, clockwise = candidate
            key = variant_key(bodies, clockwise)
            if (
                lower <= len(bodies) <= upper
                and key not in selected
                and candidate_allowed(bodies, clockwise, request, seen)
            ):
                selected[key] = {
                    "bodies": bodies,
                    "clockwise": clockwise,
                    "source": f"{parent['origin']}-mutation",
                    "parent": parent,
                    "length_band": name,
                }
                mutations += 1

        rng = stable_rng(request, f"{ASSISTED_PROTOCOL}:{name}:random-immigrants")
        band_count = sum(row["length_band"] == name for row in selected.values())
        attempts = 0
        while band_count < quota and attempts < quota * 1000:
            attempts += 1
            bodies, clockwise = random_variant_in_range(request, rng, lower, upper)
            key = variant_key(bodies, clockwise)
            if key in selected or not candidate_allowed(bodies, clockwise, request, seen):
                continue
            selected[key] = {
                "bodies": bodies,
                "clockwise": clockwise,
                "source": "stratified-random-immigrant",
                "parent": None,
                "length_band": name,
            }
            band_count += 1
        if band_count < quota:
            raise ValueError(f"assisted candidate sampler exhausted length band {name}")

    order_rng = stable_rng(request, f"{ASSISTED_PROTOCOL}:menu-order")
    rows = list(selected.values())
    order_rng.shuffle(rows)
    menu = []
    for index, row in enumerate(rows):
        bodies = row["bodies"]
        parent = row["parent"]
        nearest_distance = (
            min(body_edit_distance(bodies, value["bodies"]) for value in parents)
            if parents
            else None
        )
        menu.append(
            {
                "candidate_id": f"c{index:03d}",
                "bodies": [BODY_NAME[body] for body in bodies],
                "encounters": len(bodies),
                "length_band": row["length_band"],
                "outer_encounters": sum(body in {5, 6} for body in bodies),
                "terminal_motif": terminal_motif(bodies),
                "optimizer_cost_band": optimizer_cost_band(len(bodies)),
                "source": row["source"],
                "parent_route": compact_route(parent["bodies"]) if parent else None,
                "parent_mga_score": parent["score"] if parent else None,
                "nearest_elite_edit_distance": nearest_distance,
            }
        )
    return menu


def build_candidate_menu(
    request: dict[str, Any],
    menu_size: int,
    policy: str = "cold-v1",
    experience: Experience | None = None,
) -> list[dict[str, Any]]:
    if policy == "cold-v1":
        return build_cold_candidate_menu(request, menu_size)
    if policy == ASSISTED_PROTOCOL:
        return build_assisted_candidate_menu(request, menu_size, experience or Experience())
    raise ValueError(f"unsupported menu_policy {policy!r}")


def menu_schema(
    menu: list[dict[str, Any]], policy: str = "cold-v1", ranked_candidates: int = 1
) -> dict[str, Any]:
    identifiers = [candidate["candidate_id"] for candidate in menu]
    if policy == ASSISTED_PROTOCOL:
        return {
            "type": "object",
            "properties": {
                "ranked_candidate_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": identifiers},
                    "minItems": ranked_candidates,
                    "maxItems": ranked_candidates,
                }
            },
            "required": ["ranked_candidate_ids"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
                "enum": identifiers,
            }
        },
        "required": ["candidate_id"],
        "additionalProperties": False,
    }


def proposal_from_menu(
    value: dict[str, Any],
    menu: list[dict[str, Any]],
    policy: str = "cold-v1",
    ranked_candidates: int = 1,
    experience_digest: str = "none",
) -> dict[str, Any]:
    by_id = {candidate["candidate_id"]: candidate for candidate in menu}
    if policy == ASSISTED_PROTOCOL:
        identifiers = value.get("ranked_candidate_ids")
        if (
            not isinstance(identifiers, list)
            or len(identifiers) != ranked_candidates
            or len(set(identifiers)) != len(identifiers)
            or any(identifier not in by_id for identifier in identifiers)
        ):
            raise ValueError("model did not rank the required distinct menu candidates")
    else:
        identifiers = [value.get("candidate_id")]
        if identifiers[0] not in by_id:
            raise ValueError("model did not select a candidate from the supplied menu")
    return {
        "candidates": [
            {
                "bodies": by_id[candidate_id]["bodies"],
                "rationale": (
                    f"{policy} rank {rank} menu selection {candidate_id}; "
                    f"source={by_id[candidate_id]['source']}; "
                    f"experience_sha256={experience_digest[:16]}"
                    if policy == ASSISTED_PROTOCOL
                    else f"llama.cpp menu selection {candidate_id}"
                ),
            }
            for rank, candidate_id in enumerate(identifiers, start=1)
        ]
    }


def unique_experience_records(experience: Experience) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in experience.records:
        key = structure_key(record["bodies"])
        if key not in by_key or record["score"] > by_key[key]["score"]:
            by_key[key] = record
    return list(by_key.values())


def experience_evidence(experience: Experience) -> dict[str, Any]:
    records = unique_experience_records(experience)
    groups: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(len(record["bodies"]), []).append(record)
    lengths = []
    for encounters, rows in sorted(groups.items()):
        ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
        top = ranked[:5]
        lengths.append(
            {
                "encounters": encounters,
                "evaluated": len(ranked),
                "best_mga_score": ranked[0]["score"],
                "top5_mean_mga_score": sum(row["score"] for row in top) / len(top),
                "mean_mga_score": sum(row["score"] for row in ranked) / len(ranked),
                "mean_worker_seconds": sum(row["worker_seconds"] for row in ranked)
                / len(ranked),
                "best_route": compact_route(ranked[0]["bodies"]),
            }
        )
    top_routes = sorted(records, key=lambda row: row["score"], reverse=True)[:12]
    return {
        "sources": list(experience.sources),
        "sha256": experience.digest,
        "unique_routes": len(records),
        "length_evidence": lengths,
        "top_routes": [
            {
                "route": compact_route(row["bodies"]),
                "encounters": len(row["bodies"]),
                "mga_score": row["score"],
                "worker_seconds": row["worker_seconds"],
            }
            for row in top_routes
        ],
    }


def assisted_prompt_request(
    request: dict[str, Any], menu: list[dict[str, Any]], experience: Experience
) -> dict[str, Any]:
    archive = request.get("archive") or {}
    retry_requirement = None
    marker = "Specific retry requirement:"
    if marker in request.get("user", ""):
        retry_requirement = request["user"].rsplit(marker, 1)[1].strip()[:300]
    live_evidence = {
        key: archive.get(key)
        for key in ("accepted", "length_counts", "length_evidence", "portfolio", "top")
        if key in archive
    }
    portfolio_size = (archive.get("portfolio") or {}).get("target_size", 20)
    return {
        "protocol": ASSISTED_PROTOCOL,
        "phase": request.get("phase"),
        "proposal_attempt": request.get("proposal_attempt"),
        "accepted_candidates": request.get("accepted_candidates"),
        "accepted_candidates_target": request.get("accepted_candidates_target"),
        "objective": {
            "primary": f"increase the sum of the best {portfolio_size} MGA scores",
            "secondary": "prefer lower optimizer cost when expected MGA quality is comparable",
            "warning": "encounter count has no score bonus; extra encounters increase numerical cost",
        },
        "constraints": request["constraints"],
        "live_evidence": live_evidence,
        "declared_prior_evidence": experience_evidence(experience),
        "retry_requirement": retry_requirement,
        "candidate_menu": menu,
    }


def anthropic_user_prompt(request: dict[str, Any]) -> str:
    constraints = json.dumps(request["constraints"], separators=(",", ":"))
    schema = json.dumps(request["response_schema"], separators=(",", ":"))
    return (
        f"{request['user']}\n\nProtocol-owned constraints:\n{constraints}\n"
        "Return only one JSON object matching this schema:\n" + schema
    )


def is_loopback_url(url: str) -> bool:
    return urllib.parse.urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}


def build_provider_call(
    request: dict[str, Any], api_key: str | None
) -> tuple[str, dict[str, Any], dict[str, str], str, str, dict[str, Any] | None]:
    adapter = request["adapter"]
    provider = adapter.get("provider")
    model = adapter.get("model")
    base_url = adapter.get("base_url")
    if not isinstance(model, str) or not model:
        raise ValueError("adapter.model must be configured")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("adapter.base_url must be configured")
    if not api_key and not is_loopback_url(base_url):
        raise ValueError("remote provider requires the configured credential")

    options = dict(adapter.get("provider_options") or {})
    if not isinstance(options, dict):
        raise ValueError("adapter.provider_options must be an object")
    protected = {"model", "messages", "max_tokens", "stream", "system"}
    if protected.intersection(options):
        raise ValueError("provider_options may not replace protocol-owned request fields")
    options.pop("http_timeout_seconds", None)

    if provider == "openai-compatible":
        if "response_format" in options:
            raise ValueError("provider_options may not replace response_format")
        menu_size = int(options.pop("candidate_menu_size", 96))
        policy = str(options.pop("menu_policy", "cold-v1"))
        ranked_candidates = int(options.pop("ranked_candidates", 1))
        experience_paths = options.pop("experience_archive_paths", [])
        expected_experience_digest = options.pop("experience_sha256", None)
        experience = load_experience(experience_paths)
        if policy == "cold-v1" and (
            ranked_candidates != 1 or experience.records or expected_experience_digest is not None
        ):
            raise ValueError(
                "ranked candidates and experience archives require menu_policy gemma4-assisted-v1"
            )
        if policy == ASSISTED_PROTOCOL and not experience.records:
            raise ValueError("gemma4-assisted-v1 requires declared experience archives")
        if policy == ASSISTED_PROTOCOL and expected_experience_digest != experience.digest:
            raise ValueError(
                "experience archive SHA-256 does not match provider_options.experience_sha256"
            )
        if not 1 <= ranked_candidates <= min(8, menu_size):
            raise ValueError("ranked_candidates must be in 1..=min(8,candidate_menu_size)")
        menu = build_candidate_menu(request, menu_size, policy, experience)
        if policy == ASSISTED_PROTOCOL:
            prompt = (
                f"Rank exactly {ranked_candidates} distinct candidate IDs from best to worst. "
                "Every listed order is grammar-valid and unseen in this campaign and the "
                "declared prior archives. Do not equate more encounters with higher quality. "
                "Use the score, cost, portfolio, diversity, parent, and terminal-motif evidence. "
                f'Return only {{"ranked_candidate_ids":["cNNN",...]}}.\n'
                + json.dumps(
                    assisted_prompt_request(request, menu, experience),
                    separators=(",", ":"),
                )
            )
        else:
            prompt_request = dict(request)
            prompt_request["candidate_menu"] = menu
            prompt = (
                "Choose exactly one candidate_id from candidate_menu. Every listed route "
                "is grammar-valid and has not been evaluated as a body order. Rust assigns "
                "the historical pair-dependent direction pattern. Higher MGA score is better; "
                "seek both quality and route diversity. "
                'Return only {"candidate_id":"cNNN"}.\n'
                + json.dumps(prompt_request, separators=(",", ":"))
            )
        menu_context = {
            "rows": menu,
            "policy": policy,
            "ranked_candidates": ranked_candidates,
            "experience_digest": experience.digest,
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": request["system"]},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": adapter["maximum_tokens"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "gtoc1_route_menu_selection",
                    "strict": True,
                    "schema": menu_schema(menu, policy, ranked_candidates),
                },
            },
            **options,
        }
        base = base_url.rstrip("/")
        endpoint = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    elif provider == "anthropic-compatible":
        menu_context = None
        payload = {
            "model": model,
            "system": request["system"],
            "messages": [{"role": "user", "content": [{"type": "text", "text": anthropic_user_prompt(request)}]}],
            "max_tokens": adapter["maximum_tokens"],
            "stream": True,
            **options,
        }
        base = base_url.rstrip("/")
        endpoint = f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
            "X-Api-Key": str(api_key),
            "Content-Type": "application/json",
        }
    else:
        raise ValueError("adapter.provider must be 'openai-compatible' or 'anthropic-compatible'")
    return endpoint, payload, headers, provider, model, menu_context


def sse_json_events(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                data = "\n".join(data_lines)
                data_lines.clear()
                if data != "[DONE]":
                    value = json.loads(data)
                    if not isinstance(value, dict):
                        raise ValueError("provider SSE data is not a JSON object")
                    yield value
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("provider SSE data is not a JSON object")
            yield value


def parse_anthropic_stream(lines: Iterable[bytes], configured_model: str) -> dict[str, Any]:
    model = configured_model
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    message_started = False
    message_stopped = False
    stop_reason = None
    for event in sse_json_events(lines):
        event_type = event.get("type")
        if event_type == "error":
            error = event.get("error") or {}
            raise RuntimeError(
                f"provider stream error {error.get('type', 'unknown')}: {error.get('message', 'unknown')}"
            )
        if event_type == "message_start":
            message_started = True
            message = event.get("message") or {}
            model = message.get("model", model)
            usage.update(message.get("usage") or {})
        elif event_type == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "text" and block.get("text"):
                text_parts.append(block["text"])
        elif event_type == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                text_parts.append(delta["text"])
        elif event_type == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason", stop_reason)
            usage.update(event.get("usage") or {})
        elif event_type == "message_stop":
            message_stopped = True
    if not message_started:
        raise ValueError("provider SSE stream contains no message_start")
    if not message_stopped:
        raise ValueError("provider SSE stream ended before message_stop")
    if not text_parts:
        raise ValueError(
            "provider SSE stream contains no text "
            f"(stop_reason={stop_reason}, output_tokens={usage.get('output_tokens')})"
        )
    return {"model": model, "content": [{"type": "text", "text": "".join(text_parts)}], "usage": usage}


def parse_provider_response(
    provider: str,
    provider_response: dict[str, Any],
    configured_model: str,
    menu_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if provider == "openai-compatible":
        choices = provider_response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("provider response contains no choices")
        text = response_text(choices[0]["message"])
        usage = provider_response.get("usage") or {}
        if menu_context is None:
            raise ValueError("OpenAI-compatible response has no candidate menu")
        candidate_response = proposal_from_menu(
            first_json_object(text),
            menu_context["rows"],
            menu_context["policy"],
            menu_context["ranked_candidates"],
            menu_context["experience_digest"],
        )
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        cache_read_tokens = None
        cache_write_tokens = None
    else:
        text = response_text(provider_response)
        usage = provider_response.get("usage") or {}
        candidate_response = first_json_object(text)
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cache_read_tokens = usage.get("cache_read_input_tokens")
        cache_write_tokens = usage.get("cache_creation_input_tokens")
    candidate_response["usage"] = {
        "provider": provider,
        "model": provider_response.get("model", configured_model),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
    }
    return candidate_response


def main() -> None:
    request = json.load(sys.stdin)
    adapter = request["adapter"]
    key_name = adapter.get("api_key_env")
    api_key = os.environ.get(key_name) if isinstance(key_name, str) and key_name else None
    endpoint, payload, headers, provider, model, menu_context = build_provider_call(request, api_key)
    http_request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout = int((adapter.get("provider_options") or {}).get("http_timeout_seconds", 300))
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            provider_response = (
                parse_anthropic_stream(response, model)
                if provider == "anthropic-compatible"
                else json.load(response)
            )
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"provider HTTP {error.code}: {detail}") from error
    candidate_response = parse_provider_response(
        provider, provider_response, model, menu_context
    )
    json.dump(candidate_response, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"llm_agent: {error}", file=sys.stderr)
        raise SystemExit(1) from error
