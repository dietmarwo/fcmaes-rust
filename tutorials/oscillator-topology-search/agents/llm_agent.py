#!/usr/bin/env python3
"""Anthropic/OpenAI-compatible topology-proposal adapter.

The Anthropic path deliberately uses the same dual-authenticated SSE transport
as the GTOC1 route-search tutorial. Both providers constrain the public answer
to a versioned contract. Local models choose from a deterministic menu of
valid unseen candidates. Thinking deltas are never mixed into the topology
returned to the Rust campaign driver.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator


PROTOCOL = "oscillator-topology-proposal-v4"
TOPOLOGY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {"type": "integer", "enum": [0, 1, 2]},
            "minItems": 9,
            "maxItems": 9,
        }
    },
    "required": ["edges"],
    "additionalProperties": False,
}
TOPOLOGY_TOOL: dict[str, Any] = {
    "name": "propose_topology",
    "description": "Return one grammar-valid signed three-gene topology.",
    "input_schema": TOPOLOGY_SCHEMA,
}
EDGE_INDEX = [
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
]


def rust_valid_edges(edges: tuple[int, ...]) -> bool:
    active = sum(value != 0 for value in edges)
    return 2 <= active <= 6 and all(
        any(
            value != 0 and (source == gene or target == gene)
            for value, (source, target) in zip(edges, EDGE_INDEX)
        )
        for gene in range(3)
    )


# The complete grammar is small enough to enumerate once. Local models receive
# a much smaller, dynamically filtered menu rather than this complete list.
VALID_EDGE_ARRAYS = [
    list(edges)
    for edges in itertools.product(range(3), repeat=9)
    if rust_valid_edges(edges)
]


def topology_key(edges: Iterable[int]) -> str:
    return "".join(str(value) for value in edges)


def topology_from_key(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, str) or re.fullmatch(r"[012]{9}", value) is None:
        return None
    edges = tuple(int(digit) for digit in value)
    return edges if rust_valid_edges(edges) else None


def structural_signature(edges: tuple[int, ...]) -> tuple[int, int, int]:
    """Active, activating and active self-edge counts for menu balancing."""
    return (
        sum(value != 0 for value in edges),
        sum(value == 1 for value in edges),
        sum(value != 0 for value in edges[:3]),
    )


def stable_order(
    values: Iterable[tuple[int, ...]], attempt: int, namespace: str
) -> list[tuple[int, ...]]:
    """Order candidates reproducibly without depending on Python's hash seed."""
    prefix = f"{PROTOCOL}:{attempt}:{namespace}:"
    return sorted(
        values,
        key=lambda edges: hashlib.sha256(
            (prefix + topology_key(edges)).encode("ascii")
        ).digest(),
    )


def seen_topologies(observation: dict[str, Any]) -> set[tuple[int, ...]]:
    seen = {
        edges
        for item in observation.get("evaluated", [])
        if isinstance(item, dict)
        for edges in [topology_from_key(item.get("topology"))]
        if edges is not None
    }
    seen.update(
        edges
        for value in observation.get("rejected_keys", [])
        for edges in [topology_from_key(value)]
        if edges is not None
    )
    return seen


def elite_topologies(
    observation: dict[str, Any], maximum: int = 8
) -> list[tuple[int, ...]]:
    ranked: list[tuple[float, str, tuple[int, ...]]] = []
    for item in observation.get("evaluated", []):
        if not isinstance(item, dict):
            continue
        edges = topology_from_key(item.get("topology"))
        score = item.get("validation_score")
        if (
            edges is None
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            continue
        ranked.append((float(score), topology_key(edges), edges))
    ranked.sort()
    return [edges for _, _, edges in ranked[:maximum]]


def one_edge_mutations(edges: tuple[int, ...]) -> set[tuple[int, ...]]:
    mutations = set()
    for index, current in enumerate(edges):
        for replacement in range(3):
            if replacement == current:
                continue
            mutated = edges[:index] + (replacement,) + edges[index + 1 :]
            if rust_valid_edges(mutated):
                mutations.add(mutated)
    return mutations


def build_candidate_menu(
    observation: dict[str, Any], menu_size: int
) -> list[dict[str, Any]]:
    """Build a deterministic, diverse menu containing only valid unseen rows."""
    if menu_size <= 0:
        raise ValueError("candidate_menu_size must be positive")
    menu_size = min(menu_size, 256)
    attempt = int(observation.get("proposal_attempt", 0))
    seen = seen_topologies(observation)
    universe = [tuple(edges) for edges in VALID_EDGE_ARRAYS if tuple(edges) not in seen]
    if not universe:
        raise ValueError("the complete topology grammar has already been evaluated")
    target = min(menu_size, len(universe))
    mutation_quota = target // 3
    structure_quota = target // 3
    selected: dict[tuple[int, ...], str] = {}

    # Round-robin mutations prevent one best incumbent from monopolizing the
    # exploitation third of the menu.
    mutation_lists = [
        stable_order(one_edge_mutations(elite) - seen, attempt, f"elite-{rank}")
        for rank, elite in enumerate(elite_topologies(observation))
    ]
    depth = 0
    while len(selected) < mutation_quota and any(
        depth < len(mutations) for mutations in mutation_lists
    ):
        for mutations in mutation_lists:
            if depth < len(mutations):
                selected.setdefault(mutations[depth], "elite-mutation")
                if len(selected) >= mutation_quota:
                    break
        depth += 1

    evaluated_signatures: dict[tuple[int, int, int], int] = {}
    for edges in seen_topologies({"evaluated": observation.get("evaluated", [])}):
        signature = structural_signature(edges)
        evaluated_signatures[signature] = evaluated_signatures.get(signature, 0) + 1
    groups: dict[tuple[int, int, int], list[tuple[int, ...]]] = {}
    for edges in universe:
        if edges not in selected:
            groups.setdefault(structural_signature(edges), []).append(edges)
    ordered_groups = sorted(
        groups,
        key=lambda signature: (
            evaluated_signatures.get(signature, 0),
            signature,
        ),
    )
    ordered_group_values = [
        stable_order(groups[signature], attempt, f"structure-{signature}")
        for signature in ordered_groups
    ]
    structure_target = min(target, len(selected) + structure_quota)
    depth = 0
    while len(selected) < structure_target and any(
        depth < len(values) for values in ordered_group_values
    ):
        for values in ordered_group_values:
            if depth < len(values):
                selected.setdefault(values[depth], "underrepresented-structure")
                if len(selected) >= structure_target:
                    break
        depth += 1

    immigrants = stable_order(
        (edges for edges in universe if edges not in selected),
        attempt,
        "random-immigrant",
    )
    for edges in immigrants:
        selected[edges] = "random-immigrant"
        if len(selected) >= target:
            break

    # Shuffle source blocks deterministically before assigning opaque IDs. The
    # model cannot learn that an early ID always means an elite mutation.
    ordered = stable_order(selected, attempt, "menu-order")
    menu = []
    for index, edges in enumerate(ordered):
        active, activating, self_edges = structural_signature(edges)
        menu.append(
            {
                "candidate_id": f"c{index:03d}",
                "edges": list(edges),
                "source": selected[edges],
                "active_edges": active,
                "activating_edges": activating,
                "self_edges": self_edges,
            }
        )
    return menu


def menu_selection_schema(menu: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
                "enum": [candidate["candidate_id"] for candidate in menu],
            }
        },
        "required": ["candidate_id"],
        "additionalProperties": False,
    }


def proposal_from_menu(
    value: dict[str, Any], menu: list[dict[str, Any]]
) -> dict[str, Any]:
    candidates = {candidate["candidate_id"]: candidate["edges"] for candidate in menu}
    candidate_id = value.get("candidate_id")
    if candidate_id not in candidates:
        raise ValueError("model did not select a candidate from the supplied menu")
    return {"edges": candidates[candidate_id]}


def http_request(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> urllib.request.Request:
    """Create a compact JSON POST request without exposing credentials."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    return request


def provider_http_error(error: urllib.error.HTTPError) -> RuntimeError:
    """Preserve the provider response body while keeping the API key private."""
    detail = error.read(4096).decode("utf-8", errors="replace")
    return RuntimeError(f"provider HTTP {error.code}: {detail}")


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(
            http_request(url, headers, payload), timeout=timeout_seconds
        ) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        raise provider_http_error(error) from error
    if not isinstance(value, dict):
        raise ValueError("provider response is not a JSON object")
    return value


def sse_json_events(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Decode JSON payloads from an Anthropic-compatible SSE byte stream."""
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                data = "\n".join(data_lines)
                data_lines.clear()
                if data != "[DONE]":
                    event = json.loads(data)
                    if not isinstance(event, dict):
                        raise ValueError("provider SSE data is not a JSON object")
                    yield event
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            event = json.loads(data)
            if not isinstance(event, dict):
                raise ValueError("provider SSE data is not a JSON object")
            yield event


def parse_anthropic_stream(
    lines: Iterable[bytes], configured_model: str
) -> dict[str, Any]:
    """Assemble tool input and usage while discarding private thinking deltas."""
    model = configured_model
    text_parts: list[str] = []
    tool_blocks: dict[int, dict[str, Any]] = {}
    tool_json_parts: dict[int, list[str]] = {}
    usage: dict[str, Any] = {}
    message_started = False
    message_stopped = False
    stop_reason = None

    for event in sse_json_events(lines):
        event_type = event.get("type")
        if event_type == "error":
            error = event.get("error") or {}
            raise RuntimeError(
                "provider stream error "
                f"{error.get('type', 'unknown')}: {error.get('message', 'unknown')}"
            )
        if event_type == "message_start":
            message_started = True
            message = event.get("message") or {}
            model = message.get("model", model)
            usage.update(message.get("usage") or {})
        elif event_type == "content_block_start":
            index = int(event.get("index", 0))
            block = event.get("content_block") or {}
            if block.get("type") == "text" and block.get("text"):
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_blocks[index] = {
                    "type": "tool_use",
                    "name": block.get("name"),
                    "input": block.get("input") or {},
                }
                tool_json_parts[index] = []
        elif event_type == "content_block_delta":
            index = int(event.get("index", 0))
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                text_parts.append(delta["text"])
            elif (
                delta.get("type") == "input_json_delta"
                and delta.get("partial_json") is not None
            ):
                tool_json_parts.setdefault(index, []).append(delta["partial_json"])
        elif event_type == "message_delta":
            delta = event.get("delta") or {}
            stop_reason = delta.get("stop_reason", stop_reason)
            usage.update(event.get("usage") or {})
        elif event_type == "message_stop":
            message_stopped = True

    if not message_started:
        raise ValueError("provider SSE stream contains no message_start")
    if not message_stopped:
        raise ValueError("provider SSE stream ended before message_stop")
    content: list[dict[str, Any]] = []
    for index in sorted(tool_blocks):
        block = tool_blocks[index]
        partial = "".join(tool_json_parts.get(index, []))
        if partial:
            block["input"] = json.loads(partial)
        content.append(block)
    if text_parts:
        content.append({"type": "text", "text": "".join(text_parts)})
    if not content:
        raise ValueError(
            "provider SSE stream contains no tool or text result "
            f"(stop_reason={stop_reason}, output_tokens={usage.get('output_tokens')})"
        )
    return {
        "model": model,
        "content": content,
        "usage": usage,
    }


def post_anthropic_stream(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    configured_model: str,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(
            http_request(url, headers, payload), timeout=timeout_seconds
        ) as response:
            return parse_anthropic_stream(response, configured_model)
    except urllib.error.HTTPError as error:
        raise provider_http_error(error) from error


def object_from_text(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response contains no JSON object")
    return json.loads(text[start : end + 1])


def is_loopback_url(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname
    return host in {"127.0.0.1", "localhost", "::1"}


def load_config(path: Path) -> tuple[dict[str, Any], str | None, int]:
    if not path.is_file():
        raise SystemExit(f"agent config does not exist: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read agent config {path}: {error}") from error
    required = {"provider", "model", "base_url", "max_tokens"}
    missing = sorted(required.difference(config))
    if missing:
        raise SystemExit(f"agent config is missing: {', '.join(missing)}")
    if config["provider"] not in {"anthropic", "openai-compatible"}:
        raise SystemExit(f"unsupported provider {config['provider']}")
    if not str(config["model"]).strip() or not str(config["base_url"]).strip():
        raise SystemExit("model and base_url must be non-empty")
    key_name = str(config.get("api_key_env", "")).strip()
    key = os.environ.get(key_name) if key_name else None
    if not key and not is_loopback_url(str(config["base_url"])):
        if not key_name:
            raise SystemExit("remote base_url requires api_key_env")
        raise SystemExit(f"missing environment variable {key_name}")
    maximum = int(config["max_tokens"])
    if maximum <= 0:
        raise SystemExit("max_tokens must be a positive deliberate budget")
    menu_size = int(config.get("candidate_menu_size", 96))
    if menu_size <= 0 or menu_size > 256:
        raise SystemExit("candidate_menu_size must be in 1..=256")
    return config, key, maximum


def build_anthropic_call(
    config: dict[str, Any], key: str, maximum: int, prompt: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build the dual-authenticated streaming request proven by GTOC1."""
    base_url = str(config["base_url"]).rstrip("/")
    endpoint = (
        f"{base_url}/messages"
        if base_url.endswith("/v1")
        else f"{base_url}/v1/messages"
    )
    headers = {
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {key}",
        "X-Api-Key": key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["model"],
        "max_tokens": maximum,
        "thinking": config.get("thinking", {"type": "adaptive"}),
        "stream": True,
        "tools": [TOPOLOGY_TOOL],
        "tool_choice": {"type": "tool", "name": TOPOLOGY_TOOL["name"]},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }
    return endpoint, headers, payload


def build_openai_call(
    config: dict[str, Any],
    key: str | None,
    maximum: int,
    prompt: str,
    schema: dict[str, Any] = TOPOLOGY_SCHEMA,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build a llama.cpp-compatible schema-constrained chat request."""
    base_url = str(config["base_url"]).rstrip("/")
    endpoint = (
        f"{base_url}/chat/completions"
        if base_url.endswith("/v1")
        else f"{base_url}/v1/chat/completions"
    )
    headers = {"authorization": f"Bearer {key}"} if key else {}
    payload = {
        "model": config["model"],
        "max_tokens": maximum,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "oscillator_topology",
                "strict": True,
                "schema": schema,
            },
        },
    }
    if "temperature" in config:
        payload["temperature"] = float(config["temperature"])
    if "chat_template_kwargs" in config:
        payload["chat_template_kwargs"] = config["chat_template_kwargs"]
    return endpoint, headers, payload


def proposal_from_anthropic(response: dict[str, Any]) -> dict[str, Any]:
    calls = [
        block
        for block in response.get("content", [])
        if block.get("type") == "tool_use"
        and block.get("name") == TOPOLOGY_TOOL["name"]
    ]
    if len(calls) != 1 or not isinstance(calls[0].get("input"), dict):
        raise ValueError("provider returned no unique propose_topology tool call")
    return calls[0]["input"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate local configuration and credentials without an API request",
    )
    arguments = parser.parse_args()
    config, key, maximum = load_config(arguments.config)
    if arguments.check:
        json.dump(
            {
                "status": "ok",
                "provider": config["provider"],
                "model": config["model"],
                "max_tokens": maximum,
                "protocol": PROTOCOL,
                "output_contract": (
                    "forced-tool"
                    if config["provider"] == "anthropic"
                    else "candidate-menu-json-schema"
                ),
                "authentication": "environment" if key else "loopback-none",
            },
            sys.stdout,
            separators=(",", ":"),
        )
        sys.stdout.write("\n")
        return 0
    observation = json.load(sys.stdin)
    base_prompt = (
        "Propose one signed three-gene regulatory topology. Think briefly "
        "(at most 500 reasoning tokens), then return only "
        '{"edges":[nine integers]}. Obey the grammar in this observation and '
        "do not repeat a rejected/evaluated topology. Lower score is better.\n"
        + json.dumps(observation, separators=(",", ":"))
    )
    provider = config["provider"]
    timeout_seconds = int(config.get("timeout_seconds", 180))
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if provider == "anthropic":
        if key is None:  # remote Anthropic URLs are rejected by load_config
            raise AssertionError("Anthropic transport requires a key")
        endpoint, headers, payload = build_anthropic_call(
            config, key, maximum, base_prompt
        )
        response = post_anthropic_stream(
            endpoint, headers, payload, str(config["model"]), timeout_seconds
        )
        proposal = proposal_from_anthropic(response)
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    elif provider == "openai-compatible":
        menu_size = int(config.get("candidate_menu_size", 96))
        menu = build_candidate_menu(observation, menu_size)
        local_observation = dict(observation)
        local_observation["rejected_keys"] = list(
            dict.fromkeys(observation.get("rejected_keys", []))
        )
        local_observation["candidate_menu"] = menu
        prompt = (
            "Choose exactly one candidate_id from candidate_menu. Every listed "
            "topology is grammar-valid and unseen. Use scores and structural "
            "metadata to balance exploitation and novelty; lower score is better. "
            'Return only {"candidate_id":"cNNN"}.\n'
            + json.dumps(local_observation, separators=(",", ":"))
        )
        endpoint, headers, payload = build_openai_call(
            config, key, maximum, prompt, menu_selection_schema(menu)
        )
        response = post_json(endpoint, headers, payload, timeout_seconds)
        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        proposal = proposal_from_menu(object_from_text(text), menu)
    else:  # validated by load_config
        raise AssertionError(f"unsupported provider {provider}")
    proposal["input_tokens"] = int(input_tokens)
    proposal["output_tokens"] = int(output_tokens)
    json.dump(proposal, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"llm_agent: {error}", file=sys.stderr)
        raise SystemExit(1) from error
