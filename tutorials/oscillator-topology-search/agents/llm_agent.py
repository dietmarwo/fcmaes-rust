#!/usr/bin/env python3
"""Anthropic/OpenAI-compatible topology-proposal adapter.

The Anthropic path deliberately uses the same dual-authenticated SSE transport
as the GTOC1 route-search tutorial. Thinking deltas are consumed but never
mixed into the topology JSON returned to the Rust campaign driver.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator


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
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(
            http_request(url, headers, payload), timeout=180
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
    """Assemble text and usage while deliberately discarding thinking deltas."""
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
                "provider stream error "
                f"{error.get('type', 'unknown')}: {error.get('message', 'unknown')}"
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
            delta = event.get("delta") or {}
            stop_reason = delta.get("stop_reason", stop_reason)
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
    return {
        "model": model,
        "content": [{"type": "text", "text": "".join(text_parts)}],
        "usage": usage,
    }


def post_anthropic_stream(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    configured_model: str,
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(
            http_request(url, headers, payload), timeout=180
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


def load_config(path: Path) -> tuple[dict[str, Any], str, int]:
    if not path.is_file():
        raise SystemExit(f"agent config does not exist: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read agent config {path}: {error}") from error
    required = {"provider", "model", "base_url", "api_key_env", "max_tokens"}
    missing = sorted(required.difference(config))
    if missing:
        raise SystemExit(f"agent config is missing: {', '.join(missing)}")
    if config["provider"] not in {"anthropic", "openai-compatible"}:
        raise SystemExit(f"unsupported provider {config['provider']}")
    if not str(config["model"]).strip() or not str(config["base_url"]).strip():
        raise SystemExit("model and base_url must be non-empty")
    key = os.environ.get(str(config["api_key_env"]))
    if not key:
        raise SystemExit(f"missing environment variable {config['api_key_env']}")
    maximum = int(config["max_tokens"])
    if maximum <= 0:
        raise SystemExit("max_tokens must be a positive deliberate budget")
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
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }
    return endpoint, headers, payload


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
            },
            sys.stdout,
            separators=(",", ":"),
        )
        sys.stdout.write("\n")
        return 0
    observation = json.load(sys.stdin)
    prompt = (
        "Propose one signed three-gene regulatory topology. Return only "
        '{"edges":[nine integers]}. Obey the grammar in this observation and '
        "do not repeat a rejected/evaluated topology. Lower score is better.\n"
        + json.dumps(observation, separators=(",", ":"))
    )
    provider = config["provider"]
    if provider == "anthropic":
        endpoint, headers, payload = build_anthropic_call(
            config, key, maximum, prompt
        )
        response = post_anthropic_stream(
            endpoint, headers, payload, str(config["model"])
        )
        text = "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if block.get("type") == "text"
        )
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    elif provider == "openai-compatible":
        response = post_json(
            config["base_url"].rstrip("/") + "/chat/completions",
            {"authorization": f"Bearer {key}"},
            {
                "model": config["model"],
                "max_tokens": maximum,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
        )
        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
    else:  # validated by load_config
        raise AssertionError(f"unsupported provider {provider}")
    proposal = object_from_text(text)
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
