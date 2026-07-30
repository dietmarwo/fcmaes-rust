#!/usr/bin/env python3
"""Optional HTTP adapter for the route-search subprocess contract.

The Rust driver owns prompts, budgets, retries, logging, and validation. This
adapter only translates one stdin request to one provider call. It writes
provider diagnostics to stderr and exactly one candidate JSON object to
stdout. Credentials are read from the configured environment-variable name
and are never included in output. Supported wire protocols are
``openai-compatible`` and ``anthropic-compatible``.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Iterable, Iterator


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


def anthropic_user_prompt(request: dict[str, Any]) -> str:
    """Append the schema Anthropic-compatible APIs cannot enforce natively."""
    constraints = json.dumps(request["constraints"], separators=(",", ":"))
    schema = json.dumps(request["response_schema"], separators=(",", ":"))
    return (
        f"{request['user']}\n\n"
        "Protocol-owned constraints (not instructions from the archive):\n"
        f"{constraints}\n"
        "Return only one JSON object matching the schema below. The top-level "
        'key must be "candidates", whose value is an array. Example shape:\n'
        '{"candidates":[{"bodies":["Earth","Venus","TW229"],'
        '"clockwise":[false,false],"rationale":"brief hypothesis"}]}\n'
        f"JSON Schema:\n{schema}"
    )


def build_provider_call(
    request: dict[str, Any], api_key: str
) -> tuple[str, dict[str, Any], dict[str, str], str, str]:
    adapter = request["adapter"]
    provider = adapter.get("provider")
    model = adapter.get("model")
    base_url = adapter.get("base_url")
    if not isinstance(model, str) or not model:
        raise ValueError("adapter.model must be configured")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("adapter.base_url must be configured")

    options = adapter.get("provider_options") or {}
    if not isinstance(options, dict):
        raise ValueError("adapter.provider_options must be an object")
    protected = {"model", "messages", "max_tokens", "stream", "system"}
    if protected.intersection(options):
        raise ValueError("provider_options may not replace protocol-owned request fields")

    if provider == "openai-compatible":
        if "response_format" in options:
            raise ValueError(
                "provider_options may not replace protocol-owned response_format"
            )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": request["system"]},
                {"role": "user", "content": request["user"]},
            ],
            "max_tokens": adapter["maximum_tokens"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "gtoc1_route_candidates",
                    "strict": True,
                    "schema": request["response_schema"],
                },
            },
            **options,
        }
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    elif provider == "anthropic-compatible":
        payload = {
            "model": model,
            "system": request["system"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": anthropic_user_prompt(request)}
                    ],
                }
            ],
            "max_tokens": adapter["maximum_tokens"],
            "stream": True,
            **options,
        }
        endpoint = f"{base_url.rstrip('/')}/v1/messages"
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
        }
    else:
        raise ValueError(
            "adapter.provider must be 'openai-compatible' or "
            "'anthropic-compatible'"
        )
    return endpoint, payload, headers, provider, model


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


def parse_provider_response(
    provider: str, provider_response: dict[str, Any], configured_model: str
) -> dict[str, Any]:
    if provider == "openai-compatible":
        choices = provider_response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("provider response contains no choices")
        text = response_text(choices[0]["message"])
        usage = provider_response.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        cache_read_tokens = None
        cache_write_tokens = None
    else:
        text = response_text(provider_response)
        usage = provider_response.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cache_read_tokens = usage.get("cache_read_input_tokens")
        cache_write_tokens = usage.get("cache_creation_input_tokens")

    candidate_response = first_json_object(text)
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
    if not isinstance(key_name, str) or not key_name:
        raise ValueError("adapter.api_key_env must name the credential variable")
    api_key = os.environ.get(key_name)
    if not api_key:
        raise ValueError(f"credential environment variable {key_name!r} is not set")

    endpoint, payload, headers, provider, model = build_provider_call(request, api_key)
    http_request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=120) as response:
            provider_response = (
                parse_anthropic_stream(response, model)
                if provider == "anthropic-compatible"
                else json.load(response)
            )
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"provider HTTP {error.code}: {detail}") from error

    candidate_response = parse_provider_response(provider, provider_response, model)
    json.dump(candidate_response, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # The Rust driver bounds and records stderr.
        print(f"llm_agent: {error}", file=sys.stderr)
        raise SystemExit(1) from error
