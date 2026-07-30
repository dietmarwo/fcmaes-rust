"""Offline protocol tests for the optional provider adapter."""

from __future__ import annotations

import json
import unittest

from agents.llm_agent import (
    build_provider_call,
    parse_anthropic_stream,
    parse_provider_response,
)


def request(provider: str) -> dict:
    return {
        "system": "system",
        "user": "user",
        "constraints": {"bodies": {"Earth": 3, "Venus": 2, "TW229": 10}},
        "response_schema": {"type": "object"},
        "adapter": {
            "provider": provider,
            "model": "MiniMax-M3",
            "base_url": "https://api.minimax.io/anthropic",
            "maximum_tokens": 8192,
            "provider_options": {"thinking": {"type": "adaptive"}},
        },
    }


def sse_lines(*events: dict) -> list[bytes]:
    lines: list[bytes] = []
    for event in events:
        lines.extend(
            [
                f"event: {event['type']}\n".encode(),
                f"data: {json.dumps(event)}\n".encode(),
                b"\n",
            ]
        )
    return lines


class ProviderAdapterTest(unittest.TestCase):
    def test_minimax_anthropic_adaptive_thinking_request(self) -> None:
        endpoint, payload, headers, provider, model = build_provider_call(
            request("anthropic-compatible"), "test-key"
        )
        self.assertEqual(
            endpoint, "https://api.minimax.io/anthropic/v1/messages"
        )
        self.assertEqual(provider, "anthropic-compatible")
        self.assertEqual(model, "MiniMax-M3")
        self.assertEqual(headers["Accept"], "text/event-stream")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["X-Api-Key"], "test-key")
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertIs(payload["stream"], True)
        user_prompt = payload["messages"][0]["content"][0]["text"]
        self.assertIn('"candidates":[{"bodies"', user_prompt)
        self.assertIn('"Earth":3', user_prompt)
        self.assertIn("JSON Schema:", user_prompt)
        self.assertNotIn("response_format", payload)

    def test_anthropic_thinking_is_not_mixed_with_candidate_json(self) -> None:
        response = {
            "model": "MiniMax-M3",
            "content": [
                {"type": "thinking", "thinking": "private reasoning"},
                {
                    "type": "text",
                    "text": (
                        '{"candidates":[{"bodies":["Earth","Venus","TW229"],'
                        '"clockwise":[false,false],"rationale":"test"}]}'
                    ),
                },
            ],
            "usage": {
                "input_tokens": 41,
                "output_tokens": 73,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 7,
            },
        }
        parsed = parse_provider_response(
            "anthropic-compatible", response, "MiniMax-M3"
        )
        self.assertEqual(parsed["candidates"][0]["rationale"], "test")
        self.assertNotIn("private reasoning", str(parsed))
        self.assertEqual(parsed["usage"]["input_tokens"], 41)
        self.assertEqual(parsed["usage"]["output_tokens"], 73)
        self.assertEqual(parsed["usage"]["cache_read_tokens"], 5)
        self.assertEqual(parsed["usage"]["cache_write_tokens"], 7)

    def test_anthropic_sse_assembles_text_usage_and_discards_thinking(self) -> None:
        response = parse_anthropic_stream(
            sse_lines(
                {
                    "type": "message_start",
                    "message": {
                        "model": "MiniMax-M3",
                        "usage": {
                            "input_tokens": 41,
                            "cache_read_input_tokens": 5,
                            "cache_creation_input_tokens": 7,
                        },
                    },
                },
                {
                    "type": "content_block_delta",
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": "private reasoning",
                    },
                },
                {
                    "type": "content_block_start",
                    "content_block": {"type": "text", "text": '{"candidates":'},
                },
                {
                    "type": "content_block_delta",
                    "delta": {
                        "type": "text_delta",
                        "text": '[{"bodies":["Earth","Venus","TW229"],',
                    },
                },
                {
                    "type": "content_block_delta",
                    "delta": {
                        "type": "text_delta",
                        "text": (
                            '"clockwise":[false,false],"rationale":"stream"}]}'
                        ),
                    },
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 73},
                },
                {"type": "message_stop"},
            ),
            "configured-model",
        )
        parsed = parse_provider_response(
            "anthropic-compatible", response, "configured-model"
        )
        self.assertEqual(parsed["candidates"][0]["rationale"], "stream")
        self.assertEqual(parsed["usage"]["model"], "MiniMax-M3")
        self.assertEqual(parsed["usage"]["input_tokens"], 41)
        self.assertEqual(parsed["usage"]["output_tokens"], 73)
        self.assertNotIn("private reasoning", str(response))

    def test_anthropic_sse_rejects_provider_errors_and_truncation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "overloaded_error"):
            parse_anthropic_stream(
                sse_lines(
                    {
                        "type": "error",
                        "error": {
                            "type": "overloaded_error",
                            "message": "try later",
                        },
                    }
                ),
                "MiniMax-M3",
            )
        with self.assertRaisesRegex(ValueError, "before message_stop"):
            parse_anthropic_stream(
                sse_lines(
                    {
                        "type": "message_start",
                        "message": {"model": "MiniMax-M3", "usage": {}},
                    }
                ),
                "MiniMax-M3",
            )
        with self.assertRaisesRegex(
            ValueError, r"stop_reason=max_tokens, output_tokens=4096"
        ):
            parse_anthropic_stream(
                sse_lines(
                    {
                        "type": "message_start",
                        "message": {"model": "MiniMax-M3", "usage": {}},
                    },
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "max_tokens"},
                        "usage": {"output_tokens": 4096},
                    },
                    {"type": "message_stop"},
                ),
                "MiniMax-M3",
            )


if __name__ == "__main__":
    unittest.main()
