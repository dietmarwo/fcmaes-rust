#!/usr/bin/env python3

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agents.llm_agent import build_anthropic_call, parse_anthropic_stream


ROOT = Path(__file__).resolve().parents[1]


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


class MockAgentTest(unittest.TestCase):
    def test_fixture_is_valid_and_contains_no_reference(self) -> None:
        request = {
            "proposal_attempt": 1,
            "grammar": "fixture",
            "objective": "fixture",
            "evaluated": [],
            "rejected_keys": [],
            "repair_error": None,
        }
        completed = subprocess.run(
            ["python3", "agents/mock_agent.py"],
            cwd=ROOT,
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=True,
        )
        edges = json.loads(completed.stdout)["edges"]
        references = {
            (0, 0, 0, 2, 0, 0, 2, 2, 0),
            (0, 0, 0, 1, 0, 0, 1, 2, 0),
            (0, 0, 0, 1, 0, 0, 1, 1, 0),
            (0, 0, 0, 2, 1, 2, 0, 0, 0),
        }
        self.assertEqual(len(edges), 9)
        self.assertNotIn(tuple(edges), references)

    def test_live_adapter_preflight_is_local_and_checks_credentials(self) -> None:
        config = {
            "provider": "anthropic",
            "model": "test-model",
            "base_url": "https://invalid.example/v1",
            "api_key_env": "OSCILLATOR_TEST_API_KEY",
            "max_tokens": 123,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            environment = os.environ.copy()
            environment["OSCILLATOR_TEST_API_KEY"] = "not-a-real-key"
            completed = subprocess.run(
                ["python3", "agents/llm_agent.py", "--config", str(path), "--check"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(
                json.loads(completed.stdout),
                {
                    "status": "ok",
                    "provider": "anthropic",
                    "model": "test-model",
                    "max_tokens": 123,
                },
            )

    def test_live_adapter_preflight_rejects_missing_config(self) -> None:
        completed = subprocess.run(
            [
                "python3",
                "agents/llm_agent.py",
                "--config",
                "does-not-exist.json",
                "--check",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("agent config does not exist", completed.stderr)

    def test_anthropic_call_uses_proven_minimax_transport(self) -> None:
        config = {
            "provider": "anthropic",
            "model": "MiniMax-M3",
            "base_url": "https://api.minimax.io/anthropic/v1",
            "thinking": {"type": "adaptive"},
        }
        endpoint, headers, payload = build_anthropic_call(
            config, "test-key", 4096, "proposal prompt"
        )
        self.assertEqual(endpoint, "https://api.minimax.io/anthropic/v1/messages")
        self.assertEqual(headers["Accept"], "text/event-stream")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["X-Api-Key"], "test-key")
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertIs(payload["stream"], True)
        self.assertEqual(
            payload["messages"][0]["content"],
            [{"type": "text", "text": "proposal prompt"}],
        )
        config["base_url"] = "https://api.minimax.io/anthropic"
        gtoc_endpoint, _, _ = build_anthropic_call(
            config, "test-key", 4096, "proposal prompt"
        )
        self.assertEqual(gtoc_endpoint, endpoint)

    def test_anthropic_sse_discards_thinking_and_collects_usage(self) -> None:
        response = parse_anthropic_stream(
            sse_lines(
                {
                    "type": "message_start",
                    "message": {
                        "model": "MiniMax-M3",
                        "usage": {"input_tokens": 41},
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
                    "content_block": {"type": "text", "text": "{\"edges\":"},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "[0,1,0,0,0,2,0,0,0]}"},
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
        self.assertEqual(response["model"], "MiniMax-M3")
        self.assertEqual(
            response["content"][0]["text"],
            '{"edges":[0,1,0,0,0,2,0,0,0]}',
        )
        self.assertEqual(response["usage"]["input_tokens"], 41)
        self.assertEqual(response["usage"]["output_tokens"], 73)
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


if __name__ == "__main__":
    unittest.main()
