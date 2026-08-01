#!/usr/bin/env python3

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agents.llm_agent import (
    PROTOCOL,
    VALID_EDGE_ARRAYS,
    build_anthropic_call,
    build_candidate_menu,
    build_openai_call,
    menu_selection_schema,
    parse_anthropic_stream,
    proposal_from_anthropic,
    proposal_from_menu,
    rust_valid_edges,
)


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
                    "protocol": PROTOCOL,
                    "output_contract": "forced-tool",
                    "authentication": "environment",
                },
            )

    def test_local_llamacpp_preflight_needs_no_fake_secret(self) -> None:
        config = {
            "provider": "openai-compatible",
            "model": "local-test-model",
            "base_url": "http://127.0.0.1:8080/v1",
            "max_tokens": 8000,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            completed = subprocess.run(
                ["python3", "agents/llm_agent.py", "--config", str(path), "--check"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["authentication"], "loopback-none")
            self.assertEqual(result["output_contract"], "candidate-menu-json-schema")
            self.assertEqual(result["protocol"], PROTOCOL)

    def test_remote_openai_compatible_url_still_requires_a_secret(self) -> None:
        config = {
            "provider": "openai-compatible",
            "model": "remote-test-model",
            "base_url": "https://invalid.example/v1",
            "max_tokens": 8000,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            completed = subprocess.run(
                ["python3", "agents/llm_agent.py", "--config", str(path), "--check"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("remote base_url requires api_key_env", completed.stderr)

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
        self.assertEqual(payload["tool_choice"], {"type": "tool", "name": "propose_topology"})
        schema = payload["tools"][0]["input_schema"]
        self.assertEqual(schema["properties"]["edges"]["minItems"], 9)
        self.assertEqual(schema["properties"]["edges"]["maxItems"], 9)
        self.assertEqual(
            payload["messages"][0]["content"],
            [{"type": "text", "text": "proposal prompt"}],
        )
        config["base_url"] = "https://api.minimax.io/anthropic"
        gtoc_endpoint, _, _ = build_anthropic_call(
            config, "test-key", 4096, "proposal prompt"
        )
        self.assertEqual(gtoc_endpoint, endpoint)

    def test_anthropic_sse_collects_tool_input_usage_and_discards_thinking(self) -> None:
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
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "name": "propose_topology",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"edges":[0,1,0,0,0,2,0,0,0]}',
                    },
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 73},
                },
                {"type": "message_stop"},
            ),
            "configured-model",
        )
        self.assertEqual(response["model"], "MiniMax-M3")
        self.assertEqual(
            proposal_from_anthropic(response),
            {"edges": [0, 1, 0, 0, 0, 2, 0, 0, 0]},
        )
        self.assertEqual(response["usage"]["input_tokens"], 41)
        self.assertEqual(response["usage"]["output_tokens"], 73)
        self.assertNotIn("private reasoning", str(response))

    def test_openai_call_uses_llamacpp_candidate_menu_schema(self) -> None:
        menu = build_candidate_menu(
            {
                "proposal_attempt": 7,
                "evaluated": [],
                "rejected_keys": [],
            },
            12,
        )
        selection_schema = menu_selection_schema(menu)
        endpoint, headers, payload = build_openai_call(
            {
                "model": "gemma-4-12b-it-q4",
                "base_url": "http://127.0.0.1:8080/v1",
                "temperature": 0.6,
            },
            None,
            8000,
            "proposal prompt",
            selection_schema,
        )
        self.assertEqual(endpoint, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(headers, {})
        self.assertEqual(payload["max_tokens"], 8000)
        self.assertEqual(payload["temperature"], 0.6)
        response_format = payload["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        self.assertIs(schema, selection_schema)
        choices = schema["properties"]["candidate_id"]["enum"]
        self.assertEqual(choices, [candidate["candidate_id"] for candidate in menu])

        _, _, remote_payload = build_openai_call(
            {
                "model": "remote-model",
                "base_url": "https://remote.example/v1",
            },
            "secret",
            8000,
            "proposal prompt",
        )
        remote_schema = remote_payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(remote_schema["properties"]["edges"]["maxItems"], 9)

    def test_candidate_menu_is_deterministic_valid_unique_and_unseen(self) -> None:
        observation = {
            "proposal_attempt": 19,
            "evaluated": [
                {"topology": "111000000", "validation_score": 0.4},
                {"topology": "000200220", "validation_score": 0.8},
                {"topology": "120010001", "validation_score": 1.1},
            ],
            "rejected_keys": ["111000000", "020222002", "020222002"],
        }
        menu = build_candidate_menu(observation, 96)
        self.assertEqual(menu, build_candidate_menu(observation, 96))
        keys = ["".join(map(str, candidate["edges"])) for candidate in menu]
        self.assertEqual(len(keys), 96)
        self.assertEqual(len(set(keys)), 96)
        self.assertTrue(
            all(rust_valid_edges(tuple(candidate["edges"])) for candidate in menu)
        )
        self.assertTrue(
            {"111000000", "000200220", "120010001", "020222002"}.isdisjoint(keys)
        )
        self.assertEqual(
            {candidate["source"] for candidate in menu},
            {"elite-mutation", "underrepresented-structure", "random-immigrant"},
        )
        self.assertNotEqual(menu, build_candidate_menu({**observation, "proposal_attempt": 20}, 96))

    def test_menu_choice_translates_to_existing_rust_contract(self) -> None:
        menu = build_candidate_menu(
            {"proposal_attempt": 1, "evaluated": [], "rejected_keys": []}, 8
        )
        selected = proposal_from_menu({"candidate_id": "c003"}, menu)
        self.assertEqual(selected, {"edges": menu[3]["edges"]})
        with self.assertRaisesRegex(ValueError, "supplied menu"):
            proposal_from_menu({"candidate_id": "not-listed"}, menu)

    def test_complete_rust_grammar_is_stable(self) -> None:
        self.assertEqual(len(VALID_EDGE_ARRAYS), 12_024)
        self.assertNotIn([1] * 9, VALID_EDGE_ARRAYS)
        self.assertIn([0, 0, 0, 2, 0, 0, 2, 2, 0], VALID_EDGE_ARRAYS)

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
