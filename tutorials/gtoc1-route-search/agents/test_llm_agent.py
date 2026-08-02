"""Offline tests for Anthropic and llama.cpp route adapters."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agents.llm_agent import (
    PROTOCOL,
    build_candidate_menu,
    build_provider_call,
    canonical_clockwise,
    parse_anthropic_stream,
    parse_provider_response,
    proposal_from_menu,
)


def request(provider: str, base_url: str | None = None) -> dict:
    return {
        "protocol_version": 2,
        "accepted_candidates": 0,
        "accepted_candidates_target": 100,
        "proposal_attempt": 7,
        "phase": "bootstrap",
        "system": "system",
        "user": "user",
        "constraints": {
            "bodies": {"Venus": 2, "Earth": 3, "Jupiter": 5, "Saturn": 6, "TW229": 10},
            "maximum_encounters": 14,
            "maximum_same_body_run": 4,
            "maximum_outer_encounters": 4,
            "maximum_variants_per_structure": 1,
            "minimum_edit_distance": 3,
        },
        "archive": {
            "already_evaluated_variants": ["3-2-3-3-3-5-6-5-10|00000011"],
            "structure_variant_counts": {"3-2-3-3-3-5-6-5-10": 1},
            "top": [
                {
                    "variant_key": "3-2-3-3-3-5-6-5-10|00000011",
                    "mga_score": 1_840_000.0,
                }
            ],
        },
        "response_schema": {"type": "object"},
        "adapter": {
            "provider": provider,
            "model": "MiniMax-M3" if provider == "anthropic-compatible" else "gemma-4-31b-it",
            "base_url": base_url
            or (
                "https://api.minimax.io/anthropic"
                if provider == "anthropic-compatible"
                else "http://127.0.0.1:8080/v1"
            ),
            "maximum_tokens": 8000,
            "provider_options": (
                {"thinking": {"type": "adaptive"}}
                if provider == "anthropic-compatible"
                else {
                    "candidate_menu_size": 24,
                    "temperature": 0.6,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            ),
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
    def test_local_llamacpp_needs_no_fake_secret_and_uses_menu_schema(self) -> None:
        endpoint, payload, headers, provider, model, menu_context = build_provider_call(
            request("openai-compatible"), None
        )
        self.assertEqual(PROTOCOL, "gtoc1-mga-route-proposal-v2")
        self.assertEqual(endpoint, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(provider, "openai-compatible")
        self.assertEqual(model, "gemma-4-31b-it")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(len((menu_context or {})["rows"]), 24)
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(len(schema["properties"]["candidate_id"]["enum"]), 24)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_menu_contains_only_unseen_body_orders_and_canonical_directions(self) -> None:
        value = request("openai-compatible")
        menu = build_candidate_menu(value, 64)
        structures = ["-".join(str({"Venus": 2, "Earth": 3, "Jupiter": 5, "Saturn": 6, "TW229": 10}[name]) for name in row["bodies"]) for row in menu]
        self.assertEqual(len(structures), len(set(structures)))
        self.assertNotIn("3-2-3-3-3-5-6-5-10", structures)
        jpl_bodies = (3, 2, 3, 3, 3, 5, 6, 5, 10)
        self.assertEqual(canonical_clockwise(jpl_bodies), (False,) * 6 + (True, True))
        selected = proposal_from_menu({"candidate_id": menu[0]["candidate_id"]}, menu)
        self.assertNotIn("clockwise", selected["candidates"][0])

    def test_remote_openai_compatible_endpoint_requires_a_secret(self) -> None:
        with self.assertRaisesRegex(ValueError, "remote provider requires"):
            build_provider_call(
                request("openai-compatible", "https://invalid.example/v1"), None
            )

    def test_minimax_anthropic_adaptive_thinking_request(self) -> None:
        endpoint, payload, headers, provider, model, menu_context = build_provider_call(
            request("anthropic-compatible"), "test-key"
        )
        self.assertEqual(endpoint, "https://api.minimax.io/anthropic/v1/messages")
        self.assertEqual(provider, "anthropic-compatible")
        self.assertEqual(model, "MiniMax-M3")
        self.assertIsNone(menu_context)
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["X-Api-Key"], "test-key")
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertIs(payload["stream"], True)

    def test_anthropic_stream_discards_thinking_and_preserves_usage(self) -> None:
        response = parse_anthropic_stream(
            sse_lines(
                {
                    "type": "message_start",
                    "message": {"model": "MiniMax-M3", "usage": {"input_tokens": 41}},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "private"},
                },
                {
                    "type": "content_block_start",
                    "content_block": {"type": "text", "text": '{"candidates":'},
                },
                {
                    "type": "content_block_delta",
                    "delta": {
                        "type": "text_delta",
                        "text": '[{"bodies":["Earth","Venus","TW229"],"rationale":"test"}]}',
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
        self.assertEqual(parsed["candidates"][0]["rationale"], "test")
        self.assertEqual(parsed["usage"]["input_tokens"], 41)
        self.assertEqual(parsed["usage"]["output_tokens"], 73)
        self.assertNotIn("private", str(response))

    def test_assisted_menu_is_stratified_compact_and_returns_ranked_fallbacks(self) -> None:
        rows = []
        for index, bodies in enumerate(
            (
                [3, 6, 5, 10],
                [3, 2, 6, 5, 10],
                [3, 3, 3, 2, 3, 3, 5, 10],
                [3, 3, 3, 2, 2, 3, 3, 5, 10],
                [3, 3, 2, 2, 3, 2, 3, 3, 5, 10],
                [3, 3, 3, 2, 3, 2, 2, 3, 3, 5, 10],
                [3, 3, 3, 3, 5, 2, 3, 2, 3, 5, 6, 5, 10],
                [3, 3, 3, 3, 5, 3, 2, 3, 2, 3, 5, 6, 5, 10],
            )
        ):
            rows.append(
                {
                    "structure": {"bodies": bodies},
                    "l0": {
                        "estimated_score": 900_000.0 + index * 10_000.0,
                        "worker_seconds": 1_000.0 + index * 100.0,
                    },
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps({"results": rows}), encoding="utf-8")
            value = request("openai-compatible")
            value["phase"] = "exploit"
            value["accepted_candidates"] = 8
            value["archive"]["length_counts"] = [
                {"encounters": 8, "evaluated": 2},
                {"encounters": 14, "evaluated": 6},
            ]
            value["archive"]["length_evidence"] = []
            value["archive"]["portfolio"] = {
                "target_size": 20,
                "retained": 8,
                "score_sum": 7_000_000.0,
                "cutoff_mga_score": 700_000.0,
            }
            value["adapter"]["provider_options"].update(
                {
                    "menu_policy": "gemma4-assisted-v1",
                    "candidate_menu_size": 24,
                    "ranked_candidates": 3,
                    "experience_archive_paths": [str(path)],
                    "experience_sha256": hashlib.sha256(
                        len(path.read_bytes()).to_bytes(8, "big") + path.read_bytes()
                    ).hexdigest(),
                }
            )
            _endpoint, payload, _headers, _provider, model, context = build_provider_call(
                value, None
            )
            self.assertEqual(model, "gemma-4-31b-it")
            self.assertIsNotNone(context)
            menu = context["rows"]
            self.assertEqual(len(menu), 24)
            self.assertEqual(
                {name: sum(row["length_band"] == name for row in menu) for name in {row["length_band"] for row in menu}},
                {"3-6": 4, "7-9": 8, "10-11": 8, "12-14": 4},
            )
            prompt = payload["messages"][1]["content"]
            self.assertNotIn("already_evaluated_variants", prompt)
            self.assertNotIn("structure_variant_counts", prompt)
            self.assertIn("declared_prior_evidence", prompt)
            schema = payload["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema["properties"]["ranked_candidate_ids"]["minItems"], 3)
            identifiers = [row["candidate_id"] for row in menu[:3]]
            response = proposal_from_menu(
                {"ranked_candidate_ids": identifiers},
                menu,
                context["policy"],
                context["ranked_candidates"],
                context["experience_digest"],
            )
            self.assertEqual(len(response["candidates"]), 3)
            self.assertIn("experience_sha256=", response["candidates"][0]["rationale"])
            parsed = parse_provider_response(
                "openai-compatible",
                {
                    "model": "gemma-4-31b-it",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"ranked_candidate_ids": identifiers}
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 123, "completion_tokens": 12},
                },
                "gemma-4-31b-it",
                context,
            )
            self.assertEqual(len(parsed["candidates"]), 3)
            self.assertEqual(parsed["usage"]["input_tokens"], 123)
            value["adapter"]["provider_options"]["experience_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_provider_call(value, None)

    def test_assisted_bootstrap_cycles_controlled_length_bands(self) -> None:
        rows = [
            {
                "structure": {"bodies": [3, 6, 5, 10]},
                "l0": {"estimated_score": 1_000_000.0, "worker_seconds": 1_000.0},
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps({"results": rows}), encoding="utf-8")
            expected = ["7-9", "10-11", "3-6", "12-14"]
            for accepted, band in enumerate(expected):
                value = request("openai-compatible")
                value["accepted_candidates"] = accepted
                value["adapter"]["provider_options"].update(
                    {
                        "menu_policy": "gemma4-assisted-v1",
                        "candidate_menu_size": 16,
                        "ranked_candidates": 3,
                        "experience_archive_paths": [str(path)],
                        "experience_sha256": hashlib.sha256(
                            len(path.read_bytes()).to_bytes(8, "big") + path.read_bytes()
                        ).hexdigest(),
                    }
                )
                _endpoint, _payload, _headers, _provider, _model, context = build_provider_call(
                    value, None
                )
                self.assertEqual({row["length_band"] for row in context["rows"]}, {band})


if __name__ == "__main__":
    unittest.main()
