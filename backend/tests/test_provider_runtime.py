from __future__ import annotations

import json

import httpx

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import ExternalSignal
from trenches_env.provider_runtime import ProviderDecisionRuntime
from trenches_env.source_ingestion import SourceHarvester


def test_configured_provider_runtime_drives_entity_action(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MODEL_PROVIDER_US", "openai")
    monkeypatch.setenv("TRENCHES_MODEL_NAME_US", "gpt-4.1")
    monkeypatch.setenv("TRENCHES_MODEL_API_KEY_ENV_US", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "gpt-4.1"
        assert payload["tool_choice"]["function"]["name"] == "emit_action"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "emit_action",
                                        "arguments": json.dumps(
                                            {
                                                "type": "negotiate",
                                                "target": "gulf",
                                                "summary": "Negotiate deconfliction around the shipping corridor.",
                                            }
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    runtime = ProviderDecisionRuntime(client=httpx.Client(transport=httpx.MockTransport(handler)))
    env = FogOfWarDiplomacyEnv(
        source_harvester=SourceHarvester(auto_start=False),
        provider_runtime=runtime,
    )
    session = env.create_session(seed=7)

    actions = env.resolve_policy_actions(
        session,
        [
            ExternalSignal(
                source="test-feed",
                headline="Shipping risk rises near Hormuz.",
                region="gulf",
                tags=["shipping", "oil"],
                severity=0.4,
            )
        ],
        agent_ids=["us"],
    )

    assert actions["us"].type == "negotiate"
    assert actions["us"].target == "gulf"
    assert actions["us"].metadata["mode"] == "provider_inference"
    assert actions["us"].metadata["provider"] == "openai"


def test_invalid_provider_output_falls_back_to_heuristic_policy(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MODEL_PROVIDER_US", "openai")
    monkeypatch.setenv("TRENCHES_MODEL_NAME_US", "gpt-4.1")
    monkeypatch.setenv("TRENCHES_MODEL_API_KEY_ENV_US", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "emit_action",
                                        "arguments": json.dumps(
                                            {
                                                "type": "oversight_review",
                                                "summary": "Illegal action for US runtime.",
                                            }
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    runtime = ProviderDecisionRuntime(client=httpx.Client(transport=httpx.MockTransport(handler)))
    env = FogOfWarDiplomacyEnv(
        source_harvester=SourceHarvester(auto_start=False),
        provider_runtime=runtime,
    )
    session = env.create_session(seed=7)

    actions = env.resolve_policy_actions(
        session,
        [
            ExternalSignal(
                source="test-feed",
                headline="Shipping risk rises near Hormuz.",
                region="gulf",
                tags=["shipping", "oil"],
                severity=0.4,
            )
        ],
        agent_ids=["us"],
    )

    assert actions["us"].type in {"defend", "negotiate", "intel_query", "hold", "mobilize", "sanction", "strike", "deceive"}
    assert actions["us"].metadata["mode"] == "heuristic_fallback"
    assert "provider_error" in actions["us"].metadata
