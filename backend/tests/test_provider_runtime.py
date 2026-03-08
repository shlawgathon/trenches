from __future__ import annotations

import json
import time

import httpx

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import AgentAction, ExternalSignal
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


def test_provider_runtime_retries_transient_failure_and_records_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MODEL_PROVIDER_US", "openai")
    monkeypatch.setenv("TRENCHES_MODEL_NAME_US", "gpt-4.1")
    monkeypatch.setenv("TRENCHES_MODEL_API_KEY_ENV_US", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    attempts = {"count": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, json={"error": "upstream overloaded"})
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
                                                "type": "defend",
                                                "target": "us",
                                                "summary": "Defend shipping lanes after transient provider recovery.",
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

    runtime = ProviderDecisionRuntime(client=httpx.Client(transport=httpx.MockTransport(handler)), max_attempts=2)
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

    diagnostics = runtime.diagnostics(session.model_bindings)
    us_diagnostics = next(entry for entry in diagnostics if entry.agent_id == "us")

    assert attempts["count"] == 2
    assert actions["us"].type == "defend"
    assert actions["us"].metadata["provider_attempts"] == 2
    assert us_diagnostics.status == "healthy"
    assert us_diagnostics.request_count == 1
    assert us_diagnostics.success_count == 1
    assert us_diagnostics.error_count == 0
    assert us_diagnostics.consecutive_failures == 0
    assert us_diagnostics.last_success_at is not None
    assert us_diagnostics.last_error is None


def test_provider_runtime_diagnostics_capture_terminal_failure(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MODEL_PROVIDER_US", "openai")
    monkeypatch.setenv("TRENCHES_MODEL_NAME_US", "gpt-4.1")
    monkeypatch.setenv("TRENCHES_MODEL_API_KEY_ENV_US", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "upstream overloaded"})

    runtime = ProviderDecisionRuntime(client=httpx.Client(transport=httpx.MockTransport(handler)), max_attempts=2)
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

    diagnostics = runtime.diagnostics(session.model_bindings)
    us_diagnostics = next(entry for entry in diagnostics if entry.agent_id == "us")

    assert actions["us"].metadata["mode"] == "heuristic_fallback"
    assert "provider returned HTTP 503" in actions["us"].metadata["provider_error"]
    assert us_diagnostics.status == "degraded"
    assert us_diagnostics.request_count == 1
    assert us_diagnostics.success_count == 0
    assert us_diagnostics.error_count == 1
    assert us_diagnostics.consecutive_failures == 1
    assert us_diagnostics.last_error is not None
    assert "provider returned HTTP 503" in us_diagnostics.last_error
    assert us_diagnostics.last_error_at is not None


def test_huggingface_provider_uses_router_endpoint_and_hf_token(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MODEL_PROVIDER_US", "huggingface")
    monkeypatch.setenv("TRENCHES_MODEL_NAME_US", "openai/gpt-oss-120b")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.delenv("TRENCHES_MODEL_API_KEY_ENV_US", raising=False)
    monkeypatch.delenv("TRENCHES_HF_ROUTING_POLICY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "router.huggingface.co"
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer hf-test-token"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "openai/gpt-oss-120b:fastest"
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
                                                "type": "intel_query",
                                                "summary": "Query more shipping-route intelligence before changing posture.",
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

    assert session.model_bindings["us"].provider == "huggingface"
    assert session.model_bindings["us"].api_key_env == "HF_TOKEN"
    assert actions["us"].type == "intel_query"
    assert actions["us"].metadata["provider"] == "huggingface"


def test_vllm_provider_disables_thinking_and_parses_wrapped_tool_arguments(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MODEL_PROVIDER_US", "vllm")
    monkeypatch.setenv("TRENCHES_MODEL_NAME_US", "AlazarM/trenches-us-qwen3-8b-real")
    monkeypatch.setenv("TRENCHES_MODEL_BASE_URL_US", "https://trendsweep--trenches-us.modal.run/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
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
                                        "arguments": (
                                            "<think>\nReason privately.\n</think>\n"
                                            '{"type":"defend","target":"gulf","summary":"Defend shipping lanes with immediate naval escort."}'
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

    assert actions["us"].type == "defend"
    assert actions["us"].target == "gulf"
    assert actions["us"].metadata["provider"] == "vllm"


def test_provider_runtime_parses_qwen_tool_call_wrapper_text(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MODEL_PROVIDER_US", "vllm")
    monkeypatch.setenv("TRENCHES_MODEL_NAME_US", "AlazarM/trenches-us-qwen3-8b-real")
    monkeypatch.setenv("TRENCHES_MODEL_BASE_URL_US", "https://trendsweep--trenches-us.modal.run/v1")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<tool_call>\n"
                                '{"name":"emit_action","arguments":{"type":"negotiate","target":"iran","summary":"Negotiate deconfliction before military escalation."}}'
                                "\n</tool_call>"
                            )
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
    assert actions["us"].target == "iran"


def test_resolve_policy_actions_parallelizes_provider_resolution(monkeypatch) -> None:
    env = FogOfWarDiplomacyEnv(source_harvester=SourceHarvester(auto_start=False))
    session = env.create_session(seed=7)
    agent_ids = ["us", "israel", "iran"]

    for agent_id in agent_ids:
        session.model_bindings[agent_id].ready_for_inference = True

    def fake_resolve_provider_action(self, current_session, agent_id, signals):
        time.sleep(0.15)
        return (
            AgentAction(
                actor=agent_id,
                type="hold",
                summary=f"{agent_id} holds while awaiting more clarity.",
            ),
            None,
        )

    monkeypatch.setattr(FogOfWarDiplomacyEnv, "_resolve_provider_action", fake_resolve_provider_action)

    started = time.perf_counter()
    actions = env.resolve_policy_actions(session, [], agent_ids=agent_ids)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.35
    assert set(actions) == set(agent_ids)
