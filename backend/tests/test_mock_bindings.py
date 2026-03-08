"""Tests for the mock provider bindings."""

from __future__ import annotations

from trenches_env.agents import AGENT_IDS
from trenches_env.model_runtime import build_entity_model_bindings


def test_mock_disabled_returns_normal_bindings(monkeypatch) -> None:
    monkeypatch.delenv("TRENCHES_MOCK_MODELS", raising=False)
    bindings = build_entity_model_bindings()
    # Without any TRENCHES_MODEL_PROVIDER_* set, all should be unconfigured
    for agent_id in AGENT_IDS:
        assert bindings[agent_id].provider == "none"
        assert bindings[agent_id].configured is False


def test_mock_enabled_returns_openrouter_bindings(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MOCK_MODELS", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.delenv("TRENCHES_MOCK_MODEL", raising=False)
    bindings = build_entity_model_bindings()
    model_names = set()
    for agent_id in AGENT_IDS:
        assert bindings[agent_id].provider == "openrouter"
        assert bindings[agent_id].configured is True
        assert bindings[agent_id].ready_for_inference is True
        assert bindings[agent_id].base_url == "https://openrouter.ai/api/v1"
        assert bindings[agent_id].api_key_env == "OPENROUTER_API_KEY"
        model_names.add(bindings[agent_id].model_name)
    # Each entity should have a distinct model
    assert len(model_names) == 6


def test_mock_model_override(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MOCK_MODELS", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("TRENCHES_MOCK_MODEL", "anthropic/claude-3.5-haiku")
    bindings = build_entity_model_bindings()
    for agent_id in AGENT_IDS:
        assert bindings[agent_id].model_name == "anthropic/claude-3.5-haiku"


def test_mock_without_api_key_sets_fallback(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MOCK_MODELS", "true")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    bindings = build_entity_model_bindings()
    for agent_id in AGENT_IDS:
        assert bindings[agent_id].provider == "openrouter"
        assert bindings[agent_id].configured is True
        assert bindings[agent_id].ready_for_inference is False
        assert bindings[agent_id].decision_mode == "heuristic_fallback"


def test_mock_status_diagnostic(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MOCK_MODELS", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    from trenches_env.mock.config import mock_status
    status = mock_status()
    assert status["mock_enabled"] is True
    assert status["api_key_present"] is True
    assert len(status["entities"]) == 6
