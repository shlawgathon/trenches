from __future__ import annotations

from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.server import DEFAULT_LOCAL_DEV_CORS_ORIGIN_REGEX, create_app
from trenches_env.session_manager import SessionManager
from trenches_env.source_ingestion import SourceHarvester


def build_manager() -> SessionManager:
    env = FogOfWarDiplomacyEnv(source_harvester=SourceHarvester(auto_start=False))
    return SessionManager(env=env)


def _cors_kwargs(app) -> dict[str, object]:
    middleware = next(entry for entry in app.user_middleware if entry.cls is CORSMiddleware)
    return middleware.kwargs


def test_server_defaults_to_localhost_any_port_cors(monkeypatch) -> None:
    monkeypatch.delenv("TRENCHES_CORS_ALLOW_ORIGINS", raising=False)
    monkeypatch.delenv("TRENCHES_CORS_ALLOW_ORIGIN_REGEX", raising=False)
    monkeypatch.delenv("TRENCHES_CORS_ALLOW_CREDENTIALS", raising=False)

    app = create_app(session_manager=build_manager())
    cors = _cors_kwargs(app)

    assert cors["allow_origins"] == []
    assert cors["allow_origin_regex"] == DEFAULT_LOCAL_DEV_CORS_ORIGIN_REGEX
    assert cors["allow_credentials"] is True


def test_server_honors_explicit_cors_origin_list(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_CORS_ALLOW_ORIGINS", "https://dashboard.example.com, https://ops.example.com")
    monkeypatch.delenv("TRENCHES_CORS_ALLOW_ORIGIN_REGEX", raising=False)
    monkeypatch.setenv("TRENCHES_CORS_ALLOW_CREDENTIALS", "false")

    app = create_app(session_manager=build_manager())
    cors = _cors_kwargs(app)

    assert cors["allow_origins"] == ["https://dashboard.example.com", "https://ops.example.com"]
    assert cors["allow_origin_regex"] is None
    assert cors["allow_credentials"] is False


def test_server_disables_credentials_for_wildcard_cors(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_CORS_ALLOW_ORIGINS", "*")
    monkeypatch.delenv("TRENCHES_CORS_ALLOW_ORIGIN_REGEX", raising=False)
    monkeypatch.setenv("TRENCHES_CORS_ALLOW_CREDENTIALS", "true")

    app = create_app(session_manager=build_manager())
    cors = _cors_kwargs(app)

    assert cors["allow_origins"] == ["*"]
    assert cors["allow_origin_regex"] is None
    assert cors["allow_credentials"] is False


def test_server_exposes_scenarios_and_benchmark_endpoints() -> None:
    app = create_app(session_manager=build_manager())
    client = TestClient(app)

    scenarios_response = client.get("/scenarios")
    assert scenarios_response.status_code == 200
    scenarios = scenarios_response.json()
    assert any(scenario["id"] == "shipping_crisis" for scenario in scenarios)

    benchmark_response = client.post(
        "/benchmarks/run",
        json={
            "scenario_ids": ["shipping_crisis"],
            "seed": 21,
            "steps_per_scenario": 2,
        },
    )
    assert benchmark_response.status_code == 200
    benchmark = benchmark_response.json()
    assert benchmark["scenario_count"] == 1
    assert benchmark["results"][0]["scenario_id"] == "shipping_crisis"


def test_capabilities_expose_model_provider_bindings(monkeypatch) -> None:
    monkeypatch.setenv("TRENCHES_MODEL_PROVIDER_US", "openai")
    monkeypatch.setenv("TRENCHES_MODEL_NAME_US", "gpt-4.1")
    monkeypatch.setenv("TRENCHES_MODEL_API_KEY_ENV_US", "OPENAI_API_KEY")
    app = create_app(session_manager=build_manager())
    client = TestClient(app)

    response = client.get("/capabilities")
    assert response.status_code == 200
    capabilities = response.json()
    assert capabilities["model_bindings"]["us"]["configured"] is True
    assert capabilities["model_bindings"]["us"]["decision_mode"] == "provider_ready"
    assert "negotiate" in capabilities["model_bindings"]["us"]["action_tools"]


def test_server_ingests_news_and_exposes_reaction_log() -> None:
    app = create_app(session_manager=build_manager())
    client = TestClient(app)

    session_response = client.post("/sessions", json={"seed": 7})
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    ingest_response = client.post(
        f"/sessions/{session_id}/news",
        json={
            "signals": [
                {
                    "source": "wire-service",
                    "headline": "Shipping risk rises in Hormuz after reported drone intercept.",
                    "region": "gulf",
                    "tags": ["shipping", "attack"],
                    "severity": 0.76,
                }
            ],
            "agent_ids": ["us", "gulf", "oversight"],
        },
    )
    assert ingest_response.status_code == 200
    payload = ingest_response.json()
    assert payload["reaction"] is not None
    assert payload["reaction"]["signals"][0]["source"] == "wire-service"
    assert payload["reaction"]["latent_event_ids"]
    assert {outcome["agent_id"] for outcome in payload["reaction"]["actor_outcomes"]} == {"us", "gulf", "oversight"}

    log_response = client.get(f"/sessions/{session_id}/reactions")
    assert log_response.status_code == 200
    reactions = log_response.json()
    assert len(reactions) == 1
    assert reactions[0]["event_id"] == payload["reaction"]["event_id"]


def test_server_rejects_empty_news_ingest() -> None:
    app = create_app(session_manager=build_manager())
    client = TestClient(app)

    session_response = client.post("/sessions", json={"seed": 7})
    session_id = session_response.json()["session_id"]

    ingest_response = client.post(
        f"/sessions/{session_id}/news",
        json={"signals": []},
    )
    assert ingest_response.status_code == 400
    assert "At least one external signal is required." in ingest_response.json()["detail"]


def test_server_exposes_provider_diagnostics() -> None:
    app = create_app(session_manager=build_manager())
    client = TestClient(app)

    session_response = client.post("/sessions", json={"seed": 7})
    session_id = session_response.json()["session_id"]

    diagnostics_response = client.get(f"/sessions/{session_id}/providers/diagnostics")
    assert diagnostics_response.status_code == 200
    diagnostics = diagnostics_response.json()
    us_diagnostics = next(entry for entry in diagnostics["agents"] if entry["agent_id"] == "us")

    assert us_diagnostics["agent_id"] == "us"
    assert us_diagnostics["status"] in {"idle", "fallback_only"}
    assert us_diagnostics["request_count"] == 0
