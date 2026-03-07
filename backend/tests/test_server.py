from __future__ import annotations

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
