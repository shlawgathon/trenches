from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.model_runtime import build_entity_model_bindings
from trenches_env.models import (
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    CreateSessionRequest,
    IngestNewsRequest,
    IngestNewsResponse,
    LiveControlRequest,
    ProviderDiagnosticsResponse,
    ReactionLogEntry,
    ResetEnvRequest,
    ResetEnvResponse,
    ResetSessionRequest,
    ScenarioSummary,
    SessionState,
    SourceMonitorReport,
    StepEnvRequest,
    StepEnvResponse,
    StepSessionRequest,
    StepSessionResponse,
)
from trenches_env.openenv_adapter import (
    OPENENV_CORE_AVAILABLE,
    OpenEnvAdapter,
    TrenchesOpenEnvEnvironment,
    create_openenv_fastapi_app,
)
from trenches_env.session_manager import SessionManager
from trenches_env.source_ingestion import SourceHarvester

DEFAULT_LOCAL_DEV_CORS_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"
logger = logging.getLogger("trenches")
logger.setLevel(logging.INFO)


def _parse_csv_env(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _resolve_cors_settings() -> dict[str, Any]:
    allow_origins = _parse_csv_env(os.getenv("TRENCHES_CORS_ALLOW_ORIGINS"))
    allow_origin_regex = os.getenv("TRENCHES_CORS_ALLOW_ORIGIN_REGEX")

    if "*" in allow_origins:
        return {
            "allow_origins": ["*"],
            "allow_origin_regex": None,
            # Browsers reject wildcard origins when credentials are enabled.
            "allow_credentials": False,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

    if not allow_origins and not allow_origin_regex:
        allow_origin_regex = DEFAULT_LOCAL_DEV_CORS_ORIGIN_REGEX

    allow_credentials = os.getenv("TRENCHES_CORS_ALLOW_CREDENTIALS", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return {
        "allow_origins": allow_origins,
        "allow_origin_regex": allow_origin_regex,
        "allow_credentials": allow_credentials,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


def _build_env(*, live_source_auto_start: bool, source_warm_start: bool) -> FogOfWarDiplomacyEnv:
    import os
    if os.getenv("TRENCHES_DISABLE_RSS") == "1":
        env = FogOfWarDiplomacyEnv(source_harvester=None)
    else:
        env = FogOfWarDiplomacyEnv(
            source_harvester=SourceHarvester(auto_start=live_source_auto_start),
        )
    if source_warm_start:
        env.enable_source_warm_start()
    return env


def create_app(
    session_manager: SessionManager | None = None,
    *,
    live_source_auto_start: bool = False,
    source_warm_start: bool = True,
) -> FastAPI:
    manager = session_manager or SessionManager(
        env=_build_env(
            live_source_auto_start=live_source_auto_start,
            source_warm_start=source_warm_start,
        )
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            logger.info("backend.start live_source_auto_start=%s source_warm_start=%s", live_source_auto_start, source_warm_start)
            manager.start_background_runner()
            yield
        finally:
            logger.info("backend.stop")
            manager.shutdown()

    app = FastAPI(title="Trenches OpenEnv Backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, **_resolve_cors_settings())
    runtime = OpenEnvAdapter(session_manager=manager)
    openenv_app = create_openenv_fastapi_app(
        lambda: TrenchesOpenEnvEnvironment(
            env=_build_env(
                live_source_auto_start=live_source_auto_start,
                source_warm_start=source_warm_start,
            )
        )
    )
    if openenv_app is not None:
        app.mount("/openenv", openenv_app)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        cors_settings = _resolve_cors_settings()
        return {
            "model_bindings": {
                agent_id: binding.model_dump(mode="json")
                for agent_id, binding in build_entity_model_bindings().items()
            },
            "session_api": True,
            "legacy_openenv_tuple_api": True,
            "native_openenv_api": OPENENV_CORE_AVAILABLE,
            "native_openenv_base_path": "/openenv" if OPENENV_CORE_AVAILABLE else None,
            "cors": {
                "allow_origins": cors_settings["allow_origins"],
                "allow_origin_regex": cors_settings["allow_origin_regex"],
                "allow_credentials": cors_settings["allow_credentials"],
            },
        }

    @app.post("/sessions", response_model=SessionState)
    async def create_session(request: CreateSessionRequest) -> SessionState:
        return manager.create_session(
            seed=request.seed,
            training_agent=request.training_agent,
            training_stage=request.training_stage,
            max_turns=request.max_turns,
            scenario_id=request.scenario_id,
            replay_id=request.replay_id,
            replay_start_index=request.replay_start_index,
        )

    @app.post("/sessions/{session_id}/reset", response_model=SessionState)
    async def reset_session(session_id: str, request: ResetSessionRequest) -> SessionState:
        try:
            return manager.reset_session(
                session_id=session_id,
                seed=request.seed,
                training_agent=request.training_agent,
                training_stage=request.training_stage,
                max_turns=request.max_turns,
                scenario_id=request.scenario_id,
                replay_id=request.replay_id,
                replay_start_index=request.replay_start_index,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/scenarios", response_model=list[ScenarioSummary])
    async def list_scenarios() -> list[ScenarioSummary]:
        return manager.list_scenarios()

    @app.post("/benchmarks/run", response_model=BenchmarkRunResponse)
    async def run_benchmark(request: BenchmarkRunRequest) -> BenchmarkRunResponse:
        try:
            return manager.run_benchmark(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}", response_model=SessionState)
    async def get_session(session_id: str) -> SessionState:
        try:
            return manager.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}") from exc

    @app.post("/sessions/{session_id}/sources/refresh", response_model=SessionState)
    async def refresh_session_sources(session_id: str) -> SessionState:
        try:
            return manager.refresh_session_sources(session_id=session_id, force=True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}") from exc

    @app.get("/sessions/{session_id}/sources/monitor", response_model=SourceMonitorReport)
    async def source_monitor(session_id: str) -> SourceMonitorReport:
        try:
            return manager.source_monitor(session_id=session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}") from exc

    @app.get("/sessions/{session_id}/reactions", response_model=list[ReactionLogEntry])
    async def reaction_log(session_id: str) -> list[ReactionLogEntry]:
        try:
            return manager.reaction_log(session_id=session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}") from exc

    @app.get("/sessions/{session_id}/providers/diagnostics", response_model=ProviderDiagnosticsResponse)
    async def provider_diagnostics(session_id: str) -> ProviderDiagnosticsResponse:
        try:
            return manager.provider_diagnostics(session_id=session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}") from exc

    @app.post("/sessions/{session_id}/news", response_model=IngestNewsResponse)
    async def ingest_news(session_id: str, request: IngestNewsRequest) -> IngestNewsResponse:
        try:
            return manager.ingest_news(session_id=session_id, request=request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/live", response_model=SessionState)
    async def set_live_mode(session_id: str, request: LiveControlRequest) -> SessionState:
        try:
            return manager.set_live_mode(session_id=session_id, request=request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/step", response_model=StepSessionResponse)
    async def step_session(session_id: str, request: StepSessionRequest) -> StepSessionResponse:
        try:
            return manager.step_session(session_id=session_id, request=request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/reset", response_model=ResetEnvResponse)
    async def reset_env(request: ResetEnvRequest) -> ResetEnvResponse:
        observations, info = runtime.reset(
            seed=request.seed,
            training_stage=request.training_stage,
            max_turns=request.max_turns,
            scenario_id=request.scenario_id,
            replay_id=request.replay_id,
            replay_start_index=request.replay_start_index,
        )
        return ResetEnvResponse(observations=observations, info=info)

    @app.post("/step", response_model=StepEnvResponse)
    async def step_env(request: StepEnvRequest) -> StepEnvResponse:
        try:
            observations, rewards, terminated, truncated, info = runtime.step(
                actions=request.actions,
                predictions=request.predictions,
                external_signals=request.external_signals,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StepEnvResponse(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    @app.get("/state", response_model=SessionState)
    async def state_env() -> SessionState:
        session = runtime.state()
        if session is None:
            raise HTTPException(status_code=404, detail="No active OpenEnv runtime session.")
        return session

    return app


app = create_app()


def run() -> None:
    uvicorn.run("trenches_env.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
