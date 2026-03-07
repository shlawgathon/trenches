from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import (
    CreateSessionRequest,
    LiveControlRequest,
    ResetEnvRequest,
    ResetEnvResponse,
    ResetSessionRequest,
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


def create_app(session_manager: SessionManager | None = None) -> FastAPI:
    manager = session_manager or SessionManager(
        env=FogOfWarDiplomacyEnv(
            source_harvester=SourceHarvester(auto_start=True),
        ).enable_source_warm_start()
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            manager.env.shutdown()

    app = FastAPI(title="Trenches OpenEnv Backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    runtime = OpenEnvAdapter(session_manager=manager)
    openenv_app = create_openenv_fastapi_app(
        lambda: TrenchesOpenEnvEnvironment(
            env=FogOfWarDiplomacyEnv(
                source_harvester=SourceHarvester(auto_start=False),
            ).enable_source_warm_start()
        )
    )
    if openenv_app is not None:
        app.mount("/openenv", openenv_app)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {
            "session_api": True,
            "legacy_openenv_tuple_api": True,
            "native_openenv_api": OPENENV_CORE_AVAILABLE,
            "native_openenv_base_path": "/openenv" if OPENENV_CORE_AVAILABLE else None,
        }

    @app.post("/sessions", response_model=SessionState)
    async def create_session(request: CreateSessionRequest) -> SessionState:
        return manager.create_session(
            seed=request.seed,
            training_stage=request.training_stage,
            max_turns=request.max_turns,
        )

    @app.post("/sessions/{session_id}/reset", response_model=SessionState)
    async def reset_session(session_id: str, request: ResetSessionRequest) -> SessionState:
        try:
            return manager.reset_session(
                session_id=session_id,
                seed=request.seed,
                training_stage=request.training_stage,
                max_turns=request.max_turns,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}") from exc

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
        )
        return ResetEnvResponse(observations=observations, info=info)

    @app.post("/step", response_model=StepEnvResponse)
    async def step_env(request: StepEnvRequest) -> StepEnvResponse:
        try:
            observations, rewards, terminated, truncated, info = runtime.step(
                actions=request.actions,
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
