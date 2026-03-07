from __future__ import annotations

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
    StepEnvRequest,
    StepEnvResponse,
    StepSessionRequest,
    StepSessionResponse,
)
from trenches_env.openenv_adapter import OpenEnvAdapter
from trenches_env.session_manager import SessionManager
from trenches_env.source_ingestion import SourceHarvester


def create_app(session_manager: SessionManager | None = None) -> FastAPI:
    app = FastAPI(title="Trenches OpenEnv Backend", version="0.1.0")
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
    manager = session_manager or SessionManager(
        env=FogOfWarDiplomacyEnv(
            source_harvester=SourceHarvester(auto_start=True),
        )
    )
    runtime = OpenEnvAdapter(session_manager=manager)

    @app.on_event("shutdown")
    async def shutdown_source_harvester() -> None:
        manager.env.shutdown()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

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
        observations, rewards, terminated, truncated, info = runtime.step(
            actions=request.actions,
            external_signals=request.external_signals,
        )
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
