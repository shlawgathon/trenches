from __future__ import annotations

from trenches_env.rl import DEFAULT_TRAINING_STAGE
from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import (
    LiveControlRequest,
    SessionState,
    SourceMonitorReport,
    StepSessionRequest,
    StepSessionResponse,
)


class SessionManager:
    def __init__(self, env: FogOfWarDiplomacyEnv | None = None) -> None:
        self.env = env or FogOfWarDiplomacyEnv()
        self._sessions: dict[str, SessionState] = {}

    def create_session(
        self,
        seed: int | None = None,
        training_stage: str = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
    ) -> SessionState:
        session = self.env.create_session(seed=seed, training_stage=training_stage, max_turns=max_turns)
        self._sessions[session.session_id] = session
        return session

    def reset_session(
        self,
        session_id: str,
        seed: int | None = None,
        training_stage: str = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
    ) -> SessionState:
        self._require_session(session_id)
        session = self.env.reset_session(
            session_id=session_id,
            seed=seed,
            training_stage=training_stage,
            max_turns=max_turns,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionState:
        session = self._require_session(session_id)
        if session.live.enabled and session.live.auto_step:
            refreshed = self.env.maybe_auto_step_live_session(session)
        else:
            refreshed = self.env.refresh_session_sources(session)
        self._sessions[session_id] = refreshed
        return refreshed

    def set_live_mode(self, session_id: str, request: LiveControlRequest) -> SessionState:
        current = self._require_session(session_id)
        updated = self.env.configure_live_session(current, request)
        self._sessions[session_id] = updated
        return updated

    def step_session(self, session_id: str, request: StepSessionRequest) -> StepSessionResponse:
        current = self._require_session(session_id)
        result = self.env.step_session(current, request)
        self._sessions[session_id] = result.session
        return result

    def refresh_session_sources(self, session_id: str, force: bool = False) -> SessionState:
        current = self._require_session(session_id)
        refreshed = self.env.refresh_session_sources(current, force=force)
        self._sessions[session_id] = refreshed
        return refreshed

    def source_monitor(self, session_id: str) -> SourceMonitorReport:
        current = self._require_session(session_id)
        refreshed = self.env.refresh_session_sources(current)
        self._sessions[session_id] = refreshed
        return self.env.source_monitor(refreshed)

    def _require_session(self, session_id: str) -> SessionState:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session
