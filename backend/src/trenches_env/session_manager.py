from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from trenches_env.benchmark_runner import ScenarioBenchmarkRunner
from trenches_env.rl import DEFAULT_TRAINING_STAGE
from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import (
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    IngestNewsRequest,
    IngestNewsResponse,
    LiveControlRequest,
    ProviderDiagnosticsResponse,
    ReactionLogEntry,
    ScenarioSummary,
    SessionState,
    SourceMonitorReport,
    StepSessionRequest,
    StepSessionResponse,
)
from trenches_env.source_ingestion import SourceHarvester


class SessionManager:
    def __init__(self, env: FogOfWarDiplomacyEnv | None = None) -> None:
        self.env = env or FogOfWarDiplomacyEnv()
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()
        self._background_tick_seconds = 1.0
        self._background_stop = threading.Event()
        self._background_thread: threading.Thread | None = None

    def start_background_runner(self, tick_interval_seconds: float | None = None) -> None:
        with self._lock:
            if tick_interval_seconds is not None:
                self._background_tick_seconds = max(0.05, tick_interval_seconds)
            if self._background_thread is not None and self._background_thread.is_alive():
                return
            self._background_stop.clear()
            self._background_thread = threading.Thread(
                target=self._run_background_loop,
                name="trenches-session-manager",
                daemon=True,
            )
            self._background_thread.start()

    def stop_background_runner(self) -> None:
        self._background_stop.set()
        thread = self._background_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self._background_tick_seconds * 2.0))
        self._background_thread = None

    def shutdown(self) -> None:
        self.stop_background_runner()
        self.env.shutdown()

    def create_session(
        self,
        seed: int | None = None,
        training_stage: str = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
        scenario_id: str | None = None,
    ) -> SessionState:
        with self._lock:
            session = self.env.create_session(
                seed=seed,
                training_stage=training_stage,
                max_turns=max_turns,
                scenario_id=scenario_id,
            )
            self._sessions[session.session_id] = session
            return session

    def reset_session(
        self,
        session_id: str,
        seed: int | None = None,
        training_stage: str = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
        scenario_id: str | None = None,
    ) -> SessionState:
        with self._lock:
            self._require_session(session_id)
            session = self.env.reset_session(
                session_id=session_id,
                seed=seed,
                training_stage=training_stage,
                max_turns=max_turns,
                scenario_id=scenario_id,
            )
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            session = self._require_session(session_id)
            if session.live.enabled and session.live.auto_step:
                refreshed = self.env.maybe_auto_step_live_session(session)
            else:
                refreshed = self.env.refresh_session_sources(session)
            self._sessions[session_id] = refreshed
            return refreshed

    def set_live_mode(self, session_id: str, request: LiveControlRequest) -> SessionState:
        with self._lock:
            current = self._require_session(session_id)
            updated = self.env.configure_live_session(current, request)
            self._sessions[session_id] = updated
            return updated

    def step_session(self, session_id: str, request: StepSessionRequest) -> StepSessionResponse:
        with self._lock:
            current = self._require_session(session_id)
            result = self.env.step_session(current, request)
            self._sessions[session_id] = result.session
            return result

    def ingest_news(self, session_id: str, request: IngestNewsRequest) -> IngestNewsResponse:
        with self._lock:
            if not request.signals:
                raise ValueError("At least one external signal is required.")
            current = self._require_session(session_id)
            refreshed = self.env.refresh_session_sources(current)
            actions = self.env.resolve_policy_actions(
                refreshed,
                request.signals,
                agent_ids=request.agent_ids or None,
            )
            result = self.env.step_session(
                refreshed,
                StepSessionRequest(actions=actions, external_signals=request.signals),
            )
            self._sessions[session_id] = result.session
            reaction: ReactionLogEntry | None = result.session.reaction_log[-1] if result.session.reaction_log else None
            return IngestNewsResponse(
                session=result.session,
                oversight=result.oversight,
                reaction=reaction,
                done=result.done,
            )

    def refresh_session_sources(self, session_id: str, force: bool = False) -> SessionState:
        with self._lock:
            current = self._require_session(session_id)
            refreshed = self.env.refresh_session_sources(current, force=force)
            self._sessions[session_id] = refreshed
            return refreshed

    def source_monitor(self, session_id: str) -> SourceMonitorReport:
        with self._lock:
            current = self._require_session(session_id)
            refreshed = self.env.refresh_session_sources(current)
            self._sessions[session_id] = refreshed
            return self.env.source_monitor(refreshed)

    def reaction_log(self, session_id: str) -> list[ReactionLogEntry]:
        with self._lock:
            current = self._require_session(session_id)
            return [entry.model_copy(deep=True) for entry in current.reaction_log]

    def provider_diagnostics(self, session_id: str) -> ProviderDiagnosticsResponse:
        with self._lock:
            current = self._require_session(session_id)
            refreshed = self.env.refresh_session_sources(current)
            self._sessions[session_id] = refreshed
            return self.env.provider_diagnostics(refreshed)

    def list_scenarios(self) -> list[ScenarioSummary]:
        return [
            ScenarioSummary(
                id=scenario.id,
                name=scenario.name,
                description=scenario.description,
                tags=list(scenario.tags),
                benchmark_turns=scenario.benchmark_turns,
                benchmark_enabled=scenario.benchmark_enabled,
            )
            for scenario in self.env.list_scenarios()
        ]

    def run_benchmark(self, request: BenchmarkRunRequest) -> BenchmarkRunResponse:
        runner = ScenarioBenchmarkRunner(
            env_factory=lambda: FogOfWarDiplomacyEnv(source_harvester=SourceHarvester(auto_start=False))
        )
        return runner.run(request)

    def _run_background_loop(self) -> None:
        while not self._background_stop.is_set():
            try:
                self._tick_live_sessions()
            except Exception:
                pass
            self._background_stop.wait(self._background_tick_seconds)

    def _tick_live_sessions(self) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if not self._session_needs_live_tick(session, now):
                    continue
                self._sessions[session_id] = self.env.maybe_auto_step_live_session(session)

    @staticmethod
    def _session_needs_live_tick(session: SessionState, now: datetime) -> bool:
        if not session.live.enabled or not session.live.auto_step:
            return False
        if session.live.last_auto_step_at is None:
            return True
        interval = timedelta(milliseconds=max(session.live.poll_interval_ms, 1_000))
        return now - session.live.last_auto_step_at >= interval

    def _require_session(self, session_id: str) -> SessionState:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session
