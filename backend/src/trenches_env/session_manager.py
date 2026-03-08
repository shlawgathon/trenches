from __future__ import annotations

import logging
import threading
from time import perf_counter

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

logger = logging.getLogger("trenches.session")


class SessionManager:
    def __init__(self, env: FogOfWarDiplomacyEnv | None = None) -> None:
        self.env = env or FogOfWarDiplomacyEnv()
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()
        self._session_locks: dict[str, threading.RLock] = {}
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
        training_agent: str = "us",
        training_stage: str = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
        scenario_id: str | None = None,
        replay_id: str | None = None,
        replay_start_index: int | None = None,
    ) -> SessionState:
        with self._lock:
            started_at = perf_counter()
            session = self.env.create_session(
                seed=seed,
                training_agent=training_agent,
                training_stage=training_stage,
                max_turns=max_turns,
                scenario_id=scenario_id,
                replay_id=replay_id,
                replay_start_index=replay_start_index,
            )
            self._sessions[session.session_id] = session
            self._session_locks.setdefault(session.session_id, threading.RLock())
            logger.info(
                "session.created id=%s seed=%s turn=%s stage=%s scenario=%s replay=%s live_capable=%s packets=%s assets=%s duration_ms=%.1f",
                session.session_id,
                session.seed,
                session.world.turn,
                session.episode.training_stage,
                session.episode.scenario_id,
                session.historical_replay.replay_id or "-",
                session.episode.live_mode_capable,
                _count_source_packets(session),
                _count_assets(session),
                (perf_counter() - started_at) * 1000.0,
            )
            return session

    def reset_session(
        self,
        session_id: str,
        seed: int | None = None,
        training_agent: str = "us",
        training_stage: str = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
        scenario_id: str | None = None,
        replay_id: str | None = None,
        replay_start_index: int | None = None,
    ) -> SessionState:
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            with self._lock:
                self._require_session(session_id)
            started_at = perf_counter()
            session = self.env.reset_session(
                session_id=session_id,
                seed=seed,
                training_agent=training_agent,
                training_stage=training_stage,
                max_turns=max_turns,
                scenario_id=scenario_id,
                replay_id=replay_id,
                replay_start_index=replay_start_index,
            )
            with self._lock:
                self._sessions[session_id] = session
                self._session_locks.setdefault(session_id, threading.RLock())
            logger.info(
                "session.reset id=%s seed=%s turn=%s stage=%s scenario=%s replay=%s duration_ms=%.1f",
                session_id,
                session.seed,
                session.world.turn,
                session.episode.training_stage,
                session.episode.scenario_id,
                session.historical_replay.replay_id or "-",
                (perf_counter() - started_at) * 1000.0,
            )
            return session

    def get_session(self, session_id: str) -> SessionState:
        if self._background_runner_active():
            with self._lock:
                return self._require_session(session_id)

        session_lock = self._get_session_lock(session_id)
        with session_lock:
            with self._lock:
                session = self._require_session(session_id)
            if session.live.enabled:
                if session.live.auto_step:
                    refreshed = self.env.maybe_auto_step_live_session(session)
                else:
                    refreshed = self.env.background_refresh_session(session)
            else:
                refreshed = self.env.refresh_session_sources(session)
            with self._lock:
                self._sessions[session_id] = refreshed
            return refreshed

    def set_live_mode(self, session_id: str, request: LiveControlRequest) -> SessionState:
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            with self._lock:
                current = self._require_session(session_id)
            started_at = perf_counter()
            updated = self.env.configure_live_session(current, request)
            with self._lock:
                self._sessions[session_id] = updated
            logger.info(
                "session.live id=%s enabled=%s auto_step=%s poll_ms=%s queue_total=%s packets=%s duration_ms=%.1f",
                session_id,
                updated.live.enabled,
                updated.live.auto_step,
                updated.live.poll_interval_ms,
                sum(updated.live.source_queue_sizes.values()),
                _count_source_packets(updated),
                (perf_counter() - started_at) * 1000.0,
            )
            return updated

    def step_session(self, session_id: str, request: StepSessionRequest) -> StepSessionResponse:
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            with self._lock:
                current = self._require_session(session_id)
            started_at = perf_counter()
            auto_resolved = False
            # Auto-resolve actions via model inference when none provided
            if not request.actions:
                signals = list(request.external_signals or [])
                actions = self.env.resolve_policy_actions(current, signals)
                request = StepSessionRequest(
                    actions=actions,
                    external_signals=signals,
                )
                auto_resolved = True
            result = self.env.step_session(current, request)
            with self._lock:
                self._sessions[session_id] = result.session
            logger.info(
                (
                    "session.step id=%s turn=%s->%s auto_actions=%s signals=%s actions=%s "
                    "oversight=%s tension=%.1f->%.1f market=%.1f oil=%.1f done=%s duration_ms=%.1f"
                ),
                session_id,
                current.world.turn,
                result.session.world.turn,
                auto_resolved,
                _summarize_signals(request.external_signals),
                _summarize_actions(request.actions),
                _summarize_oversight(result.oversight),
                current.world.tension_level,
                result.session.world.tension_level,
                result.session.world.market_stress,
                result.session.world.oil_pressure,
                result.done,
                (perf_counter() - started_at) * 1000.0,
            )
            return result

    def ingest_news(self, session_id: str, request: IngestNewsRequest) -> IngestNewsResponse:
        if not request.signals:
            raise ValueError("At least one external signal is required.")
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            with self._lock:
                current = self._require_session(session_id)
            started_at = perf_counter()
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
            with self._lock:
                self._sessions[session_id] = result.session
            reaction: ReactionLogEntry | None = result.session.reaction_log[-1] if result.session.reaction_log else None
            logger.info(
                (
                    "session.news id=%s signals=%s actions=%s oversight=%s reaction=%s "
                    "turn=%s tension=%.1f->%.1f done=%s duration_ms=%.1f"
                ),
                session_id,
                _summarize_signals(request.signals),
                _summarize_actions(actions),
                _summarize_oversight(result.oversight),
                _summarize_reaction(reaction),
                result.session.world.turn,
                current.world.tension_level,
                result.session.world.tension_level,
                result.done,
                (perf_counter() - started_at) * 1000.0,
            )
            return IngestNewsResponse(
                session=result.session,
                oversight=result.oversight,
                reaction=reaction,
                done=result.done,
            )

    def refresh_session_sources(self, session_id: str, force: bool = False) -> SessionState:
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            with self._lock:
                current = self._require_session(session_id)
            started_at = perf_counter()
            refreshed = self.env.refresh_session_sources(current, force=force)
            with self._lock:
                self._sessions[session_id] = refreshed
            logger.info(
                "session.sources id=%s force=%s packets=%s queue_total=%s duration_ms=%.1f",
                session_id,
                force,
                _summarize_packet_counts(refreshed),
                sum(refreshed.live.source_queue_sizes.values()),
                (perf_counter() - started_at) * 1000.0,
            )
            return refreshed

    def source_monitor(self, session_id: str) -> SourceMonitorReport:
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            with self._lock:
                current = self._require_session(session_id)
            refreshed = self.env.refresh_session_sources(current)
            with self._lock:
                self._sessions[session_id] = refreshed
            return self.env.source_monitor(refreshed)

    def reaction_log(self, session_id: str) -> list[ReactionLogEntry]:
        with self._lock:
            current = self._require_session(session_id)
            return [entry.model_copy(deep=True) for entry in current.reaction_log]

    def provider_diagnostics(self, session_id: str) -> ProviderDiagnosticsResponse:
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            with self._lock:
                current = self._require_session(session_id)
            refreshed = self.env.refresh_session_sources(current)
            with self._lock:
                self._sessions[session_id] = refreshed
            logger.info(
                "session.providers id=%s packets=%s",
                session_id,
                _summarize_packet_counts(refreshed),
            )
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
        with self._lock:
            session_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if self._session_needs_live_tick(session)
            ]

        for session_id in session_ids:
            session_lock = self._session_locks.get(session_id)
            if session_lock is None or not session_lock.acquire(blocking=False):
                continue
            try:
                with self._lock:
                    session = self._sessions.get(session_id)
                    if session is None:
                        continue
                started_at = perf_counter()
                updated = self.env.background_refresh_session(session)
                if updated.live.auto_step:
                    updated = self.env.maybe_auto_step_live_session(updated)
                with self._lock:
                    self._sessions[session_id] = updated
                if (
                    updated.world.turn != session.world.turn
                    or updated.live.hydration.pending != session.live.hydration.pending
                    or updated.live.hydration.ready != session.live.hydration.ready
                    or updated.live.hydration.error != session.live.hydration.error
                ):
                    logger.info(
                        (
                            "session.live_tick id=%s turn=%s->%s packets=%s queue_total=%s "
                            "hydration=%s/%s/%s duration_ms=%.1f"
                        ),
                        session_id,
                        session.world.turn,
                        updated.world.turn,
                        _summarize_packet_counts(updated),
                        sum(updated.live.source_queue_sizes.values()),
                        updated.live.hydration.ready,
                        updated.live.hydration.pending,
                        updated.live.hydration.error,
                        (perf_counter() - started_at) * 1000.0,
                    )
            finally:
                session_lock.release()

    @staticmethod
    def _session_needs_live_tick(session: SessionState) -> bool:
        return session.live.enabled

    def _require_session(self, session_id: str) -> SessionState:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def _get_session_lock(self, session_id: str) -> threading.RLock:
        with self._lock:
            self._require_session(session_id)
            return self._session_locks.setdefault(session_id, threading.RLock())

    def _background_runner_active(self) -> bool:
        thread = self._background_thread
        return thread is not None and thread.is_alive()


def _count_source_packets(session: SessionState) -> int:
    return sum(len(observation.source_packets) for observation in session.observations.values())


def _count_assets(session: SessionState) -> int:
    return sum(len(assets) for assets in session.world.asset_state.values())


def _summarize_packet_counts(session: SessionState) -> str:
    total = 0
    ok = 0
    pending = 0
    error = 0
    for observation in session.observations.values():
        for packet in observation.source_packets:
            total += 1
            if packet.status == "ok":
                ok += 1
            elif packet.status == "pending":
                pending += 1
            elif packet.status == "error":
                error += 1
    return f"total={total},ok={ok},pending={pending},error={error}"


def _summarize_actions(actions: dict[str, object]) -> str:
    if not actions:
        return "-"
    parts: list[str] = []
    for agent_id in sorted(actions):
        action = actions[agent_id]
        action_type = getattr(action, "type", "unknown")
        target = getattr(action, "target", None)
        metadata = getattr(action, "metadata", {}) or {}
        mode = metadata.get("mode")
        fragment = f"{agent_id}:{action_type}"
        if target:
            fragment = f"{fragment}->{target}"
        if mode:
            fragment = f"{fragment}[{mode}]"
        parts.append(fragment)
    return ",".join(parts)


def _summarize_signals(signals: list[object]) -> str:
    if not signals:
        return "-"
    parts: list[str] = []
    for signal in signals[:3]:
        source = getattr(signal, "source", "signal")
        severity = getattr(signal, "severity", 0.0)
        headline = getattr(signal, "headline", "")
        clipped = " ".join(str(headline).split())[:64]
        parts.append(f"{source}:{severity:.2f}:{clipped}")
    if len(signals) > 3:
        parts.append(f"+{len(signals) - 3}more")
    return " | ".join(parts)


def _summarize_oversight(oversight: object) -> str:
    triggered = getattr(oversight, "triggered", False)
    if not triggered:
        return "off"
    risk_score = getattr(oversight, "risk_score", 0.0)
    affected_agents = getattr(oversight, "affected_agents", []) or []
    return f"on:risk={risk_score:.2f}:agents={','.join(affected_agents) or '-'}"


def _summarize_reaction(reaction: ReactionLogEntry | None) -> str:
    if reaction is None:
        return "-"
    return (
        f"event={reaction.event_id}"
        f",latent={len(reaction.latent_event_ids)}"
        f",actors={len(reaction.actor_outcomes)}"
    )
