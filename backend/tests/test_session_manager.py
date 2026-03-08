import time
from datetime import datetime, timedelta, timezone

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import (
    AgentAction,
    BenchmarkRunRequest,
    ExternalSignal,
    IngestNewsRequest,
    LiveControlRequest,
    StepSessionRequest,
)
from trenches_env.session_manager import SessionManager
from trenches_env.source_bundles import AGENT_LIVE_SOURCE_BUNDLES
from trenches_env.source_ingestion import SourceHarvester


class ShippingFeedFetcher:
    def fetch(self, _: str) -> tuple[str, str]:
        return (
            """
            <rss>
              <channel>
                <title>Maritime Watch</title>
                <item>
                  <title>Shipping risk rises in Hormuz after drone intercept near tanker lanes</title>
                </item>
              </channel>
            </rss>
            """,
            "application/rss+xml",
        )


class SlowHtmlFetcher:
    def fetch(self, url: str) -> tuple[str, str]:
        time.sleep(0.15)
        return (
            f"""
            <html>
              <head>
                <title>Snapshot for {url}</title>
                <meta name="description" content="Background hydration snapshot for {url}" />
              </head>
              <body>
                <h1>Snapshot for {url}</h1>
              </body>
            </html>
            """,
            "text/html; charset=utf-8",
        )


def build_live_manager() -> SessionManager:
    harvester = SourceHarvester(fetcher=ShippingFeedFetcher(), auto_start=False)
    env = FogOfWarDiplomacyEnv(source_harvester=harvester)
    return SessionManager(env=env)


def test_session_lifecycle() -> None:
    manager = SessionManager()
    session = manager.create_session(seed=7)

    assert session.world.turn == 0
    assert session.session_id
    assert session.episode.max_turns == 1000

    live_session = manager.set_live_mode(
        session.session_id,
        LiveControlRequest(enabled=True, auto_step=False, poll_interval_ms=15_000),
    )

    assert live_session.live.enabled is True
    assert live_session.live.source_queue_sizes["us"] == len(AGENT_LIVE_SOURCE_BUNDLES["us"])
    assert live_session.live.hydration.total >= live_session.live.hydration.ready
    assert live_session.live.hydration.pending >= 0
    assert live_session.live.hydration.phase in {"seed", "background", "steady"}

    response = manager.step_session(
        session.session_id,
        StepSessionRequest(
            actions={
                "us": AgentAction(
                    actor="us",
                    type="negotiate",
                    target="gulf",
                    summary="Offer deconfliction and shipping guarantees.",
                ),
                "oversight": AgentAction(
                    actor="oversight",
                    type="oversight_review",
                    summary="Monitor for escalation drift.",
                ),
            }
        ),
    )

    assert response.session.world.turn == 1
    assert "gulf" in response.session.world.coalition_graph["us"]
    assert "us" in response.session.observations
    assert response.session.observations["us"].training_source_bundle
    assert response.session.observations["us"].live_source_bundle
    assert response.session.recent_traces
    assert response.session.recent_traces[-1].turn == 1
    assert response.session.model_bindings["us"].action_tools
    assert response.session.model_bindings["us"].decision_mode == "heuristic_fallback"
    assert response.session.action_log
    assert {entry.actor for entry in response.session.action_log[-2:]} == {"us", "oversight"}


def test_stage_1_disables_live_mode() -> None:
    manager = SessionManager()
    session = manager.create_session(seed=7, training_stage="stage_1_dense")

    assert session.episode.training_stage == "stage_1_dense"
    assert session.episode.fog_of_war is False
    assert session.observations["us"].perceived_tension == session.world.tension_level

    try:
        manager.set_live_mode(
            session.session_id,
            LiveControlRequest(enabled=True, auto_step=False, poll_interval_ms=15_000),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("stage_1_dense sessions should reject live mode")


def test_session_manager_creates_named_scenarios() -> None:
    manager = SessionManager()
    session = manager.create_session(seed=7, scenario_id="shipping_crisis")

    assert session.episode.scenario_id == "shipping_crisis"
    assert session.episode.scenario_name == "Shipping Crisis"
    assert session.world.actor_state["gulf"]["shipping_continuity"] < 78.0


def test_source_monitor_report_is_available_for_sessions() -> None:
    manager = SessionManager()
    session = manager.create_session(seed=7)
    report = manager.source_monitor(session.session_id)

    assert report.session_id == session.session_id
    assert len(report.agents) == len(AGENT_LIVE_SOURCE_BUNDLES)
    assert report.summary.active_source_count > 0


def test_live_get_session_auto_steps_once_for_new_source_packets() -> None:
    manager = build_live_manager()
    session = manager.create_session(seed=7)
    manager.set_live_mode(
        session.session_id,
        LiveControlRequest(enabled=True, auto_step=True, poll_interval_ms=1_000),
    )

    first_live_tick = manager.get_session(session.session_id)

    assert first_live_tick.world.turn == 1
    assert first_live_tick.live.last_auto_step_at is not None
    assert first_live_tick.live.reacted_packet_fetched_at
    assert first_live_tick.recent_traces[-1].actions["gulf"].type == "defend"
    assert first_live_tick.recent_traces[-1].actions["oversight"].type == "oversight_review"

    for observation in first_live_tick.observations.values():
        for packet in observation.source_packets:
            if packet.fetched_at is not None:
                manager._sessions[session.session_id].live.reacted_packet_fetched_at[packet.source_id] = packet.fetched_at
    manager._sessions[session.session_id].live.last_auto_step_at = datetime.now(timezone.utc) - timedelta(seconds=2)
    second_live_tick = manager.get_session(session.session_id)

    assert second_live_tick.world.turn == 1


def test_live_get_session_auto_steps_without_new_source_packets() -> None:
    manager = SessionManager()
    session = manager.create_session(seed=7)
    manager.set_live_mode(
        session.session_id,
        LiveControlRequest(enabled=True, auto_step=True, poll_interval_ms=1_000),
    )

    live_tick = manager.get_session(session.session_id)

    assert live_tick.world.turn == 1
    assert live_tick.live.last_auto_step_at is not None
    assert live_tick.recent_traces
    assert live_tick.recent_traces[-1].turn == 1


def test_oversight_replaces_escalatory_actions_with_valid_overrides() -> None:
    manager = SessionManager()
    session = manager.create_session(seed=7)

    response = manager.step_session(
        session.session_id,
        StepSessionRequest(
            actions={
                "us": AgentAction(actor="us", type="strike", target="iran", summary="Strike escalation lane."),
                "israel": AgentAction(actor="israel", type="strike", target="hezbollah", summary="Strike escalation lane."),
                "iran": AgentAction(actor="iran", type="mobilize", summary="Mobilize escalation lane."),
                "hezbollah": AgentAction(actor="hezbollah", type="deceive", summary="Deception escalation lane."),
            }
        ),
    )

    assert response.oversight.triggered is True
    assert response.oversight.action_override
    for agent_id, override_action in response.oversight.action_override.items():
        assert override_action.type in {"hold", "negotiate", "defend", "intel_query"}
        assert response.session.recent_traces[-1].actions[agent_id].type == override_action.type
        assert response.session.world.last_actions


def test_background_runner_advances_live_sessions_without_dashboard_polling() -> None:
    manager = build_live_manager()
    session = manager.create_session(seed=7)
    manager.set_live_mode(
        session.session_id,
        LiveControlRequest(enabled=True, auto_step=True, poll_interval_ms=1_000),
    )
    manager.start_background_runner(tick_interval_seconds=0.05)

    try:
        deadline = time.time() + 1.5
        while time.time() < deadline:
            current = manager._sessions[session.session_id]
            if current.world.turn >= 1:
                break
            time.sleep(0.05)
        current = manager._sessions[session.session_id]
    finally:
        manager.stop_background_runner()

    assert current.world.turn >= 1
    assert current.live.last_auto_step_at is not None
    assert current.reaction_log


def test_get_session_returns_cached_snapshot_while_background_runner_hydrates() -> None:
    harvester = SourceHarvester(fetcher=SlowHtmlFetcher(), auto_start=False, batch_size=4)
    manager = SessionManager(env=FogOfWarDiplomacyEnv(source_harvester=harvester))
    session = manager.create_session(seed=7)
    manager.set_live_mode(
        session.session_id,
        LiveControlRequest(enabled=True, auto_step=False, poll_interval_ms=1_000),
    )
    manager.start_background_runner(tick_interval_seconds=0.01)

    try:
        time.sleep(0.03)
        started = time.perf_counter()
        current = manager.get_session(session.session_id)
        elapsed = time.perf_counter() - started
    finally:
        manager.stop_background_runner()

    assert elapsed < 0.1
    assert current.live.hydration.total > 0
    assert current.live.hydration.phase in {"seed", "background", "steady"}


def test_ingest_news_generates_structured_reaction_log() -> None:
    manager = SessionManager()
    session = manager.create_session(seed=7)

    response = manager.ingest_news(
        session.session_id,
        IngestNewsRequest(
            signals=[
                ExternalSignal(
                    source="wire-service",
                    headline="Shipping risk rises in Hormuz after reported drone intercept.",
                    region="gulf",
                    tags=["shipping", "attack"],
                    severity=0.76,
                )
            ],
            agent_ids=["us", "gulf", "oversight"],
        ),
    )

    assert response.session.world.turn == 1
    assert response.reaction is not None
    assert response.reaction.turn == 1
    assert response.reaction.signals[0].source == "wire-service"
    assert response.reaction.latent_event_ids
    assert {outcome.agent_id for outcome in response.reaction.actor_outcomes} == {"us", "gulf", "oversight"}
    assert all(outcome.action.metadata["mode"] in {"heuristic_fallback", "provider_inference"} for outcome in response.reaction.actor_outcomes)
    assert response.session.reaction_log[-1].event_id == response.reaction.event_id
    assert manager.reaction_log(session.session_id)[-1].event_id == response.reaction.event_id
    assert response.session.belief_state["gulf"].beliefs
    assert response.session.observations["gulf"].belief_brief


def test_provider_diagnostics_are_available_per_session() -> None:
    manager = SessionManager()
    session = manager.create_session(seed=7)

    diagnostics = manager.provider_diagnostics(session.session_id)
    us_diagnostics = next(entry for entry in diagnostics.agents if entry.agent_id == "us")

    assert us_diagnostics.agent_id == "us"
    assert us_diagnostics.status in {"idle", "fallback_only"}
    assert us_diagnostics.request_count == 0


def test_session_manager_lists_scenarios_and_runs_benchmarks() -> None:
    manager = SessionManager()

    scenarios = manager.list_scenarios()
    assert any(scenario.id == "shipping_crisis" for scenario in scenarios)

    result = manager.run_benchmark(
        BenchmarkRunRequest(
            scenario_ids=["shipping_crisis"],
            seed=9,
            steps_per_scenario=3,
        )
    )
    assert result.scenario_count == 1
    assert result.results[0].scenario_id == "shipping_crisis"
