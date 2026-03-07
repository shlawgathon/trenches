import time
from datetime import datetime, timedelta, timezone

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import AgentAction, LiveControlRequest, StepSessionRequest
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
