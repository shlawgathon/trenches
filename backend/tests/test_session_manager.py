from trenches_env.models import AgentAction, LiveControlRequest, StepSessionRequest
from trenches_env.session_manager import SessionManager
from trenches_env.source_bundles import AGENT_LIVE_SOURCE_BUNDLES


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
