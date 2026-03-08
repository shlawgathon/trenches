import pytest

from trenches_env.models import AgentAction, ExternalSignal, Prediction
from trenches_env.openenv_client import TrenchesEnvClient
from trenches_env.openenv_adapter import (
    OPENENV_CORE_AVAILABLE,
    OpenEnvAdapter,
    TrenchesOpenEnvAction,
    TrenchesOpenEnvEnvironment,
    create_openenv_fastapi_app,
)


def test_openenv_adapter_reset_and_step() -> None:
    runtime = OpenEnvAdapter()
    observations, info = runtime.reset(seed=11, training_stage="stage_3_sparse", max_turns=12)

    assert "us" in observations
    assert info["episode"]["training_stage"] == "stage_3_sparse"
    assert info["episode"]["max_turns"] == 12

    next_observations, rewards, terminated, truncated, next_info = runtime.step(
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
                summary="Review escalation drift and intervention triggers.",
            ),
        },
        external_signals=[
            ExternalSignal(
                source="training-sim",
                headline="Shipping risk rises near Hormuz.",
                region="gulf",
                tags=["shipping", "oil"],
                severity=0.3,
            )
        ],
    )

    assert "us" in next_observations
    assert "us" in rewards
    assert terminated is False
    assert truncated is False
    assert next_info["turn"] == 1
    assert next_info["world"]["turn"] == 1
    assert next_info["belief_state"]["us"]["beliefs"]


def test_trenches_openenv_environment_returns_scalar_reward_for_active_agent() -> None:
    runtime = TrenchesOpenEnvEnvironment()
    observation = runtime.reset(
        seed=11,
        training_agent="us",
        training_stage="stage_3_sparse",
        max_turns=12,
        include_joint_observations=True,
    )

    assert observation.training_agent == "us"
    assert observation.turn == 0
    assert observation.reward == 0.0
    assert observation.done is False
    assert "us" in observation.joint_observations

    next_observation = runtime.step(
        TrenchesOpenEnvAction(
            action=AgentAction(
                actor="us",
                type="negotiate",
                target="gulf",
                summary="Offer deconfliction and shipping guarantees.",
            ),
            external_signals=[
                ExternalSignal(
                    source="training-sim",
                    headline="Shipping risk rises near Hormuz.",
                    region="gulf",
                    tags=["shipping", "oil"],
                    severity=0.3,
                )
            ],
        )
    )

    assert next_observation.turn == 1
    assert next_observation.training_agent == "us"
    assert next_observation.reward == runtime.state.reward_breakdowns["us"].total
    assert next_observation.agent_observation.known_coalitions
    assert next_observation.agent_observation.belief_brief
    assert runtime.state.step_count == 1
    runtime.close()


def test_openenv_reset_accepts_named_scenario() -> None:
    runtime = TrenchesOpenEnvEnvironment()
    observation = runtime.reset(
        seed=17,
        training_agent="gulf",
        training_stage="stage_3_sparse",
        max_turns=12,
        scenario_id="shipping_crisis",
    )

    assert observation.agent_observation.strategic_state["shipping_continuity"] < 78.0
    assert runtime.state.session is not None
    assert runtime.state.session.episode.scenario_id == "shipping_crisis"
    assert runtime.state.session.world.latent_state["gulf"]["shipping_continuity"] < 78.0
    assert observation.agent_observation.projection.enabled
    runtime.close()


def test_openenv_autofills_missing_agents_with_shared_policy() -> None:
    runtime = TrenchesOpenEnvEnvironment()
    runtime.reset(
        seed=11,
        training_agent="us",
        training_stage="stage_3_sparse",
        max_turns=12,
        include_joint_observations=True,
    )

    runtime.step(
        TrenchesOpenEnvAction(
            action=AgentAction(
                actor="us",
                type="negotiate",
                target="gulf",
                summary="Offer deconfliction and shipping guarantees.",
            ),
            external_signals=[
                ExternalSignal(
                    source="training-sim",
                    headline="Shipping risk rises near Hormuz.",
                    region="gulf",
                    tags=["shipping", "oil"],
                    severity=0.3,
                )
            ],
        )
    )

    assert runtime.state.session is not None
    assert runtime.state.session.recent_traces[-1].actions["gulf"].type in {"defend", "negotiate", "intel_query"}
    runtime.close()


def test_historical_replay_step_records_prediction_and_scores_forecast() -> None:
    runtime = TrenchesOpenEnvEnvironment()
    observation = runtime.reset(
        seed=11,
        training_agent="us",
        training_stage="stage_1_dense",
        max_turns=4,
        replay_id="us_synthetic_seed_2025_2026",
        replay_start_index=0,
    )

    assert observation.historical_replay.enabled is True
    assert observation.historical_replay.current_event_index == 0
    assert observation.agent_observation.historical_brief

    next_observation = runtime.step(
        TrenchesOpenEnvAction(
            action=AgentAction(
                actor="us",
                type="negotiate",
                target="gulf",
                summary="Reassure Gulf partners and reinforce shipping protection.",
            ),
            prediction=Prediction(
                agent_id="us",
                topic="shipping",
                predicted_actor="us",
                predicted_target="shipping_lanes",
                time_horizon_turns=1,
                expected_severity="medium",
                confidence=0.74,
                summary="The next visible event is likely a US maritime reassurance move.",
                rationale="Washington is likely to answer shipping pressure with a visible assurance posture.",
            ),
        )
    )

    assert next_observation.revealed_event is not None
    assert next_observation.revealed_event.event_id == "evt-2025-02-us-maritime-posture"
    assert next_observation.reward_breakdown.forecast_total > 0.0
    assert next_observation.prediction_assessments["us"].evaluated_event_id == "evt-2025-02-us-maritime-posture"
    assert runtime.state.session is not None
    assert runtime.state.session.prediction_log[-1].agent_id == "us"
    assert runtime.state.session.prediction_assessments[-1].total > 0.0
    runtime.close()


def test_trenches_openenv_environment_rejects_unknown_training_agent() -> None:
    runtime = TrenchesOpenEnvEnvironment()

    with pytest.raises(ValueError):
        runtime.reset(training_agent="russia")


def test_native_openenv_fastapi_app_can_be_created() -> None:
    app = create_openenv_fastapi_app()

    if OPENENV_CORE_AVAILABLE:
        assert app is not None
    else:
        assert app is None


def test_typed_openenv_client_class_is_declared() -> None:
    assert TrenchesEnvClient is not None
