import pytest

from trenches_env.models import AgentAction, ExternalSignal
from trenches_env.openenv_client import TrenchesEnvClient
from trenches_env.openenv_adapter import (
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
    assert runtime.state.step_count == 1
    runtime.close()


def test_trenches_openenv_environment_rejects_unknown_training_agent() -> None:
    runtime = TrenchesOpenEnvEnvironment()

    with pytest.raises(ValueError):
        runtime.reset(training_agent="russia")


def test_native_openenv_fastapi_app_can_be_created() -> None:
    app = create_openenv_fastapi_app()

    assert app is not None


def test_typed_openenv_client_class_is_declared() -> None:
    assert TrenchesEnvClient is not None
