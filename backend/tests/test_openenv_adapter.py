from trenches_env.models import AgentAction, ExternalSignal
from trenches_env.openenv_adapter import OpenEnvAdapter


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
