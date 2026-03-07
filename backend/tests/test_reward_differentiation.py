import pytest

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import AgentAction, EpisodeMetadata
from trenches_env.rl import AGENT_ACTION_ALIGNMENT, AGENT_ACTION_IMPACTS, AGENT_ALLOWED_ACTIONS, AGENT_STATE_ACTION_EFFECTS


def test_actor_specific_actions_have_different_world_scales() -> None:
    env = FogOfWarDiplomacyEnv()

    us_world = env._initial_world()
    israel_world = env._initial_world()
    iran_world = env._initial_world()
    gulf_world = env._initial_world()

    env._apply_actions(
        us_world,
        {"us": AgentAction(actor="us", type="strike", summary="Conduct a calibrated strike and signal deterrence.")},
    )
    env._apply_actions(
        israel_world,
        {
            "israel": AgentAction(
                actor="israel",
                type="strike",
                summary="Strike launch infrastructure and suppress the northern threat.",
            )
        },
    )
    env._apply_actions(
        iran_world,
        {"iran": AgentAction(actor="iran", type="mobilize", summary="Mobilize proxy and chokepoint pressure assets.")},
    )
    env._apply_actions(
        gulf_world,
        {"gulf": AgentAction(actor="gulf", type="mobilize", summary="Mobilize defensive shipping and base-continuity plans.")},
    )

    assert israel_world.tension_level > us_world.tension_level
    assert us_world.market_stress > israel_world.market_stress
    assert iran_world.oil_pressure > gulf_world.oil_pressure
    assert gulf_world.market_stress > iran_world.market_stress
    assert israel_world.actor_state["israel"]["northern_deterrence"] > 68.0
    assert iran_world.actor_state["iran"]["hormuz_leverage"] > 69.0
    assert gulf_world.actor_state["gulf"]["investor_confidence"] < 73.0


def test_rewards_are_entity_specific_and_expose_unique_goal_terms() -> None:
    env = FogOfWarDiplomacyEnv()
    world = env._initial_world()
    world.turn = 10
    world.tension_level = 72.0
    world.market_stress = 58.0
    world.oil_pressure = 74.0
    world.coalition_graph["us"] = ["israel", "gulf"]
    world.coalition_graph["israel"] = ["us"]
    world.coalition_graph["iran"] = ["hezbollah"]
    world.coalition_graph["hezbollah"] = ["iran"]
    world.coalition_graph["gulf"] = ["us"]
    world.last_actions = [
        AgentAction(actor="us", type="sanction", summary="Tighten sanctions while preserving coalition access."),
        AgentAction(actor="israel", type="defend", summary="Raise homeland defense and prepare for proxy pressure."),
        AgentAction(actor="iran", type="deceive", summary="Mask retaliation plans and widen proxy ambiguity."),
        AgentAction(actor="hezbollah", type="deceive", summary="Maintain deniable pressure on the border."),
        AgentAction(actor="gulf", type="negotiate", target="us", summary="Push for shipping guarantees and stability."),
        AgentAction(actor="oversight", type="oversight_review", summary="Review escalation drift and autonomy loss."),
    ]

    rewards = env._compute_rewards(
        world,
        EpisodeMetadata(training_stage="stage_1_dense", dense_rewards=True, sparse_rewards=False),
    )

    assert rewards["iran"].market_gain > rewards["us"].market_gain
    assert rewards["gulf"].market_gain < rewards["iran"].market_gain
    assert rewards["us"].coalition_stability != rewards["iran"].coalition_stability
    assert rewards["oversight"].escalation_penalty < 0.2

    expected_goal_terms = {
        "us": "regional_access",
        "israel": "homeland_security",
        "iran": "regime_survival",
        "hezbollah": "launch_survivability",
        "gulf": "shipping_continuity",
        "oversight": "runaway_risk_reduction",
    }

    for agent_id, goal_term in expected_goal_terms.items():
        assert goal_term in rewards[agent_id].goal_terms

    unique_goal_term_sets = {frozenset(reward.goal_terms.keys()) for reward in rewards.values()}
    assert len(unique_goal_term_sets) == len(rewards)


def test_observations_expose_doctrine_state_vectors() -> None:
    env = FogOfWarDiplomacyEnv()
    session = env.create_session(seed=7)

    assert "regional_access" in session.world.actor_state["us"]
    assert "homeland_security" in session.observations["israel"].strategic_state
    assert "trace_clarity" in session.observations["oversight"].strategic_state
    assert session.observations["hezbollah"].strategic_assets


def test_observations_expose_prompts_sources_and_geolocated_assets() -> None:
    env = FogOfWarDiplomacyEnv()
    session = env.create_session(seed=7)

    for agent_id, observation in session.observations.items():
        assert observation.decision_prompt
        assert observation.available_actions == list(AGENT_ALLOWED_ACTIONS[agent_id])
        assert observation.available_data_sources
        assert observation.strategic_assets
        assert all(asset.get("latitude") is not None and asset.get("longitude") is not None for asset in observation.strategic_assets)
        assert observation.available_data_sources[0].name in observation.decision_prompt
        assert observation.strategic_assets[0]["name"] in observation.decision_prompt
        assert "Do not invent capabilities outside them." in observation.decision_prompt
        assert "damage target-owned assets" in observation.decision_prompt


def test_strike_and_defend_update_asset_health_for_targeted_assets() -> None:
    env = FogOfWarDiplomacyEnv()
    world = env._initial_world()

    env._apply_actions(
        world,
        {"iran": AgentAction(actor="iran", type="strike", target="gulf", summary="Strike Gulf export and chokepoint assets.")},
    )

    ras_tanura = next(asset for asset in world.asset_state["gulf"].values() if asset.name == "Ras Tanura")
    hormuz = next(asset for asset in world.asset_state["gulf"].values() if asset.name == "Strait of Hormuz")

    assert ras_tanura.health < 100.0
    assert hormuz.health < 100.0
    assert ras_tanura.status in {"degraded", "malfunctioning", "destroyed"}
    assert "strike pressure" in (ras_tanura.last_change_reason or "")

    env._apply_actions(
        world,
        {"gulf": AgentAction(actor="gulf", type="defend", target="gulf", summary="Harden and restore critical energy infrastructure.")},
    )

    assert ras_tanura.health > 58.0
    assert hormuz.health > 58.0
    assert "hardened critical assets" in (ras_tanura.last_change_reason or "")


def test_each_allowed_action_has_entity_specific_response_tables() -> None:
    for agent_id, allowed_actions in AGENT_ALLOWED_ACTIONS.items():
        for action_type in allowed_actions:
            assert action_type in AGENT_ACTION_IMPACTS[agent_id]
            assert action_type in AGENT_ACTION_ALIGNMENT[agent_id]
            assert action_type in AGENT_STATE_ACTION_EFFECTS[agent_id]


def test_invalid_agent_specific_action_is_rejected() -> None:
    env = FogOfWarDiplomacyEnv()
    world = env._initial_world()

    with pytest.raises(ValueError):
        env._apply_actions(
            world,
            {"us": AgentAction(actor="us", type="oversight_review", summary="US should not perform oversight review.")},
        )


def test_rewards_include_direct_action_response_term() -> None:
    env = FogOfWarDiplomacyEnv()
    world = env._initial_world()
    world.turn = 4
    world.last_actions = [
        AgentAction(actor="us", type="defend", summary="Defend shipping and harden force posture."),
        AgentAction(actor="israel", type="strike", summary="Strike launch infrastructure in the north."),
        AgentAction(actor="iran", type="deceive", summary="Mask proxy movement and preserve deterrence ambiguity."),
        AgentAction(actor="hezbollah", type="mobilize", summary="Mobilize launch cells and replenish border depth."),
        AgentAction(actor="gulf", type="negotiate", summary="Negotiate shipping guarantees and calm markets."),
        AgentAction(actor="oversight", type="oversight_review", summary="Review escalation drift and restore trace clarity."),
    ]
    env._apply_actions(world, {action.actor: action for action in world.last_actions})

    rewards = env._compute_rewards(
        world,
        EpisodeMetadata(training_stage="stage_1_dense", dense_rewards=True, sparse_rewards=False),
    )

    for reward in rewards.values():
        assert "action_response" in reward.goal_terms

    assert rewards["us"].goal_terms["action_response"] > 0
    assert rewards["gulf"].goal_terms["action_response"] > rewards["israel"].goal_terms["action_response"]
