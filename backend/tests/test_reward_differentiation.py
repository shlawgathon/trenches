from datetime import datetime, timezone

import pytest

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import (
    AgentAction,
    AgentBeliefEntry,
    AgentBeliefState,
    EpisodeMetadata,
    ExternalSignal,
    LatentEvent,
    SourcePacket,
    StepSessionRequest,
)
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

    assert "regional_access" in session.world.latent_state["us"]
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


def test_dense_stage_keeps_direct_observation_projection_disabled() -> None:
    env = FogOfWarDiplomacyEnv()
    session = env.create_session(seed=7, training_stage="stage_1_dense")
    observation = session.observations["israel"]

    assert not observation.projection.enabled
    assert observation.projection.mode == "direct"
    assert observation.strategic_state == session.world.latent_state["israel"]


def test_fog_of_war_projection_shapes_observation_view() -> None:
    env = FogOfWarDiplomacyEnv()
    session = env.create_session(seed=7, training_stage="stage_2_partial")
    observation = session.observations["israel"]

    assert observation.projection.enabled
    assert observation.projection.mode == "partial"
    assert observation.projection.notes
    assert observation.projection.obscured_metric_count > 0
    assert observation.projection.worldview_reliability < 1.0
    assert "Observation reliability is partial" in observation.decision_prompt
    assert observation.strategic_state["homeland_security"] != session.world.latent_state["israel"]["homeland_security"]


def test_world_tracks_latent_state_separately_from_public_state() -> None:
    env = FogOfWarDiplomacyEnv()
    session = env.create_session(seed=7, scenario_id="shipping_crisis")

    assert session.world.latent_state["gulf"]["shipping_continuity"] < session.world.actor_state["gulf"]["shipping_continuity"]
    assert session.world.latent_events
    assert any(event.topic == "shipping" for event in session.world.latent_events)
    assert session.belief_state["gulf"].beliefs
    assert "shipping" in session.belief_state["gulf"].dominant_topics


def test_scenario_latent_events_flow_into_private_observations() -> None:
    env = FogOfWarDiplomacyEnv()
    session = env.create_session(seed=7, scenario_id="shipping_crisis", training_stage="stage_3_sparse")

    assert any("coordinated" in brief.summary.lower() for brief in session.observations["gulf"].private_brief)
    assert session.observations["gulf"].belief_brief
    assert session.observations["gulf"].belief_topics
    assert "Belief memory:" in session.observations["gulf"].decision_prompt


def test_external_signals_create_linked_latent_events() -> None:
    env = FogOfWarDiplomacyEnv()
    session = env.create_session(seed=7)

    result = env.step_session(
        session,
        StepSessionRequest(
            actions={},
            external_signals=[
                ExternalSignal(
                    source="wire-service",
                    headline="Shipping risk rises in Hormuz after new tanker disruption reports.",
                    region="gulf",
                    tags=["shipping", "oil"],
                    severity=0.7,
                )
            ],
        ),
    )

    topics = {event.topic for event in result.session.world.latent_events if event.started_at_turn == result.session.world.turn}
    assert "shipping" in topics
    assert "market" in topics
    assert any("shipping" in belief.topic for belief in result.session.belief_state["gulf"].beliefs)


def test_source_projection_can_express_explicit_contradictions() -> None:
    env = FogOfWarDiplomacyEnv()
    session = env.create_session(seed=7, scenario_id="shipping_crisis", training_stage="stage_3_sparse")
    timestamp = datetime.now(timezone.utc)
    worsening_packet = SourcePacket(
        source_id="gulf-chokepoint-status",
        source_name="Maritime Chokepoint Disruption Panel",
        delivery="training_core",
        kind="api",
        endpoint_kind="worldmonitor",
        status="ok",
        fetched_at=timestamp,
        summary="Shipping disruption intensifies near Hormuz and tanker insurance costs rise.",
        sample_items=["Shipping risk rises"],
    )
    easing_packet = SourcePacket(
        source_id="gulf-shipping-rates",
        source_name="Shipping Rates Monitor",
        delivery="training_core",
        kind="api",
        endpoint_kind="worldmonitor",
        status="ok",
        fetched_at=timestamp,
        summary="Shipping disruption intensifies near Hormuz and tanker insurance costs rise.",
        sample_items=["Shipping risk rises"],
    )

    worsening_briefs, worsening_meta = env._source_packets_to_briefs(
        [worsening_packet],
        category="training_source",
        world=session.world,
        episode=session.episode,
        agent_id="gulf",
    )
    easing_briefs, easing_meta = env._source_packets_to_briefs(
        [easing_packet],
        category="training_source",
        world=session.world,
        episode=session.episode,
        agent_id="gulf",
    )

    assert worsening_meta["contradiction_packets"] == 1.0
    assert easing_meta["contradiction_packets"] == 1.0
    assert worsening_meta["contradiction_topics"] == ["shipping disruption"]
    assert any(event.topic == "shipping" for event in session.world.latent_events)
    assert "multiple commercial tankers report evasive maneuvers" in worsening_briefs[0].summary.lower()
    assert "renewed deterioration" in worsening_briefs[0].summary
    assert "partial stabilization" in easing_briefs[0].summary


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


def test_doctrine_specific_beliefs_weight_shared_events_differently() -> None:
    env = FogOfWarDiplomacyEnv()
    shipping_session = env.create_session(seed=7, scenario_id="shipping_crisis", training_stage="stage_3_sparse")
    border_session = env.create_session(seed=7, scenario_id="border_flareup", training_stage="stage_3_sparse")

    gulf_shipping = next(belief for belief in shipping_session.belief_state["gulf"].beliefs if belief.topic == "shipping")
    israel_shipping = next(belief for belief in shipping_session.belief_state["israel"].beliefs if belief.topic == "shipping")
    israel_border = next(belief for belief in border_session.belief_state["israel"].beliefs if belief.topic == "border")
    gulf_border = next(belief for belief in border_session.belief_state["gulf"].beliefs if belief.topic == "border")

    assert gulf_shipping.confidence > israel_shipping.confidence
    assert israel_border.confidence > gulf_border.confidence


def test_false_beliefs_persist_across_turns_without_reconfirmation() -> None:
    env = FogOfWarDiplomacyEnv()
    session = env.create_session(seed=7, training_stage="stage_3_sparse")
    session.world.turn = 1
    session.belief_state["israel"] = AgentBeliefState(
        agent_id="israel",
        dominant_topics=["border"],
        beliefs=[
            AgentBeliefEntry(
                belief_id="israel:false-border",
                topic="border",
                summary="Assessment: hidden launch-cell replenishment remains active north of the border.",
                confidence=0.74,
                status="active",
                source="analyst_note",
                suspected_agents=["hezbollah"],
                confirmation_count=2,
                contradiction_count=0,
                last_confirmed_turn=0,
                last_updated_turn=0,
            )
        ],
        last_revision_turn=0,
    )

    updated = env._update_belief_state(session)
    persisted = next(belief for belief in updated["israel"].beliefs if belief.belief_id == "israel:false-border")

    assert persisted.confidence < 0.74
    assert persisted.confidence > 0.6
    assert persisted.status == "active"


def test_contradictions_downgrade_beliefs_before_disconfirming_them() -> None:
    env = FogOfWarDiplomacyEnv()
    session = env.create_session(seed=7, training_stage="stage_3_sparse")
    shipping_event = LatentEvent(
        event_id="manual-shipping",
        topic="shipping",
        status="contained",
        severity=0.42,
        visibility="mixed",
        reliability=0.12,
        origin="test",
        affected_agents=["gulf"],
        started_at_turn=0,
        last_updated_turn=0,
    )
    session.world.latent_events = [shipping_event]
    session.belief_state["gulf"] = AgentBeliefState(
        agent_id="gulf",
        dominant_topics=["shipping"],
        beliefs=[
            AgentBeliefEntry(
                belief_id=f"gulf:{shipping_event.event_id}",
                topic="shipping",
                summary="Assessment: coordinated disruption remains active across the chokepoint.",
                confidence=0.78,
                status="confirmed",
                source="latent_event",
                suspected_agents=["iran"],
                related_event_ids=[shipping_event.event_id],
                confirmation_count=2,
                contradiction_count=0,
                last_confirmed_turn=0,
                last_updated_turn=0,
            )
        ],
        last_revision_turn=0,
    )

    session.world.turn = 1
    first_pass = env._update_belief_state(session)
    contested = next(belief for belief in first_pass["gulf"].beliefs if belief.belief_id == f"gulf:{shipping_event.event_id}")

    assert contested.status == "contested"
    assert contested.contradiction_count == 1
    assert contested.confidence > 0.42

    session.belief_state = first_pass
    session.world.turn = 2
    second_pass = env._update_belief_state(session)
    disconfirmed = next(belief for belief in second_pass["gulf"].beliefs if belief.belief_id == f"gulf:{shipping_event.event_id}")

    assert disconfirmed.contradiction_count == 2
    assert disconfirmed.status == "disconfirmed"
