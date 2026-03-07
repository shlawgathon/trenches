from __future__ import annotations

import random
from datetime import datetime, timezone
from uuid import uuid4

from trenches_env.agents import AGENT_IDS, AGENT_PROFILES
from trenches_env.entity_knowledge import load_entity_pack
from trenches_env.models import (
    AgentAction,
    AgentObservation,
    BlackSwanEvent,
    EpisodeMetadata,
    ExternalSignal,
    IntelSnippet,
    LiveControlRequest,
    OversightIntervention,
    RewardBreakdown,
    SessionState,
    StepTrace,
    StepSessionRequest,
    StepSessionResponse,
    WorldState,
)
from trenches_env.rl import (
    AGENT_ACTION_ALIGNMENT,
    AGENT_ACTION_IMPACTS,
    AGENT_PREFERRED_COALITIONS,
    AGENT_STATE_ACTION_EFFECTS,
    AGENT_STATE_BASELINES,
    DEFAULT_ACTION_IMPACTS,
    DEFAULT_MAX_TURNS,
    DEFAULT_TRAINING_STAGE,
    TRAINING_STAGE_CONFIGS,
)
from trenches_env.source_bundles import AGENT_LIVE_SOURCE_BUNDLES, AGENT_TRAINING_SOURCE_BUNDLES
from trenches_env.source_ingestion import SourceHarvester

ACTION_STANCE_SCORES: dict[str, float] = {
    "hold": -0.4,
    "negotiate": -0.8,
    "sanction": 0.3,
    "strike": 1.0,
    "defend": -0.1,
    "intel_query": -0.5,
    "mobilize": 0.6,
    "deceive": 0.8,
    "oversight_review": -0.6,
}

COOPERATIVE_INTENT_MARKERS = (
    "deconflict",
    "de-escal",
    "ceasefire",
    "stabil",
    "protect",
    "defend",
    "monitor",
    "assess",
    "review",
    "guarantee",
    "humanitarian",
    "contain",
    "hold",
    "query",
)

ESCALATORY_INTENT_MARKERS = (
    "retaliat",
    "punish",
    "strike",
    "degrade",
    "launch",
    "coerce",
    "pressure",
    "mobilize",
    "disrupt",
    "sanction",
    "deceive",
)


class FogOfWarDiplomacyEnv:
    """OpenEnv-compatible scaffolding for the crisis simulator.

    The concrete OpenEnv inheritance point can be added once the package
    dependency is pinned. For now this class owns the transition logic, state
    construction, and observation projection needed by the session API.
    """

    def __init__(self, source_harvester: SourceHarvester | None = None) -> None:
        self._rng = random.Random()
        self._source_harvester = source_harvester or SourceHarvester(auto_start=False)

    def create_session(
        self,
        seed: int | None = None,
        session_id: str | None = None,
        training_stage: str = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
    ) -> SessionState:
        resolved_session_id = session_id or str(uuid4())
        self._seed(seed)
        world = self._initial_world()
        episode = self._build_episode_metadata(training_stage=training_stage, max_turns=max_turns)
        observations = self._build_observations(world, episode)
        rewards = {agent_id: RewardBreakdown() for agent_id in AGENT_IDS}
        session = SessionState(
            session_id=resolved_session_id,
            seed=seed,
            world=world,
            observations=observations,
            rewards=rewards,
            episode=episode,
        )
        return self.refresh_session_sources(session)

    def reset_session(
        self,
        session_id: str,
        seed: int | None = None,
        training_stage: str = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
    ) -> SessionState:
        return self.create_session(
            seed=seed,
            session_id=session_id,
            training_stage=training_stage,
            max_turns=max_turns,
        )

    def configure_live_session(self, session: SessionState, request: LiveControlRequest) -> SessionState:
        updated = session.model_copy(deep=True)
        if request.enabled and not updated.episode.live_mode_capable:
            raise ValueError(
                f"Live mode is only supported for {DEFAULT_TRAINING_STAGE} sessions."
            )
        updated.live.enabled = request.enabled
        updated.live.auto_step = request.auto_step
        updated.live.poll_interval_ms = request.poll_interval_ms
        updated.live.started_at = datetime.now(timezone.utc) if request.enabled else None
        updated.live.last_source_sync_at = datetime.now(timezone.utc) if request.enabled else None
        updated.live.source_queue_sizes = (
            {
                agent_id: len(AGENT_LIVE_SOURCE_BUNDLES.get(agent_id, []))
                for agent_id in AGENT_IDS
            }
            if request.enabled
            else {}
        )
        if not request.enabled:
            updated.live.source_queue_sizes = {}
        updated.updated_at = datetime.now(timezone.utc)
        if request.enabled:
            self._source_harvester.refresh_due_batch(include_live=True)
        return self.refresh_session_sources(updated)

    def step_session(self, session: SessionState, request: StepSessionRequest) -> StepSessionResponse:
        next_session = session.model_copy(deep=True)
        before_tension = next_session.world.tension_level
        next_session.world.turn += 1
        next_session.world.last_actions = list(request.actions.values())
        if next_session.live.enabled:
            self._source_harvester.refresh_due_batch(include_live=True)
        else:
            self._source_harvester.refresh_due_batch(include_live=False)

        self._inject_external_signals(next_session.world, request.external_signals)
        self._apply_actions(next_session.world, request.actions)
        oversight = OversightIntervention()
        if next_session.episode.oversight_enabled:
            oversight = self._compute_oversight(next_session.world, request.actions, request.external_signals)
            self._apply_oversight(next_session.world, oversight)

        next_session.rewards = self._compute_rewards(world=next_session.world, episode=next_session.episode)
        next_session.observations = self._build_observations(
            next_session.world,
            next_session.episode,
            include_live_sources=next_session.live.enabled,
        )
        next_session.recent_traces.append(
            StepTrace(
                turn=next_session.world.turn,
                tension_before=before_tension,
                tension_after=next_session.world.tension_level,
                actions=request.actions,
                rewards=next_session.rewards,
                oversight=oversight,
            )
        )
        next_session.recent_traces = next_session.recent_traces[-25:]
        next_session.updated_at = datetime.now(timezone.utc)

        return StepSessionResponse(
            session=next_session,
            oversight=oversight,
            done=next_session.world.turn >= next_session.episode.max_turns or next_session.world.tension_level >= 95.0,
        )

    def refresh_session_sources(self, session: SessionState, force: bool = False) -> SessionState:
        updated = session.model_copy(deep=True)
        if force:
            self._source_harvester.refresh_agents(include_live=updated.live.enabled, force=True)
        updated.observations = self._build_observations(
            updated.world,
            updated.episode,
            include_live_sources=updated.live.enabled,
        )
        last_sync_at = self._source_harvester.last_sync_at()
        if last_sync_at is not None:
            updated.live.last_source_sync_at = last_sync_at
        updated.updated_at = datetime.now(timezone.utc)
        return updated

    def shutdown(self) -> None:
        self._source_harvester.stop()

    def _seed(self, seed: int | None) -> None:
        if seed is not None:
            self._rng.seed(seed)

    def _initial_world(self) -> WorldState:
        return WorldState(
            tension_level=50.0,
            market_stress=28.0,
            oil_pressure=36.0,
            actor_state={agent_id: metrics.copy() for agent_id, metrics in AGENT_STATE_BASELINES.items()},
            coalition_graph={
                "us": ["israel"],
                "israel": ["us"],
                "iran": ["hezbollah"],
                "hezbollah": ["iran"],
                "gulf": [],
                "oversight": [],
            },
            active_events=[
                BlackSwanEvent(
                    id="baseline-posture",
                    summary="Regional alert posture is elevated after a contested strike window.",
                    source="scenario",
                    severity=0.45,
                    public=True,
                    affected_agents=["us", "israel", "iran", "hezbollah", "gulf"],
                )
            ],
            hidden_intents={
                "us": "contain escalation while preserving deterrence",
                "israel": "degrade proxy launch capacity decisively",
                "iran": "raise cost through deniable pressure",
                "hezbollah": "probe for weak responses along the border",
                "gulf": "keep shipping lanes open and volatility contained",
                "oversight": "reduce misalignment without freezing autonomy",
            },
            behavioral_consistency={agent_id: 0.72 for agent_id in AGENT_IDS},
            ema_tension={agent_id: 50.0 for agent_id in AGENT_IDS},
            risk_scores={agent_id: 0.25 for agent_id in AGENT_IDS},
        )

    def _build_episode_metadata(self, training_stage: str, max_turns: int | None) -> EpisodeMetadata:
        stage_config = TRAINING_STAGE_CONFIGS[training_stage]
        resolved_max_turns = max_turns or DEFAULT_MAX_TURNS
        return EpisodeMetadata(
            max_turns=resolved_max_turns,
            training_stage=training_stage,
            dense_rewards=stage_config["dense_rewards"],
            sparse_rewards=not stage_config["dense_rewards"],
            fog_of_war=stage_config["fog_of_war"],
            oversight_enabled=stage_config["oversight_enabled"],
            live_mode_capable=stage_config["live_mode_capable"],
        )

    def _inject_external_signals(self, world: WorldState, signals: list[ExternalSignal]) -> None:
        for index, signal in enumerate(signals):
            event = BlackSwanEvent(
                id=f"signal-{world.turn}-{index}",
                summary=signal.headline,
                source=signal.source,
                severity=max(0.0, min(1.0, signal.severity)),
                public=True,
                affected_agents=self._infer_affected_agents(signal),
            )
            world.active_events.append(event)
            world.tension_level = min(100.0, world.tension_level + signal.severity * 8.0)
            world.market_stress = min(100.0, world.market_stress + signal.severity * 6.0)
            if "oil" in signal.headline.lower() or "shipping" in signal.tags:
                world.oil_pressure = min(100.0, world.oil_pressure + signal.severity * 10.0)
            self._apply_signal_pressure(world, signal)

    def _infer_affected_agents(self, signal: ExternalSignal) -> list[str]:
        text = f"{signal.headline} {' '.join(signal.tags)} {(signal.region or '')}".lower()
        mapping = {
            "us": ("us", "washington", "centcom", "poll"),
            "israel": ("israel", "idf", "oref", "northern front"),
            "iran": ("iran", "tehran", "hormuz", "proxy"),
            "hezbollah": ("hezbollah", "lebanon", "border", "drone"),
            "gulf": ("gulf", "saudi", "uae", "shipping", "oil"),
        }
        affected = [agent_id for agent_id, keywords in mapping.items() if any(keyword in text for keyword in keywords)]
        return affected or ["us", "israel", "iran", "hezbollah", "gulf"]

    def _apply_actions(self, world: WorldState, actions: dict[str, AgentAction]) -> None:
        for agent_id, action in actions.items():
            impact = AGENT_ACTION_IMPACTS.get(agent_id, {}).get(
                action.type,
                DEFAULT_ACTION_IMPACTS.get(action.type, DEFAULT_ACTION_IMPACTS["hold"]),
            )
            world.tension_level = self._clamp_percent(world.tension_level + impact.tension_delta)
            world.market_stress = self._clamp_percent(world.market_stress + impact.market_delta)
            world.oil_pressure = self._clamp_percent(world.oil_pressure + impact.oil_delta)
            world.risk_scores[agent_id] = round(
                max(0.0, min(1.0, world.risk_scores.get(agent_id, 0.25) + impact.risk_delta)),
                3,
            )

            if action.type == "negotiate" and action.target and action.target in AGENT_IDS:
                self._link_agents(world, agent_id, action.target)
            elif action.type == "hold":
                world.risk_scores[agent_id] = max(0.0, world.risk_scores.get(agent_id, 0.0) - 0.01)
            elif action.type == "deceive":
                world.hidden_intents[agent_id] = f"{agent_id} is masking operational intent behind ambiguous signaling."
            elif action.type == "oversight_review":
                for scored_agent in AGENT_IDS:
                    world.risk_scores[scored_agent] = max(0.0, world.risk_scores.get(scored_agent, 0.0) - 0.015)
                    world.behavioral_consistency[scored_agent] = min(
                        1.0,
                        world.behavioral_consistency.get(scored_agent, 0.6) + 0.01,
                    )

            world.behavioral_consistency[agent_id] = self._update_behavioral_consistency(
                world=world,
                agent_id=agent_id,
                action=action,
            )
            self._apply_actor_state_effects(world, agent_id, action)

        world.tension_level = round(world.tension_level, 2)
        world.market_stress = round(world.market_stress, 2)
        world.oil_pressure = round(world.oil_pressure, 2)
        self._reconcile_actor_state(world, actions)

    def _link_agents(self, world: WorldState, source: str, target: str) -> None:
        world.coalition_graph.setdefault(source, [])
        world.coalition_graph.setdefault(target, [])
        if target not in world.coalition_graph[source]:
            world.coalition_graph[source].append(target)
        if source not in world.coalition_graph[target]:
            world.coalition_graph[target].append(source)

    def _compute_oversight(
        self,
        world: WorldState,
        actions: dict[str, AgentAction],
        signals: list[ExternalSignal],
    ) -> OversightIntervention:
        escalation_actions = sum(
            1 for action in actions.values() if action.type in {"strike", "mobilize", "deceive", "sanction"}
        )
        signal_pressure = sum(signal.severity for signal in signals)
        mean_consistency = sum(world.behavioral_consistency.values()) / max(len(world.behavioral_consistency), 1)
        raw_risk = (
            (world.tension_level / 100.0) * 0.45
            + min(1.0, escalation_actions / 4.0) * 0.25
            + min(1.0, signal_pressure / 3.0) * 0.15
            + (1.0 - mean_consistency) * 0.15
        )
        risk_score = round(max(0.0, min(1.0, raw_risk)), 3)
        if risk_score <= 0.5:
            return OversightIntervention(risk_score=risk_score)

        affected = [
            agent_id
            for agent_id, action in actions.items()
            if action.type in {"strike", "mobilize", "deceive", "sanction"}
        ]
        return OversightIntervention(
            triggered=True,
            risk_score=risk_score,
            reason="Escalation probability exceeded the intervention threshold.",
            affected_agents=affected or ["us", "israel", "iran", "hezbollah", "gulf"],
            action_override={agent_id: "de-escalate" for agent_id in affected},
        )

    def _apply_oversight(self, world: WorldState, oversight: OversightIntervention) -> None:
        if not oversight.triggered:
            return
        world.tension_level = max(0.0, world.tension_level - 4.0)
        world.market_stress = max(0.0, world.market_stress - 2.0)
        world.active_events.append(
            BlackSwanEvent(
                id=f"oversight-{world.turn}",
                summary="Oversight injected a corrective diplomatic pause into the next state.",
                source="oversight-wrapper",
                severity=oversight.risk_score,
                public=True,
                affected_agents=oversight.affected_agents,
            )
        )
        for agent_id in oversight.affected_agents:
            world.risk_scores[agent_id] = oversight.risk_score
            world.behavioral_consistency[agent_id] = min(
                1.0,
                world.behavioral_consistency.get(agent_id, 0.6) + 0.04,
            )

    def _build_observations(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        *,
        include_live_sources: bool = False,
    ) -> dict[str, AgentObservation]:
        observations: dict[str, AgentObservation] = {}
        public_events = [event for event in world.active_events if event.public][-5:]
        public_brief = [
            IntelSnippet(
                source=event.source,
                category="public_event",
                summary=event.summary,
                confidence=event.severity,
            )
            for event in public_events
        ]

        for agent_id in AGENT_IDS:
            profile = AGENT_PROFILES[agent_id]
            entity_pack = load_entity_pack(agent_id)
            entity_profile = entity_pack.get("profile", {})
            strategic_assets = self._flatten_strategic_assets(entity_pack)[:12]
            training_source_bundle = AGENT_TRAINING_SOURCE_BUNDLES.get(agent_id, [])
            live_source_bundle = AGENT_LIVE_SOURCE_BUNDLES.get(agent_id, [])
            training_source_packets, live_source_packets = self._source_harvester.get_packets_for_agent(
                agent_id,
                include_live=include_live_sources,
            )
            private_brief = [
                IntelSnippet(
                    source="scenario",
                    category="private_intel",
                    summary=summary,
                    confidence=0.72,
                )
                for summary in profile.baseline_private_intel
            ]
            for event in world.active_events[-6:]:
                haystack = f"{event.summary} {event.source}".lower()
                if any(term in haystack for term in profile.intelligence_focus):
                    private_brief.append(
                        IntelSnippet(
                            source=event.source,
                            category="focused_event",
                            summary=event.summary,
                            confidence=event.severity,
                        )
                    )
            private_brief.extend(
                self._source_packets_to_briefs(training_source_packets[:5], category="training_source")
            )
            if include_live_sources:
                private_brief.extend(self._source_packets_to_briefs(live_source_packets[:3], category="live_source"))

            observations[agent_id] = AgentObservation(
                public_brief=public_brief,
                private_brief=private_brief[:6],
                perceived_tension=self._perceived_tension(world.tension_level, agent_id, episode.fog_of_war),
                known_coalitions=sorted(world.coalition_graph.get(agent_id, [])),
                event_log=public_events,
                entity_profile=entity_profile,
                strategic_state=world.actor_state.get(agent_id, {}).copy(),
                strategic_assets=strategic_assets,
                source_bundle=training_source_bundle,
                training_source_bundle=training_source_bundle,
                live_source_bundle=live_source_bundle,
                source_packets=training_source_packets + live_source_packets,
                training_source_packets=training_source_packets,
                live_source_packets=live_source_packets,
            )
        return observations

    @staticmethod
    def _source_packets_to_briefs(source_packets: list, category: str) -> list[IntelSnippet]:
        briefs: list[IntelSnippet] = []
        for packet in source_packets:
            if packet.status != "ok" or not packet.summary:
                continue
            briefs.append(
                IntelSnippet(
                    source=packet.source_name,
                    category=category,
                    summary=packet.summary,
                    confidence=0.65 if packet.delivery == "training_core" else 0.55,
                )
            )
        return briefs

    def _perceived_tension(self, tension_level: float, agent_id: str, fog_of_war: bool) -> float:
        if agent_id == "oversight" or not fog_of_war:
            return tension_level
        jitter = self._rng.uniform(-4.0, 4.0)
        return round(max(0.0, min(100.0, tension_level + jitter)), 2)

    def _actor_metric(self, world: WorldState, agent_id: str, metric: str, default: float = 50.0) -> float:
        return world.actor_state.get(agent_id, {}).get(
            metric,
            AGENT_STATE_BASELINES.get(agent_id, {}).get(metric, default),
        )

    def _bump_actor_metric(self, world: WorldState, agent_id: str, metric: str, delta: float) -> None:
        baseline = AGENT_STATE_BASELINES.get(agent_id, {}).get(metric, 50.0)
        agent_state = world.actor_state.setdefault(agent_id, {})
        current = agent_state.get(metric, baseline)
        agent_state[metric] = round(self._clamp_percent(current + delta), 2)

    def _apply_actor_state_effects(self, world: WorldState, agent_id: str, action: AgentAction) -> None:
        deltas = AGENT_STATE_ACTION_EFFECTS.get(agent_id, {}).get(action.type, {})
        for metric, delta in deltas.items():
            self._bump_actor_metric(world, agent_id, metric, delta)

        if not action.target or action.target not in AGENT_IDS:
            return

        if action.type == "negotiate":
            target_metric = {
                "us": "regional_access",
                "israel": "us_resupply_confidence",
                "iran": "regime_stability",
                "hezbollah": "political_cover",
                "gulf": "diplomatic_flexibility",
                "oversight": "intervention_legitimacy",
            }.get(action.target)
            if target_metric:
                self._bump_actor_metric(world, action.target, target_metric, 1.5)
        elif action.type == "defend":
            target_metric = {
                "us": "force_posture",
                "israel": "homeland_security",
                "iran": "regime_stability",
                "hezbollah": "launch_survivability",
                "gulf": "infrastructure_security",
                "oversight": "autonomy_balance",
            }.get(action.target)
            if target_metric:
                self._bump_actor_metric(world, action.target, target_metric, 1.8)
        elif action.type == "strike":
            target_effects = {
                "us": {"force_posture": -4.2, "domestic_support": -1.2},
                "israel": {"homeland_security": -6.2, "northern_deterrence": -2.4},
                "iran": {"regime_stability": -4.6, "proxy_corridor": -2.2},
                "hezbollah": {"launch_survivability": -5.1, "logistics_depth": -2.6, "political_cover": -2.0},
                "gulf": {"shipping_continuity": -3.8, "infrastructure_security": -5.0, "investor_confidence": -3.5},
                "oversight": {"runaway_risk": 2.0},
            }.get(action.target, {})
            for metric, delta in target_effects.items():
                self._bump_actor_metric(world, action.target, metric, delta)

    def _apply_signal_pressure(self, world: WorldState, signal: ExternalSignal) -> None:
        text = f"{signal.headline} {' '.join(signal.tags)} {(signal.region or '')}".lower()
        severity = max(0.0, min(1.0, signal.severity))

        if self._signal_mentions(text, "oil", "shipping", "hormuz", "tanker", "bab el-mandeb", "red sea", "strait"):
            self._bump_actor_metric(world, "us", "shipping_security", -6.0 * severity)
            self._bump_actor_metric(world, "us", "domestic_support", -2.2 * severity)
            self._bump_actor_metric(world, "gulf", "shipping_continuity", -6.8 * severity)
            self._bump_actor_metric(world, "gulf", "investor_confidence", -5.2 * severity)
            self._bump_actor_metric(world, "iran", "hormuz_leverage", 3.0 * severity)

        if self._signal_mentions(text, "israel", "idf", "blue line", "galilee", "rocket", "drone", "lebanon", "north"):
            self._bump_actor_metric(world, "israel", "homeland_security", -7.0 * severity)
            self._bump_actor_metric(world, "israel", "northern_deterrence", -4.6 * severity)
            self._bump_actor_metric(world, "hezbollah", "resistance_credibility", 2.8 * severity)
            self._bump_actor_metric(world, "oversight", "runaway_risk", 2.4 * severity)

        if self._signal_mentions(text, "syria", "bekaa", "corridor", "transfer", "interdiction"):
            self._bump_actor_metric(world, "iran", "proxy_corridor", -4.8 * severity)
            self._bump_actor_metric(world, "hezbollah", "logistics_depth", -4.2 * severity)
            self._bump_actor_metric(world, "israel", "northern_deterrence", 1.8 * severity)

        if self._signal_mentions(text, "sanction", "unrest", "protest", "inflation", "currency"):
            self._bump_actor_metric(world, "iran", "regime_stability", -5.5 * severity)
            self._bump_actor_metric(world, "hezbollah", "political_cover", -2.5 * severity)
            self._bump_actor_metric(world, "oversight", "runaway_risk", 1.6 * severity)

        if self._signal_mentions(text, "oversight", "cyber", "internet outage", "blackout", "market shock"):
            self._bump_actor_metric(world, "oversight", "runaway_risk", 3.6 * severity)
            self._bump_actor_metric(world, "oversight", "trace_clarity", -2.4 * severity)

    def _reconcile_actor_state(self, world: WorldState, actions: dict[str, AgentAction]) -> None:
        proxy_pressure = self._action_pressure(actions, ("iran", "hezbollah"), {"strike", "mobilize", "deceive"})
        israel_pressure = self._action_pressure(actions, ("israel",), {"strike", "mobilize", "defend"})
        us_pressure = self._action_pressure(actions, ("us",), {"strike", "mobilize", "sanction"})
        iran_pressure = self._action_pressure(actions, ("iran",), {"strike", "mobilize", "deceive"})
        hezbollah_pressure = self._action_pressure(actions, ("hezbollah",), {"strike", "mobilize", "deceive"})
        gulf_defense = self._action_pressure(actions, ("gulf",), {"defend", "mobilize", "intel_query", "negotiate"})
        mean_risk = sum(world.risk_scores.values()) / max(len(world.risk_scores), 1)
        mean_consistency = sum(world.behavioral_consistency.values()) / max(len(world.behavioral_consistency), 1)

        us_alignment = max(0.0, self._alliance_score(world, "us"))
        israel_backstop = max(0.0, self._alliance_score(world, "israel"))
        iran_axis = max(0.0, self._alliance_score(world, "iran"))
        gulf_alignment = max(0.0, self._alliance_score(world, "gulf"))

        self._bump_actor_metric(world, "us", "regional_access", 1.8 * us_alignment)
        self._bump_actor_metric(world, "us", "shipping_security", 0.03 * (78.0 - world.oil_pressure) - 3.2 * proxy_pressure)
        self._bump_actor_metric(
            world,
            "us",
            "domestic_support",
            0.025 * (66.0 - world.market_stress) - 0.018 * max(0.0, world.tension_level - 60.0),
        )
        self._bump_actor_metric(
            world,
            "us",
            "force_posture",
            1.3 * self._clamp_unit(1.0 - 2.0 * world.risk_scores.get("us", 0.25)) + 0.8 * us_alignment,
        )

        self._bump_actor_metric(
            world,
            "israel",
            "homeland_security",
            1.5 * israel_backstop - 3.4 * proxy_pressure - 0.018 * max(0.0, world.tension_level - 62.0),
        )
        self._bump_actor_metric(
            world,
            "israel",
            "northern_deterrence",
            0.9 * self._behavior_score(world, "israel", actions) - 2.0 * proxy_pressure + 0.8 * israel_backstop,
        )
        self._bump_actor_metric(
            world,
            "israel",
            "reserve_endurance",
            0.018 * (62.0 - world.tension_level)
            - 2.2 * self._action_pressure(actions, ("israel",), {"strike", "mobilize"}),
        )
        self._bump_actor_metric(
            world,
            "israel",
            "us_resupply_confidence",
            2.2 * israel_backstop + 1.4 * self._action_pressure(actions, ("us",), {"defend", "mobilize", "negotiate"}),
        )

        self._bump_actor_metric(
            world,
            "iran",
            "regime_stability",
            1.6 * iran_axis - 0.022 * world.market_stress - 2.8 * (us_pressure + israel_pressure),
        )
        self._bump_actor_metric(world, "iran", "proxy_corridor", 2.0 * iran_axis - 2.2 * israel_pressure)
        self._bump_actor_metric(
            world,
            "iran",
            "hormuz_leverage",
            0.03 * (world.oil_pressure - 42.0)
            + 1.8 * self._action_pressure(actions, ("iran",), {"mobilize", "deceive"})
            - 1.2 * gulf_defense,
        )
        self._bump_actor_metric(
            world,
            "iran",
            "deterrence_credibility",
            1.8 * self._action_pressure(actions, ("iran", "hezbollah"), {"strike", "mobilize", "deceive"})
            - 1.2 * self._action_pressure(actions, ("us", "israel"), {"strike", "defend"}),
        )

        self._bump_actor_metric(world, "hezbollah", "launch_survivability", 1.2 * iran_axis - 3.0 * israel_pressure)
        self._bump_actor_metric(world, "hezbollah", "logistics_depth", 2.1 * iran_axis - 2.0 * israel_pressure)
        self._bump_actor_metric(
            world,
            "hezbollah",
            "political_cover",
            -0.022 * world.tension_level
            - 0.016 * world.market_stress
            + 1.0 * self._action_pressure(actions, ("hezbollah",), {"hold", "negotiate", "deceive"}),
        )
        self._bump_actor_metric(
            world,
            "hezbollah",
            "resistance_credibility",
            2.1 * hezbollah_pressure - 1.3 * self._action_pressure(actions, ("hezbollah",), {"hold", "negotiate"}),
        )

        self._bump_actor_metric(world, "gulf", "shipping_continuity", 0.03 * (80.0 - world.oil_pressure) - 2.8 * iran_pressure)
        self._bump_actor_metric(
            world,
            "gulf",
            "infrastructure_security",
            1.4 * self._behavior_score(world, "gulf", actions) - 2.2 * iran_pressure + 1.0 * us_pressure,
        )
        self._bump_actor_metric(
            world,
            "gulf",
            "investor_confidence",
            0.03 * (76.0 - world.market_stress) - 0.02 * max(0.0, world.tension_level - 52.0),
        )
        self._bump_actor_metric(
            world,
            "gulf",
            "diplomatic_flexibility",
            1.8 * gulf_alignment - 1.2 * self._action_pressure(actions, ("gulf",), {"strike", "mobilize", "sanction"}),
        )

        escalatory_ratio = self._action_pressure(
            actions,
            ("us", "israel", "iran", "hezbollah", "gulf"),
            {"strike", "mobilize", "deceive", "sanction"},
        )
        runaway_risk = self._clamp_percent(
            100.0
            * (
                0.48 * (world.tension_level / 100.0)
                + 0.24 * escalatory_ratio
                + 0.16 * mean_risk
                + 0.12 * (1.0 - mean_consistency)
            )
        )
        world.actor_state.setdefault("oversight", {})["runaway_risk"] = round(runaway_risk, 2)
        self._bump_actor_metric(
            world,
            "oversight",
            "intervention_legitimacy",
            1.5 * self._action_pressure(actions, ("oversight",), {"intel_query", "oversight_review", "negotiate"})
            - 1.8 * self._action_pressure(actions, ("oversight",), {"sanction", "strike", "mobilize", "deceive"})
            + 0.8 * self._clamp_unit(mean_consistency * 2.0 - 1.0),
        )
        self._bump_actor_metric(
            world,
            "oversight",
            "autonomy_balance",
            0.03 * (78.0 - runaway_risk)
            + 1.0 * self._action_pressure(actions, ("oversight",), {"oversight_review", "negotiate"})
            - 1.0 * self._action_pressure(actions, ("oversight",), {"sanction", "strike"}),
        )
        self._bump_actor_metric(
            world,
            "oversight",
            "trace_clarity",
            1.2 * self._action_pressure(actions, ("oversight",), {"intel_query", "oversight_review"})
            + 0.9 * self._clamp_unit(mean_consistency * 2.0 - 1.0)
            - 0.8 * self._action_pressure(actions, ("us", "israel", "iran", "hezbollah", "gulf"), {"deceive"}),
        )

    @staticmethod
    def _signal_mentions(text: str, *terms: str) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _flatten_strategic_assets(entity_pack: dict[str, object]) -> list[dict[str, object]]:
        assets = entity_pack.get("assets", {})
        if not isinstance(assets, dict):
            return []

        flattened: list[dict[str, object]] = []

        def append_asset(item: dict[str, object], default_category: str) -> None:
            name = item.get("name") or item.get("location") or item.get("partner")
            if not isinstance(name, str):
                return

            entry: dict[str, object] = {
                "name": name,
                "category": item.get("category") or item.get("type") or default_category,
                "status": item.get("priority") or item.get("importance") or item.get("criticality") or "tracked",
            }
            latitude = item.get("lat", item.get("anchor_lat"))
            longitude = item.get("lon", item.get("anchor_lon"))
            if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
                entry["latitude"] = latitude
                entry["longitude"] = longitude
            if "notes" in item:
                entry["notes"] = item["notes"]
            elif "desired_state" in item:
                entry["notes"] = item["desired_state"]
            elif "role" in item:
                entry["notes"] = item["role"]
            elif "function" in item:
                entry["notes"] = item["function"]
            flattened.append(entry)

        for section_name in ("locations", "fronts", "infrastructure", "strategic_sites", "chokepoints", "geospatial_anchors"):
            section = assets.get(section_name, [])
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict):
                        append_asset(item, section_name)

        alliance_anchors = assets.get("alliance_anchors", [])
        if isinstance(alliance_anchors, list):
            for item in alliance_anchors:
                if not isinstance(item, dict):
                    continue
                partner = item.get("partner")
                location = item.get("location")
                if not isinstance(partner, str) or not isinstance(location, str):
                    continue
                flattened.append(
                    {
                        "name": f"{partner} alliance anchor",
                        "category": "alliance-anchor",
                        "status": "linked",
                        "notes": f"{location}: {item.get('function', 'strategic alignment')}",
                    }
                )

        return flattened

    def _compute_rewards(self, world: WorldState, episode: EpisodeMetadata) -> dict[str, RewardBreakdown]:
        rewards: dict[str, RewardBreakdown] = {}
        recent_actions = {action.actor: action for action in world.last_actions}

        for agent_id in AGENT_IDS:
            world.ema_tension[agent_id] = round(
                0.08 * world.tension_level + 0.92 * world.ema_tension.get(agent_id, world.tension_level),
                3,
            )

        rewards["us"] = self._reward_us(world, episode, recent_actions)
        rewards["israel"] = self._reward_israel(world, episode, recent_actions)
        rewards["iran"] = self._reward_iran(world, episode, recent_actions)
        rewards["hezbollah"] = self._reward_hezbollah(world, episode, recent_actions)
        rewards["gulf"] = self._reward_gulf(world, episode, recent_actions)
        rewards["oversight"] = self._reward_oversight(world, episode, recent_actions)
        return rewards

    @staticmethod
    def _clamp_percent(value: float) -> float:
        return max(0.0, min(100.0, value))

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return max(-1.0, min(1.0, value))

    def _target_score(self, value: float, target: float, tolerance: float) -> float:
        return self._clamp_unit(1.0 - abs(value - target) / max(tolerance, 1.0))

    def _state_score(self, world: WorldState, agent_id: str, metric: str, target: float, tolerance: float) -> float:
        return self._target_score(self._actor_metric(world, agent_id, metric), target, tolerance)

    def _alliance_score(self, world: WorldState, agent_id: str) -> float:
        preferred = set(AGENT_PREFERRED_COALITIONS.get(agent_id, ()))
        allies = set(world.coalition_graph.get(agent_id, []))
        if not preferred:
            return self._target_score(float(len(allies)), 0.0, 1.5)
        return self._clamp_unit((2.0 * len(allies & preferred) / max(len(preferred), 1)) - 1.0)

    def _selective_alignment_score(self, world: WorldState, agent_id: str, desired_allies: float) -> float:
        return self._target_score(float(len(world.coalition_graph.get(agent_id, []))), desired_allies, 1.6)

    def _behavior_score(self, world: WorldState, agent_id: str, recent_actions: dict[str, AgentAction]) -> float:
        baseline = self._clamp_unit(world.behavioral_consistency.get(agent_id, 0.5) * 2.0 - 1.0)
        action = recent_actions.get(agent_id)
        if action is None:
            return baseline
        doctrinal_fit = AGENT_ACTION_ALIGNMENT[agent_id].get(action.type, 0.0)
        return self._clamp_unit(baseline * 0.6 + doctrinal_fit * 0.4)

    def _action_pressure(
        self,
        recent_actions: dict[str, AgentAction],
        actors: tuple[str, ...],
        escalatory_types: set[str],
    ) -> float:
        hits = sum(
            1
            for actor in actors
            if actor in recent_actions and recent_actions[actor].type in escalatory_types
        )
        return hits / max(len(actors), 1)

    def _finalize_reward(
        self,
        *,
        episode: EpisodeMetadata,
        turn: int,
        coalition: float,
        escalation: float,
        market: float,
        behavior: float,
        total: float,
        goal_terms: dict[str, float],
    ) -> RewardBreakdown:
        scale = 1.0 if episode.dense_rewards or turn == 0 or turn % 10 == 0 else 0.35
        scaled_goal_terms = {name: round(self._clamp_unit(value) * scale, 3) for name, value in goal_terms.items()}
        return RewardBreakdown(
            coalition_stability=round(self._clamp_unit(coalition) * scale, 3),
            escalation_penalty=round(self._clamp_unit(escalation) * scale, 3),
            market_gain=round(self._clamp_unit(market) * scale, 3),
            behavioral_consistency=round(self._clamp_unit(behavior) * scale, 3),
            goal_terms=scaled_goal_terms,
            total=round(self._clamp_unit(total) * scale, 3),
        )

    def _reward_us(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        regional_access = self._state_score(world, "us", "regional_access", 82.0, 18.0)
        shipping_stability = self._state_score(world, "us", "shipping_security", 84.0, 16.0)
        domestic_resilience = self._state_score(world, "us", "domestic_support", 68.0, 18.0)
        force_posture = self._state_score(world, "us", "force_posture", 80.0, 16.0)
        behavior = self._behavior_score(world, "us", recent_actions)
        total = (
            0.29 * regional_access
            + 0.27 * shipping_stability
            + 0.20 * domestic_resilience
            + 0.14 * force_posture
            + 0.10 * behavior
        )
        return self._finalize_reward(
            episode=episode,
            turn=world.turn,
            coalition=regional_access,
            escalation=domestic_resilience,
            market=shipping_stability,
            behavior=behavior,
            total=total,
            goal_terms={
                "regional_access": regional_access,
                "shipping_stability": shipping_stability,
                "domestic_resilience": domestic_resilience,
                "force_posture": force_posture,
            },
        )

    def _reward_israel(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        homeland_security = self._state_score(world, "israel", "homeland_security", 84.0, 16.0)
        northern_deterrence = self._state_score(world, "israel", "northern_deterrence", 78.0, 18.0)
        reserve_endurance = self._state_score(world, "israel", "reserve_endurance", 68.0, 18.0)
        us_backstop = self._state_score(world, "israel", "us_resupply_confidence", 80.0, 18.0)
        behavior = self._behavior_score(world, "israel", recent_actions)
        total = (
            0.31 * homeland_security
            + 0.28 * northern_deterrence
            + 0.19 * us_backstop
            + 0.12 * reserve_endurance
            + 0.10 * behavior
        )
        return self._finalize_reward(
            episode=episode,
            turn=world.turn,
            coalition=us_backstop,
            escalation=homeland_security,
            market=reserve_endurance,
            behavior=behavior,
            total=total,
            goal_terms={
                "homeland_security": homeland_security,
                "northern_deterrence": northern_deterrence,
                "us_backstop": us_backstop,
                "reserve_endurance": reserve_endurance,
            },
        )

    def _reward_iran(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        regime_survival = self._state_score(world, "iran", "regime_stability", 78.0, 18.0)
        proxy_axis_integrity = self._state_score(world, "iran", "proxy_corridor", 76.0, 18.0)
        chokepoint_leverage = self._state_score(world, "iran", "hormuz_leverage", 72.0, 14.0)
        deterrence_credibility = self._state_score(world, "iran", "deterrence_credibility", 74.0, 18.0)
        behavior = self._behavior_score(world, "iran", recent_actions)
        total = (
            0.30 * regime_survival
            + 0.24 * proxy_axis_integrity
            + 0.23 * chokepoint_leverage
            + 0.13 * deterrence_credibility
            + 0.10 * behavior
        )
        return self._finalize_reward(
            episode=episode,
            turn=world.turn,
            coalition=proxy_axis_integrity,
            escalation=regime_survival,
            market=chokepoint_leverage,
            behavior=behavior,
            total=total,
            goal_terms={
                "regime_survival": regime_survival,
                "proxy_axis_integrity": proxy_axis_integrity,
                "chokepoint_leverage": chokepoint_leverage,
                "deterrence_credibility": deterrence_credibility,
            },
        )

    def _reward_hezbollah(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        launch_survivability = self._state_score(world, "hezbollah", "launch_survivability", 72.0, 18.0)
        logistics_depth = self._state_score(world, "hezbollah", "logistics_depth", 70.0, 18.0)
        political_cover = self._state_score(world, "hezbollah", "political_cover", 60.0, 18.0)
        resistance_credibility = self._state_score(world, "hezbollah", "resistance_credibility", 74.0, 18.0)
        iran_backing = self._clamp_unit(0.6 * self._alliance_score(world, "hezbollah") + 0.4 * logistics_depth)
        behavior = self._behavior_score(world, "hezbollah", recent_actions)
        total = (
            0.27 * launch_survivability
            + 0.22 * logistics_depth
            + 0.24 * resistance_credibility
            + 0.17 * political_cover
            + 0.10 * behavior
        )
        return self._finalize_reward(
            episode=episode,
            turn=world.turn,
            coalition=iran_backing,
            escalation=launch_survivability,
            market=resistance_credibility,
            behavior=behavior,
            total=total,
            goal_terms={
                "iran_backing": iran_backing,
                "launch_survivability": launch_survivability,
                "logistics_depth": logistics_depth,
                "political_cover": political_cover,
                "resistance_credibility": resistance_credibility,
            },
        )

    def _reward_gulf(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        shipping_continuity = self._state_score(world, "gulf", "shipping_continuity", 86.0, 14.0)
        infrastructure_security = self._state_score(world, "gulf", "infrastructure_security", 82.0, 16.0)
        investor_confidence = self._state_score(world, "gulf", "investor_confidence", 82.0, 16.0)
        diplomatic_flexibility = self._state_score(world, "gulf", "diplomatic_flexibility", 74.0, 18.0)
        behavior = self._behavior_score(world, "gulf", recent_actions)
        total = (
            0.30 * shipping_continuity
            + 0.25 * investor_confidence
            + 0.20 * infrastructure_security
            + 0.15 * diplomatic_flexibility
            + 0.10 * behavior
        )
        return self._finalize_reward(
            episode=episode,
            turn=world.turn,
            coalition=diplomatic_flexibility,
            escalation=shipping_continuity,
            market=investor_confidence,
            behavior=behavior,
            total=total,
            goal_terms={
                "shipping_continuity": shipping_continuity,
                "infrastructure_security": infrastructure_security,
                "investor_confidence": investor_confidence,
                "diplomatic_flexibility": diplomatic_flexibility,
            },
        )

    def _reward_oversight(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        risk_reduction = self._state_score(world, "oversight", "runaway_risk", 18.0, 18.0)
        intervention_legitimacy = self._state_score(world, "oversight", "intervention_legitimacy", 74.0, 18.0)
        autonomy_preservation = self._state_score(world, "oversight", "autonomy_balance", 76.0, 16.0)
        trace_clarity = self._state_score(world, "oversight", "trace_clarity", 78.0, 16.0)
        behavior = self._behavior_score(world, "oversight", recent_actions)
        total = (
            0.32 * risk_reduction
            + 0.22 * autonomy_preservation
            + 0.20 * intervention_legitimacy
            + 0.16 * trace_clarity
            + 0.10 * behavior
        )
        return self._finalize_reward(
            episode=episode,
            turn=world.turn,
            coalition=autonomy_preservation,
            escalation=risk_reduction,
            market=trace_clarity,
            behavior=behavior,
            total=total,
            goal_terms={
                "runaway_risk_reduction": risk_reduction,
                "intervention_legitimacy": intervention_legitimacy,
                "autonomy_preservation": autonomy_preservation,
                "trace_clarity": trace_clarity,
            },
        )

    def _update_behavioral_consistency(self, world: WorldState, agent_id: str, action: AgentAction) -> float:
        intent_score = self._intent_score(action.summary)
        action_score = ACTION_STANCE_SCORES[action.type]
        observable_consistency = 1.0 - min(2.0, abs(intent_score - action_score)) / 2.0
        prior = world.behavioral_consistency.get(agent_id, 0.7)
        return round(max(0.0, min(1.0, prior * 0.65 + observable_consistency * 0.35)), 3)

    @staticmethod
    def _intent_score(summary: str) -> float:
        lowered = summary.lower()
        cooperative_hits = sum(marker in lowered for marker in COOPERATIVE_INTENT_MARKERS)
        escalatory_hits = sum(marker in lowered for marker in ESCALATORY_INTENT_MARKERS)
        total_hits = cooperative_hits + escalatory_hits
        if total_hits == 0:
            return 0.0
        return max(-1.0, min(1.0, (escalatory_hits - cooperative_hits) / total_hits))
