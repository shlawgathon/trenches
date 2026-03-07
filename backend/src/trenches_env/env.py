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
from trenches_env.source_ingestion import SourceHarvester
from trenches_env.rl import AGENT_REWARD_WEIGHTS, DEFAULT_MAX_TURNS, DEFAULT_TRAINING_STAGE, TRAINING_STAGE_CONFIGS
from trenches_env.source_bundles import AGENT_LIVE_SOURCE_BUNDLES, AGENT_TRAINING_SOURCE_BUNDLES

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
            action_type = action.type
            if action_type == "strike":
                world.tension_level = min(100.0, world.tension_level + 12.0)
                world.market_stress = min(100.0, world.market_stress + 8.0)
            elif action_type == "sanction":
                world.tension_level = min(100.0, world.tension_level + 5.0)
                world.oil_pressure = min(100.0, world.oil_pressure + 3.0)
            elif action_type == "mobilize":
                world.tension_level = min(100.0, world.tension_level + 7.0)
                world.market_stress = min(100.0, world.market_stress + 4.0)
            elif action_type == "defend":
                world.tension_level = min(100.0, world.tension_level + 2.0)
            elif action_type == "intel_query":
                world.market_stress = min(100.0, world.market_stress + 0.5)
            elif action_type == "deceive":
                world.risk_scores[agent_id] = min(1.0, world.risk_scores.get(agent_id, 0.2) + 0.1)
            elif action_type == "negotiate":
                world.tension_level = max(0.0, world.tension_level - 6.0)
                if action.target and action.target in AGENT_IDS:
                    self._link_agents(world, agent_id, action.target)
            elif action_type == "hold":
                world.tension_level = max(0.0, world.tension_level - 1.0)

            world.behavioral_consistency[agent_id] = self._update_behavioral_consistency(
                world=world,
                agent_id=agent_id,
                action=action,
            )

        world.tension_level = round(world.tension_level, 2)
        world.market_stress = round(world.market_stress, 2)
        world.oil_pressure = round(world.oil_pressure, 2)

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
            strategic_assets = entity_pack.get("assets", {}).get("locations", [])[:8]
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

    def _compute_rewards(self, world: WorldState, episode: EpisodeMetadata) -> dict[str, RewardBreakdown]:
        rewards: dict[str, RewardBreakdown] = {}
        market_component = max(-1.0, min(1.0, 1.0 - ((world.market_stress + world.oil_pressure) / 120.0)))

        for agent_id in AGENT_IDS:
            allied_agents = len(world.coalition_graph.get(agent_id, []))
            coalition_component = max(-1.0, min(1.0, (2.0 * allied_agents / max(len(AGENT_IDS) - 1, 1)) - 1.0))
            if not episode.dense_rewards and world.turn % 10 != 0:
                coalition_component = 0.0
            world.ema_tension[agent_id] = round(
                0.05 * world.tension_level + 0.95 * world.ema_tension.get(agent_id, world.tension_level),
                3,
            )
            escalation_component = round(-world.ema_tension[agent_id] / 100.0, 3)
            behavioral_component = max(
                -1.0,
                min(1.0, world.behavioral_consistency.get(agent_id, 0.5) * 2.0 - 1.0),
            )
            weights = AGENT_REWARD_WEIGHTS[agent_id]
            total = (
                weights[0] * coalition_component
                + weights[1] * escalation_component
                + weights[2] * market_component
                + weights[3] * behavioral_component
            )
            rewards[agent_id] = RewardBreakdown(
                coalition_stability=round(coalition_component, 3),
                escalation_penalty=round(escalation_component, 3),
                market_gain=round(market_component, 3),
                behavioral_consistency=round(behavioral_component, 3),
                total=round(total, 3),
            )
        return rewards

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
