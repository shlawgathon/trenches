from __future__ import annotations

import hashlib
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from trenches_env.agents import AGENT_IDS, AGENT_PROFILES
from trenches_env.entity_knowledge import load_entity_pack
from trenches_env.historical_replay import (
    get_historical_replay,
    severity_distance,
    severity_score,
)
from trenches_env.model_runtime import build_entity_model_bindings
from trenches_env.models import (
    ActionLogEntry,
    AgentAction,
    AgentBeliefEntry,
    AgentBeliefState,
    AgentObservation,
    AssetCondition,
    BlackSwanEvent,
    DataSourceContext,
    EpisodeMetadata,
    EntityModelBinding,
    ExternalSignal,
    HistoricalEvent,
    HistoricalReplayState,
    IntelSnippet,
    LatentEvent,
    LatentEventNarrative,
    LiveControlRequest,
    ObservationProjection,
    OversightIntervention,
    Prediction,
    PredictionAssessment,
    ProviderDiagnosticsResponse,
    ReactionActorOutcome,
    ReactionLogEntry,
    RewardBreakdown,
    SessionState,
    SourcePacket,
    SourceMonitorReport,
    StepTrace,
    StepSessionRequest,
    StepSessionResponse,
    WorldState,
)
from trenches_env.provider_runtime import ProviderDecisionError, ProviderDecisionRequest, ProviderDecisionRuntime
from trenches_env.rl import (
    AGENT_ALLOWED_ACTIONS,
    AGENT_ACTION_ALIGNMENT,
    AGENT_ACTION_IMPACTS,
    AGENT_PREFERRED_COALITIONS,
    AGENT_REWARD_METRIC_CONFIGS,
    AGENT_STATE_ACTION_EFFECTS,
    AGENT_STATE_BASELINES,
    DEFAULT_ACTION_IMPACTS,
    DEFAULT_MAX_TURNS,
    DEFAULT_TRAINING_STAGE,
    TRAINING_STAGE_CONFIGS,
)
from trenches_env.source_catalog import get_source_by_id, get_sources_for_agent
from trenches_env.source_bundles import AGENT_LIVE_SOURCE_BUNDLES, AGENT_TRAINING_SOURCE_BUNDLES
from trenches_env.source_ingestion import SourceHarvester, source_ttl_seconds
from trenches_env.source_monitor import build_source_monitor_report
from trenches_env.scenarios import (
    ScenarioAssetImpact,
    ScenarioDefinition,
    ScenarioLatentEvent,
    ScenarioSignal,
    get_scenario_definition,
    list_scenario_definitions,
    scenario_signals_for_turn,
)

logger = logging.getLogger("trenches.runtime")

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

MAX_PUBLIC_BRIEF_ITEMS = 4
MAX_PRIVATE_BRIEF_ITEMS = 6
MAX_INTEL_SUMMARY_CHARS = 220
MAX_TRAINING_SOURCE_BRIEFS = 2
MAX_LIVE_SOURCE_BRIEFS = 2
MAX_AUTO_REACTION_SIGNALS = 8
MIN_LIVE_AUTO_STEP_MS = 1_000
FORECAST_REWARD_BLEND = 0.35
LOW_FIDELITY_SOURCE_KINDS = {"telegram", "scrape", "video"}
SOURCE_KIND_BASE_RELIABILITY = {
    "structured": 0.8,
    "api": 0.76,
    "rss": 0.7,
    "scrape": 0.58,
    "telegram": 0.46,
    "video": 0.38,
}
SOURCE_DELIVERY_RELIABILITY = {
    "training_core": 0.06,
    "live_demo": 0.0,
}
CONTRADICTION_TOPIC_LABELS = {
    "shipping": "shipping disruption",
    "commodities": "commodity-market disruption",
    "border": "border escalation",
    "corridor": "corridor interdiction",
    "domestic": "domestic stability",
    "cyber": "cyber disruption",
    "market": "market dislocation",
    "humanitarian": "humanitarian fallout",
    "diplomacy": "diplomatic signaling",
}
LATENT_EVENT_TOPIC_KEYWORDS = {
    "shipping": ("shipping", "tanker", "hormuz", "oil", "maritime", "terminal", "seaport", "harbor", "ais", "vessel"),
    "commodities": (
        "gold",
        "silver",
        "copper",
        "lithium",
        "nickel",
        "uranium",
        "phosphate",
        "bauxite",
        "rare earth",
        "rare-earth",
        "commodity",
        "mineral",
        "metals",
        "natural gas",
        "lng",
    ),
    "border": ("rocket", "missile", "border", "galilee", "idf", "drone", "lebanon", "launch", "intercept"),
    "corridor": ("corridor", "bekaa", "syria", "transfer", "logistics", "interdiction", "sustainment"),
    "domestic": ("sanction", "unrest", "protest", "inflation", "currency", "domestic", "regime"),
    "cyber": ("cyber", "outage", "blackout", "cable", "internet", "network", "malware"),
    "market": ("market", "investor", "trade", "stocks", "shares", "bond", "premium", "insurance"),
    "humanitarian": ("humanitarian", "aid", "displacement", "relief", "civilian", "refugee", "shelter"),
    "diplomacy": ("ceasefire", "talk", "negotiat", "summit", "diplomat", "mediat", "channel"),
}
LATENT_EVENT_LINKS = {
    "shipping": ("market",),
    "commodities": ("market", "shipping"),
    "border": ("humanitarian",),
    "corridor": ("border",),
    "cyber": ("market",),
}
BELIEF_TOPIC_PRIORS = {
    "us": {"shipping": 0.12, "commodities": 0.09, "diplomacy": 0.08, "market": 0.06, "border": 0.04},
    "israel": {"border": 0.14, "corridor": 0.08, "diplomacy": 0.04},
    "iran": {"corridor": 0.14, "domestic": 0.1, "shipping": 0.06, "commodities": 0.08},
    "hezbollah": {"border": 0.12, "corridor": 0.1, "domestic": 0.04},
    "gulf": {"shipping": 0.16, "commodities": 0.12, "market": 0.12, "diplomacy": 0.06},
    "oversight": {"cyber": 0.12, "shipping": 0.08, "commodities": 0.08, "border": 0.08, "domestic": 0.08},
}
BELIEF_PERSISTENCE_FLOOR = 0.12
BELIEF_MAX_STALE_TURNS = 4
BELIEF_CONFIRMATION_BONUS = 0.03
BELIEF_CONTRADICTION_PENALTY = 0.14
PUBLIC_STATE_SYNC_FACTORS = {
    "support": 0.62,
    "confidence": 0.6,
    "clarity": 0.58,
    "resilience": 0.66,
    "security": 0.74,
    "continuity": 0.74,
    "stability": 0.72,
    "default": 0.7,
}
ASSET_DECISION_SOURCE_LIMITS: dict[str, tuple[int, int]] = {
    "large": (8, 8),
    "medium-large": (6, 6),
    "medium": (5, 5),
}
PHYSICAL_ASSET_SECTIONS = (
    "locations",
    "fronts",
    "infrastructure",
    "strategic_sites",
    "chokepoints",
    "geospatial_anchors",
    "alliance_anchors",
)
ASSET_PRIORITY_SCORES = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "tracked": 1,
    "linked": 1,
}
ASSET_STATUS_DAMAGE_THRESHOLDS = (
    (25.0, "destroyed"),
    (55.0, "malfunctioning"),
    (85.0, "degraded"),
)

AGENT_PRIMARY_ADVERSARIES: dict[str, tuple[str, ...]] = {
    "us": ("iran",),
    "israel": ("hezbollah", "iran"),
    "iran": ("israel", "us", "gulf"),
    "hezbollah": ("israel",),
    "gulf": ("iran",),
    "oversight": (),
}

AGENT_TOOL_LABELS: dict[str, str] = {
    "us": "shipping security, coalition access, and force-posture tools",
    "israel": "air-defense, reserve, and northern-front tools",
    "iran": "proxy, chokepoint, and regime-security tools",
    "hezbollah": "launch-survivability, logistics, and deniable-pressure tools",
    "gulf": "shipping, infrastructure, and market-stability tools",
    "oversight": "trace, intervention, and autonomy-balancing tools",
}
ASSET_CATEGORY_BIAS: dict[str, dict[str, tuple[str, ...]]] = {
    "us": {
        "strike": ("energy", "port", "chokepoint", "command", "launch-zone"),
        "defend": ("airbase", "base", "naval", "logistics-port", "base-network", "command-system"),
        "deceive": ("command", "command-system", "radar", "air-defense", "theater-anchor"),
        "sanction": ("energy", "port", "logistics-network", "energy-network"),
    },
    "israel": {
        "strike": ("launch-network", "launch-zone", "corridor-node", "logistics-network", "logistics"),
        "defend": ("command", "front", "civil-core", "infrastructure-zone", "offshore-zone", "air-defense"),
        "deceive": ("launch-zone", "command-network", "corridor-node"),
    },
    "iran": {
        "strike": ("airbase", "base", "energy", "energy-port", "port", "chokepoint"),
        "defend": ("command", "energy-network", "maritime-control-zone", "command-and-industry-network"),
        "deceive": ("chokepoint", "port", "naval", "maritime-box", "maritime-access"),
    },
    "hezbollah": {
        "strike": ("front", "civil-core", "command", "offshore-zone", "infrastructure-zone"),
        "defend": ("launch-network", "front", "corridor-node", "command-network", "logistics-network"),
        "deceive": ("launch-zone", "command-network", "logistics-network"),
    },
    "gulf": {
        "strike": ("energy-network", "energy", "energy-port", "port", "chokepoint", "base"),
        "defend": ("port", "energy", "energy-port", "capital", "infrastructure-protection", "chokepoint"),
        "deceive": ("energy-port", "port", "chokepoint"),
        "sanction": ("energy", "energy-port", "port", "logistics-network"),
    },
    "oversight": {
        "defend": ("chokepoint", "theater", "civil-center"),
    },
}


class FogOfWarDiplomacyEnv:
    """OpenEnv-compatible scaffolding for the crisis simulator.

    The concrete OpenEnv inheritance point can be added once the package
    dependency is pinned. For now this class owns the transition logic, state
    construction, and observation projection needed by the session API.
    """

    def __init__(
        self,
        source_harvester: SourceHarvester | None = None,
        provider_runtime: ProviderDecisionRuntime | None = None,
    ) -> None:
        self._rng = random.Random()
        self._source_harvester = source_harvester or SourceHarvester(auto_start=False)
        self._provider_runtime = provider_runtime or ProviderDecisionRuntime()
        self._source_warm_start_enabled = False

    def enable_source_warm_start(self) -> "FogOfWarDiplomacyEnv":
        self._source_warm_start_enabled = True
        return self

    def create_session(
        self,
        seed: int | None = None,
        session_id: str | None = None,
        training_agent: str = "us",
        training_stage: str = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
        scenario_id: str | None = None,
        replay_id: str | None = None,
        replay_start_index: int | None = None,
    ) -> SessionState:
        resolved_session_id = session_id or str(uuid4())
        self._seed(seed)
        scenario = get_scenario_definition(scenario_id)
        world = self._initial_world()
        self._apply_scenario(world, scenario)
        historical_replay = self._initialize_historical_replay(
            world=world,
            training_agent=training_agent,
            replay_id=replay_id,
            replay_start_index=replay_start_index,
        )
        episode = self._build_episode_metadata(
            training_stage=training_stage,
            max_turns=max_turns,
            scenario=scenario,
            historical_replay=historical_replay,
        )
        belief_state = self._initialize_belief_state(world, episode)
        observations = self._build_observations(
            world,
            episode,
            belief_state=belief_state,
            historical_replay=historical_replay,
        )
        rewards = {agent_id: RewardBreakdown() for agent_id in AGENT_IDS}
        session = SessionState(
            session_id=resolved_session_id,
            seed=seed,
            world=world,
            observations=observations,
            belief_state=belief_state,
            rewards=rewards,
            historical_replay=historical_replay,
            model_bindings=self._build_model_bindings(),
            episode=episode,
        )
        last_sync_at = self._source_harvester.last_sync_at()
        if last_sync_at is not None:
            session.live.last_source_sync_at = last_sync_at
        self._update_live_source_state(session)
        return session

    def reset_session(
        self,
        session_id: str,
        seed: int | None = None,
        training_agent: str = "us",
        training_stage: str = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
        scenario_id: str | None = None,
        replay_id: str | None = None,
        replay_start_index: int | None = None,
    ) -> SessionState:
        return self.create_session(
            seed=seed,
            session_id=session_id,
            training_agent=training_agent,
            training_stage=training_stage,
            max_turns=max_turns,
            scenario_id=scenario_id,
            replay_id=replay_id,
            replay_start_index=replay_start_index,
        )

    def configure_live_session(self, session: SessionState, request: LiveControlRequest) -> SessionState:
        updated = session.model_copy(deep=True)
        if request.enabled and not updated.episode.live_mode_capable:
            raise ValueError(
                f"Live mode is only supported for {DEFAULT_TRAINING_STAGE} sessions."
            )
        updated.live.enabled = request.enabled
        updated.live.auto_step = request.auto_step
        updated.live.poll_interval_ms = max(request.poll_interval_ms, MIN_LIVE_AUTO_STEP_MS)
        updated.live.started_at = datetime.now(timezone.utc) if request.enabled else None
        updated.live.last_source_sync_at = datetime.now(timezone.utc) if request.enabled else None
        updated.live.last_auto_step_at = None
        updated.live.reacted_packet_fetched_at = {}
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
            self._source_harvester.warm_start_agents(
                include_live=True,
                max_training_sources=1,
                max_live_sources=1,
            )
            return self.refresh_session_sources(updated, phase="seed")
        return self.refresh_session_sources(updated)

    def step_session(self, session: SessionState, request: StepSessionRequest) -> StepSessionResponse:
        next_session = session.model_copy(deep=True)
        before_tension = next_session.world.tension_level
        prior_latent_event_ids = {event.event_id for event in next_session.world.latent_events}
        next_session.world.turn += 1

        self._inject_external_signals(next_session.world, request.external_signals)
        resolved_actions = dict(request.actions)
        oversight = OversightIntervention()
        if next_session.episode.oversight_enabled:
            oversight = self._compute_oversight(next_session.world, resolved_actions, request.external_signals)
            resolved_actions = self._resolve_oversight_actions(resolved_actions, oversight)
        next_session.world.last_actions = list(resolved_actions.values())
        self._apply_actions(next_session.world, resolved_actions)
        if next_session.episode.oversight_enabled:
            self._apply_oversight(next_session.world, oversight)
        self._resync_public_events(next_session.world)
        latent_event_ids = [
            event.event_id for event in next_session.world.latent_events if event.event_id not in prior_latent_event_ids
        ]
        revealed_event, prediction_assessments = self._advance_historical_replay(
            next_session,
            request.predictions,
        )
        next_session.rewards = self._compute_rewards(world=next_session.world, episode=next_session.episode)
        if prediction_assessments:
            self._apply_forecast_rewards(next_session.rewards, prediction_assessments)
        next_session.model_bindings = self._build_model_bindings()
        next_session.belief_state = self._update_belief_state(next_session)
        next_session.observations = self._build_observations(
            next_session.world,
            next_session.episode,
            include_live_sources=next_session.live.enabled,
            belief_state=next_session.belief_state,
            historical_replay=next_session.historical_replay,
        )
        next_session.recent_traces.append(
            StepTrace(
                turn=next_session.world.turn,
                tension_before=before_tension,
                tension_after=next_session.world.tension_level,
                actions=resolved_actions,
                predictions=request.predictions,
                prediction_assessments=prediction_assessments,
                revealed_event=revealed_event,
                rewards=next_session.rewards,
                oversight=oversight,
            )
        )
        next_session.recent_traces = next_session.recent_traces[-25:]
        next_session.action_log.extend(self._build_action_log_entries(next_session, resolved_actions))
        next_session.action_log = next_session.action_log[-250:]
        if request.external_signals:
            next_session.reaction_log.append(
                self._build_reaction_log_entry(
                    session=next_session,
                    signals=request.external_signals,
                    latent_event_ids=latent_event_ids,
                    actions=resolved_actions,
                    oversight=oversight,
                    tension_before=before_tension,
                )
            )
            next_session.reaction_log = next_session.reaction_log[-100:]
        oversight_prediction = self._generate_oversight_prediction(next_session, resolved_actions, before_tension)
        if oversight_prediction:
            next_session.prediction_log.append(oversight_prediction)
            next_session.prediction_log = next_session.prediction_log[-50:]
        self._update_live_source_state(next_session, phase="background" if next_session.live.enabled else None)
        next_session.updated_at = datetime.now(timezone.utc)

        return StepSessionResponse(
            session=next_session,
            oversight=oversight,
            done=(
                next_session.world.turn >= next_session.episode.max_turns
                or next_session.world.tension_level >= 95.0
                or (
                    next_session.historical_replay.enabled
                    and next_session.historical_replay.current_event_index
                    >= len(next_session.historical_replay.ground_truth_timeline) - 1
                )
            ),
        )

    def refresh_session_sources(
        self,
        session: SessionState,
        force: bool = False,
        *,
        phase: str | None = None,
    ) -> SessionState:
        updated = session.model_copy(deep=True)
        if self._should_collect_sources(updated):
            if force:
                self._source_harvester.refresh_agents(include_live=updated.live.enabled, force=True)
        updated.model_bindings = self._build_model_bindings()
        updated.belief_state = self._update_belief_state(updated)
        updated.observations = self._build_observations(
            updated.world,
            updated.episode,
            include_live_sources=updated.live.enabled,
            belief_state=updated.belief_state,
            historical_replay=updated.historical_replay,
        )
        last_sync_at = self._source_harvester.last_sync_at()
        if last_sync_at is not None:
            updated.live.last_source_sync_at = last_sync_at
        self._update_live_source_state(updated, phase=phase)
        updated.updated_at = datetime.now(timezone.utc)
        return updated

    def background_refresh_session(self, session: SessionState) -> SessionState:
        updated = session.model_copy(deep=True)
        if self._should_collect_sources(updated):
            self._source_harvester.refresh_due_batch(include_live=updated.live.enabled)
        return self.refresh_session_sources(updated, phase="background" if updated.live.enabled else None)

    @staticmethod
    def _should_collect_sources(session: SessionState) -> bool:
        return not session.historical_replay.enabled

    def source_monitor(self, session: SessionState) -> SourceMonitorReport:
        return build_source_monitor_report(session, harvester=self._source_harvester)

    def provider_diagnostics(self, session: SessionState) -> ProviderDiagnosticsResponse:
        bindings = session.model_bindings or self._build_model_bindings()
        return ProviderDiagnosticsResponse(
            agents=self._provider_runtime.diagnostics(bindings)
        )

    def list_scenarios(self) -> list[ScenarioDefinition]:
        return list_scenario_definitions()

    def scenario_turn_signals(self, scenario_id: str | None, turn: int) -> list[ExternalSignal]:
        return scenario_signals_for_turn(scenario_id, turn)

    def maybe_auto_step_live_session(self, session: SessionState) -> SessionState:
        # Skip redundant refresh — the background runner already called background_refresh_session
        if not session.live.enabled or not session.live.auto_step:
            return session

        now = datetime.now(timezone.utc)
        if not self._live_auto_step_due(session, now):
            return session

        signals, reacted_packets = self._collect_live_external_signals(session)
        actions = self.resolve_policy_actions(session, signals)
        result = self.step_session(
            session,
            StepSessionRequest(actions=actions, external_signals=signals),
        )
        next_session = result.session
        next_session.live.last_auto_step_at = now
        next_session.live.reacted_packet_fetched_at.update(reacted_packets)
        next_session.updated_at = datetime.now(timezone.utc)
        return next_session

    def _update_live_source_state(self, session: SessionState, *, phase: str | None = None) -> None:
        packet_by_id: dict[str, SourcePacket] = {}
        pending_by_agent: dict[str, int] = {}

        for agent_id, observation in session.observations.items():
            queue_packets = observation.live_source_packets if session.live.enabled else observation.training_source_packets
            pending_by_agent[agent_id] = sum(1 for packet in queue_packets if packet.status == "pending")
            for packet in observation.source_packets:
                packet_by_id[packet.source_id] = packet

        ready = sum(1 for packet in packet_by_id.values() if packet.status == "ok")
        pending = sum(1 for packet in packet_by_id.values() if packet.status == "pending")
        error = sum(1 for packet in packet_by_id.values() if packet.status == "error")
        total = len(packet_by_id)

        next_phase = phase or session.live.hydration.phase
        if total == 0 or pending == 0:
            next_phase = "steady"
        elif phase is None and next_phase == "steady":
            next_phase = "background" if ready + error > 0 else "seed"

        hydration_type = type(session.live).HydrationStatus
        session.live.hydration = hydration_type(
            phase=next_phase,
            total=total,
            ready=ready,
            pending=pending,
            error=error,
        )
        session.live.source_queue_sizes = pending_by_agent

    def _live_auto_step_due(self, session: SessionState, now: datetime) -> bool:
        last_auto_step_at = session.live.last_auto_step_at
        if last_auto_step_at is None:
            return True
        return now - last_auto_step_at >= timedelta(
            milliseconds=max(session.live.poll_interval_ms, MIN_LIVE_AUTO_STEP_MS)
        )

    def _collect_live_external_signals(
        self,
        session: SessionState,
    ) -> tuple[list[ExternalSignal], dict[str, datetime]]:
        newest_packets: dict[str, tuple[str, SourcePacket]] = {}
        for agent_id, observation in session.observations.items():
            for packet in observation.source_packets:
                if packet.status != "ok" or not packet.summary or packet.fetched_at is None:
                    continue
                if not self._is_source_packet_fresh(packet):
                    continue
                last_reacted = session.live.reacted_packet_fetched_at.get(packet.source_id)
                if last_reacted is not None and packet.fetched_at <= last_reacted:
                    continue
                cached = newest_packets.get(packet.source_id)
                if cached is None or packet.fetched_at > (cached[1].fetched_at or datetime.fromtimestamp(0, tz=timezone.utc)):
                    newest_packets[packet.source_id] = (agent_id, packet)

        ordered_packets = sorted(
            newest_packets.values(),
            key=lambda item: item[1].fetched_at or datetime.fromtimestamp(0, tz=timezone.utc),
            reverse=True,
        )[:MAX_AUTO_REACTION_SIGNALS]

        signals: list[ExternalSignal] = []
        reacted_packets: dict[str, datetime] = {}
        for agent_id, packet in ordered_packets:
            signal = self._packet_to_external_signal(agent_id, packet)
            signals.append(signal)
            if packet.fetched_at is not None:
                reacted_packets[packet.source_id] = packet.fetched_at
        return signals, reacted_packets

    def _packet_to_external_signal(self, agent_id: str, packet: SourcePacket) -> ExternalSignal:
        packet_text = " ".join([packet.source_name, packet.summary, *packet.sample_items]).lower()
        categories = self._classify_signal_categories(packet_text)
        severity = 0.35
        if "attack" in categories:
            severity = max(severity, 0.82)
        if "shipping" in categories:
            severity = max(severity, 0.72)
        if "commodities" in categories:
            severity = max(severity, 0.62)
        if "cyber" in categories or "unrest" in categories:
            severity = max(severity, 0.64)
        if "market" in categories:
            severity = max(severity, 0.56)
        if "humanitarian" in categories or "diplomacy" in categories:
            severity = max(severity, 0.44)
        if packet.delivery == "live_demo":
            severity += 0.05

        lead = packet.sample_items[0] if packet.sample_items else packet.summary
        return ExternalSignal(
            source=packet.source_name,
            headline=self._clip_summary(f"{packet.source_name}: {lead}", limit=160),
            region=None if agent_id == "oversight" else agent_id,
            tags=sorted(categories | {agent_id, packet.kind, packet.delivery}),
            severity=round(min(severity, 0.95), 2),
        )

    def _classify_signal_categories(self, text: str) -> set[str]:
        categories: set[str] = set()
        if self._signal_mentions(
            text,
            "strike",
            "rocket",
            "missile",
            "drone",
            "attack",
            "raid",
            "blast",
            "explosion",
            "intercept",
            "retaliat",
            "launch",
        ):
            categories.add("attack")
        if self._signal_mentions(
            text,
            "shipping",
            "tanker",
            "vessel",
            "hormuz",
            "oil",
            "terminal",
            "seaport",
            "harbor",
            "maritime",
            "strait",
            "pipeline",
            "red sea",
        ):
            categories.add("shipping")
        if self._signal_mentions(
            text,
            "gold",
            "silver",
            "copper",
            "lithium",
            "nickel",
            "uranium",
            "phosphate",
            "bauxite",
            "rare earth",
            "rare-earth",
            "commodity",
            "mineral",
            "metals",
            "natural gas",
            "lng",
        ):
            categories.add("commodities")
        if self._signal_mentions(text, "ceasefire", "talk", "negotiat", "summit", "diplomat", "mediat"):
            categories.add("diplomacy")
        if self._signal_mentions(text, "humanitarian", "aid", "displacement", "relief", "civilian", "refugee"):
            categories.add("humanitarian")
        if self._signal_mentions(text, "cyber", "internet", "blackout", "outage", "malware", "network"):
            categories.add("cyber")
        if self._signal_mentions(text, "protest", "unrest", "sanction", "inflation", "currency", "black market"):
            categories.add("unrest")
        if self._signal_mentions(text, "market", "investor", "trade", "stocks", "shares", "bond", "price", "futures"):
            categories.add("market")
        return categories or {"general"}

    def resolve_policy_actions(
        self,
        session: SessionState,
        signals: list[ExternalSignal],
        *,
        preset_actions: dict[str, AgentAction] | None = None,
        agent_ids: list[str] | None = None,
    ) -> dict[str, AgentAction]:
        actions: dict[str, AgentAction] = dict(preset_actions or {})
        target_agent_ids = agent_ids or list(AGENT_IDS)
        provider_results: dict[str, tuple[AgentAction | None, str | None]] = {}
        for agent_id, action in actions.items():
            self._validate_action(agent_id, action)

        unresolved_agent_ids = [agent_id for agent_id in target_agent_ids if agent_id not in actions]
        provider_ready_agent_ids = [
            agent_id
            for agent_id in unresolved_agent_ids
            if (binding := session.model_bindings.get(agent_id)) is not None and binding.ready_for_inference
        ]

        if len(provider_ready_agent_ids) > 1:
            with ThreadPoolExecutor(max_workers=len(provider_ready_agent_ids)) as executor:
                future_to_agent_id = {
                    executor.submit(self._resolve_provider_action, session, agent_id, signals): agent_id
                    for agent_id in provider_ready_agent_ids
                }
                for future in as_completed(future_to_agent_id):
                    provider_results[future_to_agent_id[future]] = future.result()
        elif len(provider_ready_agent_ids) == 1:
            agent_id = provider_ready_agent_ids[0]
            provider_results[agent_id] = self._resolve_provider_action(session, agent_id, signals)

        for agent_id in target_agent_ids:
            if agent_id in actions:
                continue
            provider_action, provider_error = provider_results.get(agent_id, (None, None))
            if agent_id not in provider_results:
                provider_action, provider_error = self._resolve_provider_action(session, agent_id, signals)
            if provider_action is not None:
                actions[agent_id] = provider_action
                continue
            # No real provider available — skip this agent (no heuristic fallback)
            if provider_error:
                logger.info("Agent %s skipped: provider error — %s", agent_id, provider_error)
            else:
                logger.debug("Agent %s skipped: no provider configured", agent_id)
        return actions

    def _resolve_provider_action(
        self,
        session: SessionState,
        agent_id: str,
        signals: list[ExternalSignal],
    ) -> tuple[AgentAction | None, str | None]:
        binding = session.model_bindings.get(agent_id)
        if binding is None or not binding.ready_for_inference:
            return None, None

        try:
            action = self._provider_runtime.decide_action(
                ProviderDecisionRequest(
                    agent_id=agent_id,
                    binding=binding,
                    observation=session.observations[agent_id],
                    external_signals=signals,
                )
            )
            self._validate_action(agent_id, action)
            return action, None
        except ProviderDecisionError as exc:
            logger.warning(
                "provider.fallback session=%s agent=%s provider=%s model=%s signals=%s error=%s",
                session.session_id,
                agent_id,
                binding.provider,
                binding.model_name,
                len(signals),
                str(exc),
            )
            return None, str(exc)

    def _select_live_actions(
        self,
        session: SessionState,
        signals: list[ExternalSignal],
    ) -> dict[str, AgentAction]:
        return self.resolve_policy_actions(session, signals)

    def _build_signal_context(
        self,
        agent_id: str,
        signals: list[ExternalSignal],
    ) -> tuple[dict[str, float], str]:
        signal_context = {
            "attack": 0.0,
            "shipping": 0.0,
            "commodities": 0.0,
            "diplomacy": 0.0,
            "humanitarian": 0.0,
            "cyber": 0.0,
            "unrest": 0.0,
            "market": 0.0,
            "general": 0.0,
            "pressure": 0.0,
            "relevant_count": 0.0,
        }
        top_headline = ""
        top_severity = 0.0

        for signal in signals:
            if agent_id != "oversight":
                affected_agents = set(self._infer_affected_agents(signal))
                if agent_id not in affected_agents and signal.region != agent_id:
                    continue

            text = f"{signal.headline} {' '.join(signal.tags)} {(signal.region or '')}".lower()
            categories = self._classify_signal_categories(text)
            weight = max(0.2, signal.severity)
            signal_context["relevant_count"] += 1.0
            for category in categories:
                signal_context[category] = signal_context.get(category, 0.0) + weight
            if signal.severity >= top_severity:
                top_severity = signal.severity
                top_headline = signal.headline

        signal_context["pressure"] = sum(
            signal_context[category]
            for category in ("attack", "shipping", "commodities", "diplomacy", "humanitarian", "cyber", "unrest", "market")
        ) + 0.3 * signal_context["general"]
        return signal_context, top_headline

    def _score_live_action(
        self,
        *,
        agent_id: str,
        action_type: str,
        session: SessionState,
        signal_context: dict[str, float],
    ) -> float:
        observation = session.observations[agent_id]
        metric_gain = 0.0
        action_effects = AGENT_STATE_ACTION_EFFECTS[agent_id].get(action_type, {})

        for metric, config in AGENT_REWARD_METRIC_CONFIGS[agent_id].items():
            current_value = observation.strategic_state.get(
                metric,
                AGENT_STATE_BASELINES[agent_id].get(metric, 50.0),
            )
            projected_value = self._clamp_percent(current_value + action_effects.get(metric, 0.0))
            before_score = self._target_score(current_value, config.target, config.tolerance)
            after_score = self._target_score(projected_value, config.target, config.tolerance)
            metric_gain += (after_score - before_score) * config.weight

        doctrinal_fit = AGENT_ACTION_ALIGNMENT[agent_id].get(action_type, 0.0)
        signal_bias = self._live_signal_action_bias(agent_id, action_type, signal_context)
        belief_bias = self._belief_action_bias(
            agent_id,
            action_type,
            session.belief_state.get(agent_id, AgentBeliefState(agent_id=agent_id)),
        )
        asset_pressure = self._asset_pressure(session.world, agent_id)
        if action_type in {"defend", "intel_query"}:
            signal_bias += 0.22 * asset_pressure
        elif action_type == "negotiate":
            signal_bias += 0.10 * asset_pressure
        elif action_type == "hold":
            signal_bias -= 0.14 * asset_pressure
        continuity_bonus = 0.0
        if any(action.actor == agent_id and action.type == action_type for action in session.world.last_actions):
            continuity_bonus = 0.06

        escalation_penalty = 0.0
        if session.world.tension_level >= 78.0 and action_type in {"strike", "mobilize", "deceive", "sanction"}:
            escalation_penalty += 0.28
        if signal_context["diplomacy"] >= 0.6 and action_type in {"strike", "mobilize", "deceive"}:
            escalation_penalty += 0.18
        if signal_context["attack"] >= 0.65 and action_type == "hold":
            escalation_penalty += 0.18
        if signal_context["shipping"] >= 0.55 and agent_id in {"us", "gulf"} and action_type in {"strike", "deceive"}:
            escalation_penalty += 0.2
        if asset_pressure >= 0.35 and action_type in {"strike", "mobilize", "deceive"}:
            escalation_penalty += 0.16

        if agent_id == "oversight" and action_type == "oversight_review":
            escalation_penalty -= min(0.25, signal_context["pressure"] * 0.2)

        return metric_gain * 1.8 + doctrinal_fit * 0.28 + signal_bias + belief_bias + continuity_bonus - escalation_penalty

    @staticmethod
    def _belief_action_bias(
        agent_id: str,
        action_type: str,
        belief_state: AgentBeliefState,
    ) -> float:
        topic_weights = {belief.topic: belief.confidence for belief in belief_state.beliefs[:4]}
        shipping = topic_weights.get("shipping", 0.0)
        commodities = topic_weights.get("commodities", 0.0)
        border = topic_weights.get("border", 0.0)
        corridor = topic_weights.get("corridor", 0.0)
        diplomacy = topic_weights.get("diplomacy", 0.0)
        cyber = topic_weights.get("cyber", 0.0)
        domestic = topic_weights.get("domestic", 0.0)

        if agent_id in {"us", "gulf"}:
            return {
                "defend": 0.22 * shipping,
                "negotiate": 0.16 * diplomacy + 0.08 * shipping + 0.10 * commodities,
                "intel_query": 0.14 * cyber + 0.10 * commodities,
                "strike": 0.06 * border - 0.12 * diplomacy,
            }.get(action_type, 0.0)
        if agent_id == "israel":
            return {
                "defend": 0.2 * border,
                "strike": 0.12 * border,
                "intel_query": 0.1 * corridor + 0.08 * cyber,
                "negotiate": 0.1 * diplomacy - 0.08 * border,
            }.get(action_type, 0.0)
        if agent_id in {"iran", "hezbollah"}:
            return {
                "mobilize": 0.16 * corridor + 0.08 * border,
                "deceive": 0.12 * cyber + 0.08 * corridor + 0.06 * commodities,
                "defend": 0.1 * border + 0.08 * domestic,
                "negotiate": 0.1 * diplomacy - 0.08 * corridor - 0.04 * commodities,
            }.get(action_type, 0.0)
        return {
            "oversight_review": 0.18 * (shipping + commodities + border + corridor + cyber + domestic),
            "negotiate": 0.12 * diplomacy,
            "intel_query": 0.1 * cyber + 0.08 * commodities,
        }.get(action_type, 0.0)

    def _live_signal_action_bias(
        self,
        agent_id: str,
        action_type: str,
        signal_context: dict[str, float],
    ) -> float:
        attack = signal_context["attack"]
        shipping = signal_context["shipping"]
        commodities = signal_context["commodities"]
        diplomacy = signal_context["diplomacy"]
        humanitarian = signal_context["humanitarian"]
        cyber = signal_context["cyber"]
        unrest = signal_context["unrest"]
        market = signal_context["market"]
        pressure = signal_context["pressure"]

        if agent_id == "us":
            return {
                "defend": 0.34 * shipping + 0.22 * attack + 0.18 * cyber + 0.14 * market,
                "negotiate": 0.30 * diplomacy + 0.18 * humanitarian + 0.16 * market + 0.08 * attack + 0.10 * commodities,
                "intel_query": 0.24 * cyber + 0.18 * attack + 0.12 * unrest + 0.12 * commodities,
                "mobilize": 0.16 * attack + 0.12 * shipping + 0.06 * commodities,
                "sanction": 0.18 * unrest + 0.10 * cyber,
                "strike": 0.08 * attack - 0.18 * diplomacy - 0.12 * humanitarian,
                "deceive": 0.04 * attack - 0.10 * diplomacy,
                "hold": 0.10 * diplomacy - 0.10 * attack,
            }.get(action_type, 0.0)

        if agent_id == "israel":
            return {
                "defend": 0.38 * attack + 0.14 * cyber,
                "strike": 0.22 * attack - 0.12 * humanitarian,
                "mobilize": 0.18 * attack,
                "intel_query": 0.20 * cyber + 0.16 * attack,
                "negotiate": 0.16 * diplomacy + 0.08 * humanitarian,
                "hold": 0.08 * diplomacy - 0.12 * attack,
                "deceive": 0.12 * attack,
                "sanction": 0.06 * unrest,
            }.get(action_type, 0.0)

        if agent_id == "iran":
            return {
                "mobilize": 0.26 * shipping + 0.18 * attack + 0.12 * commodities,
                "deceive": 0.22 * attack + 0.18 * shipping + 0.14 * cyber + 0.08 * commodities,
                "defend": 0.26 * unrest + 0.12 * attack,
                "intel_query": 0.18 * cyber + 0.16 * unrest,
                "negotiate": 0.14 * diplomacy - 0.14 * attack - 0.06 * commodities,
                "strike": 0.12 * attack + 0.10 * shipping + 0.06 * commodities - 0.18 * diplomacy,
                "hold": 0.08 * diplomacy - 0.06 * shipping - 0.04 * commodities,
                "sanction": 0.08 * unrest,
            }.get(action_type, 0.0)

        if agent_id == "hezbollah":
            return {
                "defend": 0.28 * attack,
                "deceive": 0.24 * attack + 0.14 * cyber,
                "mobilize": 0.18 * attack,
                "strike": 0.14 * attack - 0.14 * humanitarian,
                "negotiate": 0.22 * humanitarian + 0.14 * diplomacy - 0.18 * attack,
                "hold": 0.18 * humanitarian + 0.10 * diplomacy - 0.12 * attack,
                "intel_query": 0.12 * cyber + 0.10 * attack,
                "sanction": 0.04 * unrest,
            }.get(action_type, 0.0)

        if agent_id == "gulf":
            return {
                "defend": 0.38 * shipping + 0.18 * attack + 0.16 * market,
                "negotiate": 0.28 * diplomacy + 0.24 * market + 0.14 * humanitarian + 0.14 * commodities,
                "intel_query": 0.22 * shipping + 0.14 * cyber + 0.18 * commodities,
                "mobilize": 0.16 * attack + 0.10 * shipping + 0.04 * commodities - 0.12 * market,
                "hold": 0.12 * diplomacy - 0.12 * shipping - 0.06 * commodities,
                "strike": 0.04 * attack + 0.04 * commodities - 0.24 * market,
                "sanction": 0.06 * unrest + 0.04 * commodities - 0.10 * market,
                "deceive": 0.04 * commodities - 0.12 * market,
            }.get(action_type, 0.0)

        return {
            "oversight_review": 0.34 * pressure + 0.20 * attack + 0.16 * shipping + 0.14 * commodities,
            "intel_query": 0.24 * cyber + 0.18 * attack + 0.10 * unrest + 0.14 * commodities,
            "negotiate": 0.20 * diplomacy + 0.12 * humanitarian,
            "defend": 0.16 * attack + 0.12 * shipping,
            "hold": 0.08 * diplomacy - 0.08 * pressure,
        }.get(action_type, 0.0)

    def _select_live_action_target(
        self,
        agent_id: str,
        action_type: str,
        session: SessionState,
        signal_context: dict[str, float],
    ) -> str | None:
        if action_type == "defend":
            return agent_id if agent_id != "oversight" else None

        if action_type == "negotiate":
            if agent_id == "us":
                return "gulf" if signal_context["shipping"] >= signal_context["attack"] else "israel"
            if agent_id == "israel":
                return "us"
            if agent_id == "iran":
                return "hezbollah"
            if agent_id == "hezbollah":
                return "iran"
            if agent_id == "gulf":
                return "us"
            if agent_id == "oversight":
                return None

        if action_type in {"strike", "sanction"}:
            adversaries = AGENT_PRIMARY_ADVERSARIES.get(agent_id, ())
            if not adversaries:
                return None
            if agent_id == "iran" and signal_context["shipping"] > signal_context["attack"]:
                return "gulf"
            return adversaries[0]

        return None

    def _event_driver_label(self, signal_context: dict[str, float]) -> str:
        ranked_categories = sorted(
            (
                (category, value)
                for category, value in signal_context.items()
                if category in {"attack", "shipping", "commodities", "diplomacy", "humanitarian", "cyber", "unrest", "market"}
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked_categories or ranked_categories[0][1] <= 0.0:
            return "regional source refresh"

        top_category = ranked_categories[0][0]
        return {
            "attack": "cross-border attack reporting",
            "shipping": "shipping-lane disruption reporting",
            "commodities": "commodity-market disruption reporting",
            "diplomacy": "diplomatic movement",
            "humanitarian": "humanitarian pressure",
            "cyber": "cyber and trace uncertainty",
            "unrest": "domestic instability",
            "market": "market stress",
        }.get(top_category, "regional source refresh")

    def _build_auto_action_summary(
        self,
        agent_id: str,
        action_type: str,
        target: str | None,
        driver: str,
    ) -> str:
        tool_label = AGENT_TOOL_LABELS[agent_id]
        if action_type == "hold":
            return f"Hold with {tool_label} in reserve while {driver} clarifies."
        if action_type == "negotiate":
            target_label = target or "regional counterparts"
            return f"Use {tool_label} to negotiate with {target_label} around {driver}."
        if action_type == "sanction":
            target_label = target or "the pressure source"
            return f"Apply economic pressure tools against {target_label} after {driver}."
        if action_type == "strike":
            target_label = target or "the active threat node"
            return f"Use kinetic tools against {target_label} in response to {driver}."
        if action_type == "defend":
            target_label = target or agent_id
            return f"Harden {target_label} with {tool_label} as {driver} comes in."
        if action_type == "intel_query":
            return f"Pull more collection through {tool_label} before escalating beyond {driver}."
        if action_type == "mobilize":
            return f"Shift readiness and posture with {tool_label} around {driver}."
        if action_type == "deceive":
            return f"Use deniable signaling and masking tools while {driver} unfolds."
        return f"Run an oversight review with {tool_label} against {driver}."

    def shutdown(self) -> None:
        self._source_harvester.stop()
        self._provider_runtime.close()

    def _seed(self, seed: int | None) -> None:
        if seed is not None:
            self._rng.seed(seed)

    def _initial_world(self) -> WorldState:
        latent_state = {agent_id: metrics.copy() for agent_id, metrics in AGENT_STATE_BASELINES.items()}
        baseline_event = LatentEvent(
            event_id="baseline-posture",
            topic="diplomacy",
            status="active",
            severity=0.45,
            visibility="public",
            reliability=0.74,
            origin="scenario",
            affected_agents=["us", "israel", "iran", "hezbollah", "gulf"],
            started_at_turn=0,
            last_updated_turn=0,
            decay_rate=0.03,
            narratives=[
                LatentEventNarrative(
                    framing="baseline",
                    summary="Regional alert posture is elevated after a contested strike window.",
                    confidence=0.74,
                    public=True,
                ),
                LatentEventNarrative(
                    framing="concealed",
                    summary="Privately, all major actors assess that deterrence signaling remains brittle and prone to misread escalation.",
                    confidence=0.68,
                    public=False,
                ),
            ],
        )
        return WorldState(
            tension_level=50.0,
            market_stress=28.0,
            oil_pressure=36.0,
            latent_state=latent_state,
            latent_events=[baseline_event],
            actor_state={agent_id: metrics.copy() for agent_id, metrics in latent_state.items()},
            asset_state=self._initial_asset_state(),
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

    def _build_episode_metadata(
        self,
        training_stage: str,
        max_turns: int | None,
        *,
        scenario: ScenarioDefinition,
        historical_replay: HistoricalReplayState | None = None,
    ) -> EpisodeMetadata:
        stage_config = TRAINING_STAGE_CONFIGS[training_stage]
        resolved_max_turns = max_turns or DEFAULT_MAX_TURNS
        replay_event_count = len((historical_replay or HistoricalReplayState()).ground_truth_timeline)
        if historical_replay is not None and historical_replay.enabled:
            remaining_events = max(1, replay_event_count - historical_replay.start_event_index - 1)
            resolved_max_turns = min(resolved_max_turns, remaining_events)
        return EpisodeMetadata(
            max_turns=resolved_max_turns,
            training_stage=training_stage,
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            scenario_tags=list(scenario.tags),
            dense_rewards=stage_config["dense_rewards"],
            sparse_rewards=not stage_config["dense_rewards"],
            fog_of_war=stage_config["fog_of_war"],
            oversight_enabled=stage_config["oversight_enabled"],
            live_mode_capable=stage_config["live_mode_capable"],
            replay_mode=historical_replay is not None and historical_replay.enabled,
            replay_id=historical_replay.replay_id if historical_replay is not None and historical_replay.enabled else None,
            replay_event_count=replay_event_count,
        )

    def _initialize_historical_replay(
        self,
        *,
        world: WorldState,
        training_agent: str,
        replay_id: str | None,
        replay_start_index: int | None,
    ) -> HistoricalReplayState:
        if replay_id is None:
            return HistoricalReplayState()

        resolved_replay_id = replay_id
        replay = get_historical_replay(resolved_replay_id)
        if len(replay.events) < 2:
            raise ValueError(f"Replay {resolved_replay_id} must contain at least two events.")

        max_start_index = len(replay.events) - 2
        if replay_start_index is None:
            start_index = self._rng.randint(0, max_start_index)
        else:
            if replay_start_index < 0 or replay_start_index > max_start_index:
                raise ValueError(
                    f"Replay start index {replay_start_index} is outside the valid range 0-{max_start_index}."
                )
            start_index = replay_start_index

        state = HistoricalReplayState(
            enabled=True,
            replay_id=replay.replay_id,
            replay_name=replay.name,
            training_agent=training_agent,
            start_event_index=start_index,
            current_event_index=start_index,
            ground_truth_timeline=[event.model_copy(deep=True) for event in replay.events],
        )
        initial_event = state.ground_truth_timeline[start_index].model_copy(deep=True)
        self._apply_historical_event(world, initial_event)
        state.visible_event_ids.append(initial_event.event_id)
        state.last_revealed_event = initial_event
        return state

    def _advance_historical_replay(
        self,
        session: SessionState,
        predictions: dict[str, Prediction],
    ) -> tuple[HistoricalEvent | None, dict[str, PredictionAssessment]]:
        replay = session.historical_replay
        if not replay.enabled:
            return None, {}

        normalized_predictions = self._normalize_predictions(session, predictions)
        if normalized_predictions:
            session.prediction_log.extend(normalized_predictions.values())

        next_index = replay.current_event_index + 1
        if next_index >= len(replay.ground_truth_timeline):
            return None, {}

        revealed_event = replay.ground_truth_timeline[next_index].model_copy(deep=True)
        assessments = {
            agent_id: self._score_prediction(prediction=prediction, event=revealed_event)
            for agent_id, prediction in normalized_predictions.items()
        }

        replay.current_event_index = next_index
        replay.visible_event_ids.append(revealed_event.event_id)
        replay.last_revealed_event = revealed_event
        self._apply_historical_event(session.world, revealed_event)

        if assessments:
            session.prediction_assessments.extend(assessments.values())

        return revealed_event, assessments

    def _apply_historical_event(self, world: WorldState, event: HistoricalEvent) -> None:
        world.tension_level = self._clamp_percent(world.tension_level + event.impact.tension_delta)
        world.market_stress = self._clamp_percent(world.market_stress + event.impact.market_stress_delta)
        world.oil_pressure = self._clamp_percent(world.oil_pressure + event.impact.oil_pressure_delta)

        for agent_id, metric_deltas in event.impact.actor_metric_deltas.items():
            for metric, delta in metric_deltas.items():
                self._bump_actor_metric(world, agent_id, metric, delta)

        affected_agents = self._historical_event_affected_agents(event)
        self._register_latent_event(
            world,
            LatentEvent(
                event_id=f"historical-{event.event_id}",
                topic=event.topic,
                status="active",
                severity=severity_score(event.severity),
                visibility="public",
                reliability=0.96 if event.confirmed else 0.72,
                origin=f"historical:{event.source_type}",
                affected_agents=affected_agents,
                started_at_turn=world.turn,
                last_updated_turn=world.turn,
                decay_rate=0.04,
                narratives=[
                    LatentEventNarrative(
                        framing="baseline",
                        summary=self._clip_summary(event.public_summary or event.summary),
                        confidence=0.9 if event.confirmed else 0.68,
                        public=True,
                    ),
                    LatentEventNarrative(
                        framing="concealed",
                        summary=self._clip_summary(event.summary),
                        confidence=0.82 if event.confirmed else 0.6,
                        public=False,
                    ),
                ],
            ),
            spawn_links=False,
        )
        self._resync_public_events(world)
        self._resync_public_actor_state(world)

    @staticmethod
    def _historical_event_affected_agents(event: HistoricalEvent) -> list[str]:
        affected = [
            agent_id
            for agent_id in [*event.actors, *event.targets]
            if agent_id in AGENT_IDS
        ]
        return sorted(set(affected))

    def _normalize_predictions(
        self,
        session: SessionState,
        predictions: dict[str, Prediction],
    ) -> dict[str, Prediction]:
        normalized: dict[str, Prediction] = {}
        for agent_id, prediction in predictions.items():
            if prediction.agent_id != agent_id:
                raise ValueError(
                    f"Prediction agent mismatch: payload agent={prediction.agent_id}, slot={agent_id}"
                )
            if prediction.time_horizon_turns < 1:
                raise ValueError("Prediction time_horizon_turns must be at least 1.")
            normalized[agent_id] = prediction.model_copy(
                update={
                    "turn": max(0, session.world.turn - 1),
                    "timestamp": (
                        session.historical_replay.last_revealed_event.timestamp
                        if session.historical_replay.enabled and session.historical_replay.last_revealed_event is not None
                        else prediction.timestamp
                    ),
                },
                deep=True,
            )
        return normalized

    def _score_prediction(self, *, prediction: Prediction, event: HistoricalEvent) -> PredictionAssessment:
        topic_score = 1.0 if prediction.topic == event.topic else -0.4
        actor_score = (
            1.0
            if prediction.predicted_actor and prediction.predicted_actor in event.actors
            else (-0.2 if prediction.predicted_actor is None else -0.5)
        )
        target_score = (
            1.0
            if prediction.predicted_target and prediction.predicted_target in event.targets
            else (-0.15 if prediction.predicted_target is None else -0.45)
        )
        timing_score = self._clamp_unit(1.0 - abs(prediction.time_horizon_turns - 1) * 0.6)
        severity_match_distance = severity_distance(prediction.expected_severity, event.severity)
        severity_alignment = {0: 1.0, 1: 0.35, 2: -0.2, 3: -0.5}.get(severity_match_distance, -0.5)

        correctness = max(
            0.0,
            min(
                1.0,
                (
                    0.38 * max(topic_score, 0.0)
                    + 0.22 * max(actor_score, 0.0)
                    + 0.16 * max(target_score, 0.0)
                    + 0.12 * max(timing_score, 0.0)
                    + 0.12 * max(severity_alignment, 0.0)
                ),
            ),
        )
        calibration = self._clamp_unit(1.0 - abs(prediction.confidence - correctness) * 2.0)

        vague_penalty = 0.0
        if prediction.predicted_actor is None and prediction.predicted_target is None:
            vague_penalty -= 0.18
        if len(prediction.summary.strip()) < 24:
            vague_penalty -= 0.12

        contradiction_penalty = 0.0
        if topic_score < 0.0 and actor_score < 0.0 and prediction.confidence >= 0.55:
            contradiction_penalty = -0.22

        confident_false_penalty = 0.0
        if correctness < 0.25 and prediction.confidence >= 0.7:
            confident_false_penalty = -0.32

        total = self._clamp_unit(
            0.28 * topic_score
            + 0.18 * actor_score
            + 0.14 * target_score
            + 0.12 * timing_score
            + 0.16 * severity_alignment
            + 0.12 * calibration
            + vague_penalty
            + contradiction_penalty
            + confident_false_penalty
        )

        return PredictionAssessment(
            prediction_id=prediction.prediction_id,
            agent_id=prediction.agent_id,
            turn=prediction.turn,
            evaluated_event_id=event.event_id,
            evaluated_event_summary=event.summary,
            topic_score=round(topic_score, 3),
            actor_score=round(actor_score, 3),
            target_score=round(target_score, 3),
            timing_score=round(timing_score, 3),
            severity_score=round(severity_alignment, 3),
            confidence_calibration=round(calibration, 3),
            vague_penalty=round(vague_penalty, 3),
            contradiction_penalty=round(contradiction_penalty, 3),
            confident_false_penalty=round(confident_false_penalty, 3),
            total=round(total, 3),
        )

    def _apply_forecast_rewards(
        self,
        rewards: dict[str, RewardBreakdown],
        assessments: dict[str, PredictionAssessment],
    ) -> None:
        for agent_id, assessment in assessments.items():
            reward = rewards.get(agent_id, RewardBreakdown())
            reward.forecast_terms = {
                "topic": assessment.topic_score,
                "actor": assessment.actor_score,
                "target": assessment.target_score,
                "timing": assessment.timing_score,
                "severity": assessment.severity_score,
                "confidence_calibration": assessment.confidence_calibration,
                "vague_penalty": assessment.vague_penalty,
                "contradiction_penalty": assessment.contradiction_penalty,
                "confident_false_penalty": assessment.confident_false_penalty,
            }
            reward.forecast_total = assessment.total
            reward.total = round(
                self._clamp_unit(reward.total + FORECAST_REWARD_BLEND * assessment.total),
                3,
            )
            rewards[agent_id] = reward

    def _apply_scenario(self, world: WorldState, scenario: ScenarioDefinition) -> None:
        for field_name, value in scenario.world_overrides.items():
            if field_name in {"tension_level", "market_stress", "oil_pressure"}:
                setattr(world, field_name, round(self._clamp_percent(value), 2))

        for agent_id, allies in scenario.coalition_overrides.items():
            world.coalition_graph[agent_id] = list(allies)

        for agent_id, intent in scenario.hidden_intent_overrides.items():
            world.hidden_intents[agent_id] = intent

        for shift in scenario.metric_shifts:
            self._bump_actor_metric(world, shift.agent_id, shift.metric, shift.delta)

        for impact in scenario.asset_impacts:
            self._apply_scenario_asset_impact(world, impact)

        for event in scenario.public_events:
            self._register_latent_event(
                world,
                self._signal_to_latent_event(
                    world,
                    ExternalSignal(
                        source=event.source,
                        headline=event.headline,
                        region=event.region,
                        tags=list(event.tags),
                        severity=event.severity,
                    ),
                ),
            )
        for index, event in enumerate(scenario.latent_events):
            self._register_latent_event(
                world,
                self._scenario_latent_event_to_event(scenario.id, index, event),
            )
        self._resync_public_events(world)
        self._resync_public_actor_state(world)

    def _apply_scenario_asset_impact(self, world: WorldState, impact: ScenarioAssetImpact) -> None:
        if impact.mode == "repair":
            self._restore_assets(
                world,
                owner=impact.owner,
                intensity=impact.intensity,
                reason=impact.reason,
                section_bias=impact.section_bias,
                max_assets=impact.max_assets,
            )
            return
        self._damage_assets(
            world,
            owner=impact.owner,
            intensity=impact.intensity,
            reason=impact.reason,
            section_bias=impact.section_bias,
            max_assets=impact.max_assets,
            max_status=impact.max_status,
        )

    def _register_latent_event(
        self,
        world: WorldState,
        event: LatentEvent,
        *,
        spawn_links: bool = True,
    ) -> LatentEvent:
        world.latent_events = [existing for existing in world.latent_events if existing.event_id != event.event_id]
        world.latent_events.append(event)
        if spawn_links:
            self._spawn_linked_latent_events(world, event)
        return event

    def _resync_public_events(self, world: WorldState) -> None:
        public_events: list[BlackSwanEvent] = []
        for event in world.latent_events[-24:]:
            if event.status == "resolved" and event.severity < 0.18:
                continue
            if event.visibility == "private":
                continue
            public_events.append(
                BlackSwanEvent(
                    id=event.event_id,
                    summary=self._latent_event_public_summary(event),
                    source=event.origin,
                    severity=round(max(0.22, min(0.95, event.severity * event.reliability + 0.12)), 3),
                    public=True,
                    affected_agents=list(event.affected_agents),
                )
            )
        world.active_events = public_events[-12:]

    def _scenario_latent_event_to_event(
        self,
        scenario_id: str,
        index: int,
        event: ScenarioLatentEvent,
    ) -> LatentEvent:
        affected_agents = list(event.affected_agents) or ["us", "israel", "iran", "hezbollah", "gulf"]
        narratives = [
            LatentEventNarrative(
                framing="baseline",
                summary=event.public_summary or event.summary,
                confidence=min(0.92, event.reliability + 0.06),
                public=True,
            )
        ]
        if event.private_summary:
            narratives.append(
                LatentEventNarrative(
                    framing="concealed",
                    summary=event.private_summary,
                    confidence=max(0.36, event.reliability),
                    public=False,
                )
            )
        return LatentEvent(
            event_id=f"scenario-latent-{scenario_id}-{index}",
            topic=event.topic,
            status="active",
            severity=event.severity,
            visibility=event.visibility,  # type: ignore[arg-type]
            reliability=event.reliability,
            origin=event.source,
            affected_agents=affected_agents,
            started_at_turn=0,
            last_updated_turn=0,
            decay_rate=event.decay_rate,
            narratives=narratives or self._default_latent_event_narratives(event.topic, event.summary),
        )

    def _spawn_linked_latent_events(self, world: WorldState, event: LatentEvent) -> None:
        if event.severity < 0.48:
            return
        for linked_topic in LATENT_EVENT_LINKS.get(event.topic, ()):
            if any(
                existing.topic == linked_topic
                and event.event_id in existing.linked_event_ids
                and existing.status != "resolved"
                for existing in world.latent_events
            ):
                continue
            linked_event = LatentEvent(
                event_id=f"{event.event_id}-{linked_topic}",
                topic=linked_topic,
                status="emerging",
                severity=round(max(0.24, min(0.82, event.severity * 0.68)), 3),
                visibility="public" if linked_topic in {"market", "humanitarian", "diplomacy"} else "mixed",
                reliability=max(0.42, round(event.reliability - 0.06, 3)),
                origin=event.origin,
                affected_agents=list(event.affected_agents),
                started_at_turn=world.turn,
                last_updated_turn=world.turn,
                decay_rate=min(0.16, event.decay_rate + 0.02),
                linked_event_ids=[event.event_id],
                narratives=self._default_latent_event_narratives(
                    linked_topic,
                    self._linked_event_summary(linked_topic, event),
                ),
            )
            self._register_latent_event(world, linked_event, spawn_links=False)

    def _signal_to_latent_event(self, world: WorldState, signal: ExternalSignal) -> LatentEvent:
        topic = self._infer_event_topics_from_text(
            f"{signal.headline} {' '.join(signal.tags)} {(signal.region or '')}"
        )[0]
        affected_agents = self._infer_affected_agents(signal)
        return LatentEvent(
            event_id=f"signal-{world.turn}-{len(world.latent_events)}",
            topic=topic,
            status="active",
            severity=round(max(0.12, min(1.0, signal.severity)), 3),
            visibility="public",
            reliability=0.72,
            origin=signal.source,
            affected_agents=affected_agents,
            started_at_turn=world.turn,
            last_updated_turn=world.turn,
            decay_rate=0.08,
            narratives=self._default_latent_event_narratives(topic, signal.headline),
        )

    def _action_to_latent_event(self, world: WorldState, agent_id: str, action: AgentAction) -> LatentEvent | None:
        if action.type == "hold":
            return None
        topic = self._infer_action_event_topic(agent_id, action)
        affected_agents = [agent_id]
        if action.target in AGENT_IDS:
            affected_agents.append(action.target)
        severity = {
            "strike": 0.72,
            "mobilize": 0.62,
            "deceive": 0.54,
            "sanction": 0.5,
            "defend": 0.44,
            "intel_query": 0.36,
            "negotiate": 0.42,
            "oversight_review": 0.4,
        }.get(action.type, 0.38)
        visibility = {
            "strike": "public",
            "sanction": "public",
            "negotiate": "public",
            "oversight_review": "public",
            "mobilize": "mixed",
            "defend": "mixed",
            "intel_query": "private",
            "deceive": "private",
        }.get(action.type, "mixed")
        summary = action.summary or f"{agent_id} executed {action.type}."
        return LatentEvent(
            event_id=f"action-{agent_id}-{world.turn}-{len(world.latent_events)}",
            topic=topic,
            status="active",
            severity=severity,
            visibility=visibility,
            reliability=0.66 if visibility == "public" else 0.58,
            origin=f"{agent_id}-action",
            affected_agents=sorted(set(affected_agents)),
            started_at_turn=world.turn,
            last_updated_turn=world.turn,
            decay_rate=0.07 if visibility == "public" else 0.1,
            narratives=self._default_latent_event_narratives(topic, summary),
        )

    def _default_latent_event_narratives(self, topic: str, summary: str) -> list[LatentEventNarrative]:
        topic_label = CONTRADICTION_TOPIC_LABELS.get(topic, topic)
        clipped = self._clip_summary(summary)
        return [
            LatentEventNarrative(framing="baseline", summary=clipped, confidence=0.72, public=True),
            LatentEventNarrative(
                framing="deteriorating",
                summary=self._clip_summary(
                    f"Private reporting points to renewed deterioration in the broader {topic_label} picture. {clipped}"
                ),
                confidence=0.64,
                public=False,
            ),
            LatentEventNarrative(
                framing="stabilizing",
                summary=self._clip_summary(
                    f"Competing reporting suggests partial stabilization around the broader {topic_label} picture. {clipped}"
                ),
                confidence=0.58,
                public=False,
            ),
            LatentEventNarrative(
                framing="concealed",
                summary=self._clip_summary(f"Privately, actors suspect additional hidden activity around {topic_label} beyond what is publicly released."),
                confidence=0.52,
                public=False,
            ),
        ]

    def _linked_event_summary(self, topic: str, event: LatentEvent) -> str:
        topic_label = CONTRADICTION_TOPIC_LABELS.get(topic, topic)
        source_label = CONTRADICTION_TOPIC_LABELS.get(event.topic, event.topic)
        return f"Spillover from {source_label} is now driving {topic_label} pressure."

    @staticmethod
    def _latent_event_public_summary(event: LatentEvent) -> str:
        for narrative in event.narratives:
            if narrative.public:
                return narrative.summary
        return event.narratives[0].summary if event.narratives else event.topic

    def _infer_event_topics_from_text(self, text: str) -> list[str]:
        lowered = text.lower()
        topics = [
            topic
            for topic, keywords in LATENT_EVENT_TOPIC_KEYWORDS.items()
            if any(keyword in lowered for keyword in keywords)
        ]
        return topics or ["diplomacy"]

    def _infer_action_event_topic(self, agent_id: str, action: AgentAction) -> str:
        if action.type in {"negotiate", "oversight_review"}:
            return "diplomacy"
        if action.type == "sanction":
            return "domestic"
        if action.type in {"deceive", "intel_query"}:
            return "cyber"
        if action.type == "mobilize" and agent_id in {"iran", "hezbollah"}:
            return "corridor"
        if action.type in {"strike", "mobilize", "defend"} and action.target in {"gulf", "iran"}:
            return "shipping"
        if action.type in {"strike", "mobilize", "defend"}:
            return "border"
        return "diplomacy"

    def _apply_latent_event_pressure(self, world: WorldState) -> None:
        for event in world.latent_events:
            if event.status == "resolved":
                continue
            pressure = event.severity * max(event.reliability, 0.35)
            if event.topic == "shipping":
                world.market_stress = self._clamp_percent(world.market_stress + pressure * 0.8)
                world.oil_pressure = self._clamp_percent(world.oil_pressure + pressure * 1.1)
            elif event.topic == "commodities":
                world.market_stress = self._clamp_percent(world.market_stress + pressure * 0.9)
                world.oil_pressure = self._clamp_percent(world.oil_pressure + pressure * 0.35)
            elif event.topic == "border":
                world.tension_level = self._clamp_percent(world.tension_level + pressure * 0.9)
            elif event.topic == "corridor":
                world.tension_level = self._clamp_percent(world.tension_level + pressure * 0.7)
                world.oil_pressure = self._clamp_percent(world.oil_pressure + pressure * 0.25)
            elif event.topic == "cyber":
                world.tension_level = self._clamp_percent(world.tension_level + pressure * 0.35)
                world.market_stress = self._clamp_percent(world.market_stress + pressure * 0.45)
            elif event.topic == "domestic":
                world.market_stress = self._clamp_percent(world.market_stress + pressure * 0.4)
            elif event.topic == "humanitarian":
                world.tension_level = self._clamp_percent(world.tension_level + pressure * 0.25)
            elif event.topic == "diplomacy":
                world.tension_level = self._clamp_percent(world.tension_level - pressure * 0.35)
                world.market_stress = self._clamp_percent(world.market_stress - pressure * 0.18)

    def _decay_latent_events(self, world: WorldState) -> None:
        for event in world.latent_events:
            if event.last_updated_turn >= world.turn:
                if event.severity >= 0.66:
                    event.status = "active"
                continue
            event.severity = round(max(0.0, event.severity - event.decay_rate), 3)
            if event.severity <= 0.12:
                event.status = "resolved"
            elif event.severity <= 0.35:
                event.status = "contained"
            else:
                event.status = "active"

    def _inject_external_signals(self, world: WorldState, signals: list[ExternalSignal]) -> None:
        for signal in signals:
            self._register_latent_event(world, self._signal_to_latent_event(world, signal))
            world.tension_level = min(100.0, world.tension_level + signal.severity * 8.0)
            world.market_stress = min(100.0, world.market_stress + signal.severity * 6.0)
            if self._signal_mentions(signal.headline.lower(), "oil", "gas", "lng") or "shipping" in signal.tags:
                world.oil_pressure = min(100.0, world.oil_pressure + signal.severity * 10.0)
            self._apply_signal_pressure(world, signal)
            self._apply_signal_asset_effects(world, signal)
        self._resync_public_events(world)
        self._resync_public_actor_state(world)

    def _infer_affected_agents(self, signal: ExternalSignal) -> list[str]:
        text = f"{signal.headline} {' '.join(signal.tags)} {(signal.region or '')}".lower()
        mapping = {
            "us": ("us", "washington", "centcom", "poll"),
            "israel": ("israel", "idf", "oref", "northern front"),
            "iran": ("iran", "tehran", "hormuz", "proxy"),
            "hezbollah": ("hezbollah", "lebanon", "border", "drone"),
            "gulf": ("gulf", "saudi", "uae", "shipping", "oil", "gold", "silver", "commodity", "lng"),
        }
        affected = [agent_id for agent_id, keywords in mapping.items() if any(keyword in text for keyword in keywords)]
        return affected or ["us", "israel", "iran", "hezbollah", "gulf"]

    def _apply_actions(self, world: WorldState, actions: dict[str, AgentAction]) -> None:
        for agent_id, action in actions.items():
            self._validate_action(agent_id, action)
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
            self._apply_asset_action_effects(world, agent_id, action)
            action_event = self._action_to_latent_event(world, agent_id, action)
            if action_event is not None:
                self._register_latent_event(world, action_event)

        world.tension_level = round(world.tension_level, 2)
        world.market_stress = round(world.market_stress, 2)
        world.oil_pressure = round(world.oil_pressure, 2)
        self._apply_latent_event_pressure(world)
        self._decay_latent_events(world)
        self._reconcile_actor_state(world, actions)
        self._resync_public_events(world)
        self._resync_public_actor_state(world)

    @staticmethod
    def _validate_action(agent_id: str, action: AgentAction) -> None:
        if action.actor != agent_id:
            raise ValueError(f"Action actor mismatch: payload actor={action.actor}, slot={agent_id}")
        if action.type not in AGENT_ALLOWED_ACTIONS.get(agent_id, ()):
            raise ValueError(f"Unsupported action for {agent_id}: {action.type}")

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
        # Oversight only observes and scores risk — it does NOT override agent actions.
        # Intervention (action overrides) should only happen via observer or execute/inject.
        return OversightIntervention(
            triggered=True,
            risk_score=risk_score,
            reason="Escalation probability exceeded the intervention threshold.",
            affected_agents=affected or ["us", "israel", "iran", "hezbollah", "gulf"],
        )

    def _resolve_oversight_actions(
        self,
        actions: dict[str, AgentAction],
        oversight: OversightIntervention,
    ) -> dict[str, AgentAction]:
        if not oversight.triggered or not oversight.action_override:
            return actions

        resolved_actions = dict(actions)
        for agent_id, override_action in oversight.action_override.items():
            self._validate_action(agent_id, override_action)
            resolved_actions[agent_id] = override_action
        return resolved_actions

    def _build_oversight_override_action(
        self,
        *,
        world: WorldState,
        agent_id: str,
        action: AgentAction,
        signals: list[ExternalSignal],
    ) -> AgentAction:
        signal_text = " ".join(
            f"{signal.headline} {' '.join(signal.tags)} {(signal.region or '')}"
            for signal in signals
        ).lower()
        asset_pressure = self._asset_pressure(world, agent_id)

        if (
            self._signal_mentions(signal_text, "ceasefire", "talk", "negotiat", "summit", "mediat", "humanitarian")
            and "negotiate" in AGENT_ALLOWED_ACTIONS.get(agent_id, ())
        ):
            return AgentAction(
                actor=agent_id,
                type="negotiate",
                target=self._select_oversight_negotiation_target(agent_id),
                summary="Oversight forced a de-escalatory negotiation cycle after elevated intervention risk.",
                metadata={"mode": "oversight_override", "replaces": action.type},
            )

        if (
            asset_pressure >= 0.2
            or self._signal_mentions(signal_text, "attack", "strike", "rocket", "missile", "drone", "shipping", "cyber", "outage")
        ) and "defend" in AGENT_ALLOWED_ACTIONS.get(agent_id, ()):
            return AgentAction(
                actor=agent_id,
                type="defend",
                target=agent_id,
                summary="Oversight forced a defensive posture to absorb incoming risk instead of escalating further.",
                metadata={"mode": "oversight_override", "replaces": action.type},
            )

        if "intel_query" in AGENT_ALLOWED_ACTIONS.get(agent_id, ()):
            return AgentAction(
                actor=agent_id,
                type="intel_query",
                summary="Oversight forced an intelligence verification cycle before any further escalation.",
                metadata={"mode": "oversight_override", "replaces": action.type},
            )

        return AgentAction(
            actor=agent_id,
            type="hold",
            summary="Oversight forced a temporary operational hold to break an escalation spiral.",
            metadata={"mode": "oversight_override", "replaces": action.type},
        )

    @staticmethod
    def _select_oversight_negotiation_target(agent_id: str) -> str | None:
        return {
            "us": "gulf",
            "israel": "us",
            "iran": "hezbollah",
            "hezbollah": "iran",
            "gulf": "us",
            "oversight": None,
        }.get(agent_id)

    def _generate_oversight_prediction(
        self,
        session: SessionState,
        resolved_actions: dict[str, AgentAction],
        tension_before: float,
    ) -> Prediction | None:
        """Generate a heuristic-based oversight prediction for the next turn."""
        import random

        world = session.world
        tension_delta = world.tension_level - tension_before
        nation_ids = [aid for aid in AGENT_IDS if aid != "oversight"]

        risk_scores = world.risk_scores
        if not risk_scores:
            return None

        # Cycle through nations — penalize recently predicted actors
        previous_predictions = getattr(session, "prediction_log", []) or []
        recent_actors = [
            p.predicted_actor
            for p in previous_predictions[-3:]
            if hasattr(p, "predicted_actor")
        ]

        scored = []
        for aid in nation_ids:
            score = risk_scores.get(aid, 0.0)
            recency_penalty = sum(0.15 for r in recent_actors if r == aid)
            scored.append((aid, max(0.0, score - recency_penalty)))
        scored.sort(key=lambda x: x[1], reverse=True)

        candidates = scored[: min(3, len(scored))]
        weights = [max(0.05, s) for _, s in candidates]
        most_risky = random.choices([c for c, _ in candidates], weights=weights, k=1)[0]
        most_risky_score = risk_scores.get(most_risky, 0.0)

        actor_recent = [
            a for a in world.last_actions
            if a.actor == most_risky and a.type in {"strike", "sanction", "mobilize", "deceive"}
        ]

        adversaries = AGENT_PRIMARY_ADVERSARIES.get(most_risky, ())
        predicted_target = adversaries[0] if adversaries else None

        current_action = resolved_actions.get(most_risky)
        current_type = current_action.type if current_action else None

        # Weight possible next actions based on context
        action_weights: dict[str, float] = {
            "strike": 0.05, "sanction": 0.1, "mobilize": 0.1,
            "defend": 0.15, "negotiate": 0.25, "intel_query": 0.15, "hold": 0.2,
        }

        if tension_delta > 5.0:
            action_weights["strike"] += 0.35
            action_weights["mobilize"] += 0.2
        elif tension_delta > 2.0:
            action_weights["sanction"] += 0.25
            action_weights["mobilize"] += 0.15
        elif tension_delta < -3.0:
            action_weights["negotiate"] += 0.3

        if most_risky_score > 0.6:
            action_weights["strike"] += 0.2
            action_weights["sanction"] += 0.15
        elif most_risky_score > 0.4:
            action_weights["sanction"] += 0.1
            action_weights["defend"] += 0.1

        if actor_recent:
            action_weights["mobilize"] += 0.1 * len(actor_recent)

        if current_type and current_type in action_weights:
            action_weights[current_type] *= 0.5

        action_list = list(action_weights.keys())
        w = [max(0.01, action_weights[a]) for a in action_list]
        predicted_action = random.choices(action_list, weights=w, k=1)[0]

        total_w = sum(w)
        chosen_w = max(0.01, action_weights[predicted_action])
        confidence = min(0.85, 0.30 + (chosen_w / total_w) * 0.5 + most_risky_score * 0.15)

        if predicted_action == "negotiate":
            coalitions = world.coalition_graph.get(most_risky, [])
            predicted_target = coalitions[0] if coalitions else predicted_target
            topic = "diplomatic_channel"
        elif predicted_action == "strike":
            topic = "military_escalation"
        elif predicted_action in {"sanction", "deceive"}:
            topic = "economic_pressure"
        elif predicted_action == "mobilize":
            topic = "force_posture"
        elif predicted_action in {"defend", "intel_query"}:
            topic = "security_posture"
        else:
            topic = "status_quo"

        rationale_parts = [
            f"Tension {'rose' if tension_delta > 0 else 'fell'} by {abs(tension_delta):.1f} to {world.tension_level:.0f}.",
            f"{most_risky} has risk score {most_risky_score:.2f}.",
        ]
        if actor_recent:
            rationale_parts.append(
                f"Recent aggressive actions by {most_risky}: {', '.join(a.type for a in actor_recent[:3])}."
            )
        if world.market_stress > 50:
            rationale_parts.append(f"Market stress elevated at {world.market_stress:.0f}.")

        severity_map = {
            "strike": "critical", "mobilize": "high", "sanction": "high",
            "deceive": "medium", "defend": "medium", "intel_query": "low",
            "negotiate": "low", "hold": "low",
        }

        summary = (
            f"Oversight predicts {most_risky} will likely {predicted_action}"
            + (f" targeting {predicted_target}" if predicted_target else "")
            + f" next turn. Confidence: {confidence:.0%}."
        )

        return Prediction(
            agent_id="oversight",
            turn=world.turn,
            topic=topic,
            predicted_actor=most_risky,
            predicted_target=predicted_target,
            time_horizon_turns=1,
            expected_severity=severity_map.get(predicted_action, "medium"),
            confidence=round(confidence, 3),
            summary=summary,
            rationale=" ".join(rationale_parts),
        )

    def _apply_oversight(self, world: WorldState, oversight: OversightIntervention) -> None:
        if not oversight.triggered:
            return
        world.tension_level = max(0.0, world.tension_level - 4.0)
        world.market_stress = max(0.0, world.market_stress - 2.0)
        self._register_latent_event(
            world,
            LatentEvent(
                event_id=f"oversight-{world.turn}-{len(world.latent_events)}",
                topic="diplomacy",
                status="active",
                severity=oversight.risk_score,
                visibility="public",
                reliability=0.78,
                origin="oversight-wrapper",
                affected_agents=list(oversight.affected_agents),
                started_at_turn=world.turn,
                last_updated_turn=world.turn,
                decay_rate=0.05,
                narratives=self._default_latent_event_narratives(
                    "diplomacy",
                    "Oversight injected a corrective diplomatic pause into the next state.",
                ),
            ),
        )
        self._resync_public_events(world)
        for agent_id in oversight.affected_agents:
            world.risk_scores[agent_id] = oversight.risk_score
            world.behavioral_consistency[agent_id] = min(
                1.0,
                world.behavioral_consistency.get(agent_id, 0.6) + 0.04,
            )

    def _initialize_belief_state(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
    ) -> dict[str, AgentBeliefState]:
        belief_state: dict[str, AgentBeliefState] = {}
        for agent_id in AGENT_IDS:
            beliefs = [
                self._belief_entry_from_event(event, world=world, episode=episode, agent_id=agent_id)
                for event in self._relevant_latent_events_for_agent(world, agent_id)
            ]
            belief_state[agent_id] = AgentBeliefState(
                agent_id=agent_id,
                dominant_topics=self._dominant_belief_topics(beliefs),
                beliefs=beliefs[:8],
                last_revision_turn=world.turn,
            )
        return belief_state

    def _update_belief_state(self, session: SessionState) -> dict[str, AgentBeliefState]:
        world = session.world
        episode = session.episode
        updated_state: dict[str, AgentBeliefState] = {}
        for agent_id in AGENT_IDS:
            existing = session.belief_state.get(agent_id, AgentBeliefState(agent_id=agent_id))
            belief_index = {belief.belief_id: belief.model_copy(deep=True) for belief in existing.beliefs}
            seen_belief_ids: set[str] = set()

            for event in self._relevant_latent_events_for_agent(world, agent_id):
                next_belief = self._belief_entry_from_event(event, world=world, episode=episode, agent_id=agent_id)
                seen_belief_ids.add(next_belief.belief_id)
                prior = belief_index.get(next_belief.belief_id)
                if prior is None:
                    belief_index[next_belief.belief_id] = next_belief
                    continue
                belief_index[next_belief.belief_id] = self._revise_belief_entry(
                    prior,
                    next_belief,
                    agent_id=agent_id,
                    turn=world.turn,
                )

            for belief_id, prior in list(belief_index.items()):
                if belief_id in seen_belief_ids:
                    continue
                decayed = self._decay_unseen_belief(prior, agent_id=agent_id, turn=world.turn)
                if decayed is None:
                    belief_index.pop(belief_id, None)
                    continue
                belief_index[belief_id] = decayed

            beliefs = sorted(
                belief_index.values(),
                key=lambda belief: (belief.confidence, belief.last_updated_turn, belief.confirmation_count),
                reverse=True,
            )[:8]
            updated_state[agent_id] = AgentBeliefState(
                agent_id=agent_id,
                dominant_topics=self._dominant_belief_topics(beliefs),
                beliefs=beliefs,
                last_revision_turn=world.turn,
            )
        return updated_state

    @staticmethod
    def _belief_doctrine_prior(agent_id: str, topic: str) -> float:
        return BELIEF_TOPIC_PRIORS.get(agent_id, {}).get(topic, 0.0)

    def _revise_belief_entry(
        self,
        prior: AgentBeliefEntry,
        next_belief: AgentBeliefEntry,
        *,
        agent_id: str,
        turn: int,
    ) -> AgentBeliefEntry:
        revised = prior.model_copy(deep=True)
        doctrine_prior = self._belief_doctrine_prior(agent_id, next_belief.topic)
        contradiction = next_belief.status in {"contested", "disconfirmed"} or (
            next_belief.status == "suspected" and prior.status in {"active", "confirmed"}
        )

        revised.source = next_belief.source
        revised.suspected_agents = list(next_belief.suspected_agents)
        revised.related_event_ids = list({*prior.related_event_ids, *next_belief.related_event_ids})
        revised.last_updated_turn = turn

        if contradiction:
            revised.contradiction_count += 1
            retention = 0.78 + doctrine_prior * 0.5 + min(prior.confirmation_count, 3) * 0.02
            revised.confidence = round(
                max(
                    BELIEF_PERSISTENCE_FLOOR,
                    min(
                        0.98,
                        prior.confidence * retention
                        + next_belief.confidence * 0.12
                        - BELIEF_CONTRADICTION_PENALTY,
                    ),
                ),
                3,
            )
            revised.status = "disconfirmed" if revised.contradiction_count >= 2 and revised.confidence <= 0.5 else "contested"
            if revised.confidence < 0.46:
                revised.summary = next_belief.summary
            return revised

        revised.confirmation_count += 1
        retention = 0.38 + doctrine_prior * 0.4
        revised.confidence = round(
            max(
                BELIEF_PERSISTENCE_FLOOR,
                min(
                    0.98,
                    prior.confidence * retention
                    + next_belief.confidence * (1.0 - retention)
                    + BELIEF_CONFIRMATION_BONUS,
                ),
            ),
            3,
        )
        revised.summary = next_belief.summary
        revised.status = next_belief.status
        if revised.confirmation_count >= 2 and revised.confidence >= 0.74 and revised.status == "active":
            revised.status = "confirmed"
        if revised.status in {"active", "confirmed"}:
            revised.last_confirmed_turn = turn
        return revised

    def _decay_unseen_belief(
        self,
        prior: AgentBeliefEntry,
        *,
        agent_id: str,
        turn: int,
    ) -> AgentBeliefEntry | None:
        stale_turns = max(0, turn - prior.last_updated_turn)
        if stale_turns <= 0:
            return prior

        doctrine_prior = self._belief_doctrine_prior(agent_id, prior.topic)
        decay = max(
            0.04,
            0.11
            - doctrine_prior * 0.22
            - min(prior.confirmation_count, 3) * 0.01,
        )
        revised = prior.model_copy(deep=True)
        revised.confidence = round(max(0.06, prior.confidence - decay), 3)

        if stale_turns >= 1 and revised.status == "confirmed":
            revised.status = "active"
        if stale_turns >= 2 and revised.status in {"active", "confirmed"}:
            revised.status = "contested" if revised.confidence >= 0.42 else "suspected"
        elif stale_turns >= 2 and revised.status == "contested" and revised.confidence < 0.34:
            revised.status = "suspected"
        if stale_turns >= 3 and revised.contradiction_count > revised.confirmation_count and revised.confidence < 0.28:
            revised.status = "disconfirmed"
        if stale_turns > BELIEF_MAX_STALE_TURNS and revised.confidence <= 0.14:
            return None
        return revised

    def _relevant_latent_events_for_agent(self, world: WorldState, agent_id: str) -> list[LatentEvent]:
        relevant_events: list[LatentEvent] = []
        for event in world.latent_events:
            if event.status == "resolved":
                continue
            if agent_id == "oversight":
                relevant_events.append(event)
                continue
            if event.visibility in {"public", "mixed"} or agent_id in event.affected_agents:
                relevant_events.append(event)
        return relevant_events

    def _belief_entry_from_event(
        self,
        event: LatentEvent,
        *,
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> AgentBeliefEntry:
        summary = self._belief_summary_for_agent(event, world=world, episode=episode, agent_id=agent_id)
        confidence = self._belief_confidence_for_agent(event, world=world, episode=episode, agent_id=agent_id)
        status = self._belief_status_for_agent(event, confidence=confidence, episode=episode, agent_id=agent_id)
        return AgentBeliefEntry(
            belief_id=f"{agent_id}:{event.event_id}",
            topic=event.topic,
            summary=summary,
            confidence=confidence,
            status=status,
            source=event.origin,
            suspected_agents=list(event.affected_agents),
            related_event_ids=[event.event_id, *event.linked_event_ids],
            confirmation_count=1 if status in {"active", "confirmed"} else 0,
            contradiction_count=1 if status == "contested" else 0,
            last_confirmed_turn=world.turn if status in {"active", "confirmed"} else None,
            last_updated_turn=world.turn,
        )

    def _belief_summary_for_agent(
        self,
        event: LatentEvent,
        *,
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> str:
        narrative = self._select_private_event_narrative(event, world=world, episode=episode, agent_id=agent_id)
        prefix = "Belief" if event.visibility == "private" else "Assessment"
        return self._clip_summary(f"{prefix}: {narrative.summary}", 180)

    def _belief_confidence_for_agent(
        self,
        event: LatentEvent,
        *,
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> float:
        confidence = event.reliability
        confidence += self._belief_doctrine_prior(agent_id, event.topic)
        if event.visibility == "public":
            confidence += 0.12
        elif event.visibility == "private":
            confidence -= 0.08
        if self._observation_projection_enabled(agent_id, episode):
            confidence += (self._projection_unit(world, episode, agent_id, f"belief:{event.event_id}") - 0.5) * 0.22
        if agent_id != "oversight" and agent_id not in event.affected_agents and event.visibility != "public":
            confidence -= 0.14
        return round(max(0.08, min(0.96, confidence)), 3)

    @staticmethod
    def _belief_status_for_agent(
        event: LatentEvent,
        *,
        confidence: float,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> str:
        doctrine_prior = BELIEF_TOPIC_PRIORS.get(agent_id, {}).get(event.topic, 0.0)
        if event.visibility == "private" and agent_id not in event.affected_agents and agent_id != "oversight":
            return "suspected"
        if event.status == "contained" and confidence < 0.45:
            return "contested"
        if not episode.fog_of_war or agent_id == "oversight":
            return "confirmed"
        if confidence >= 0.7 - min(0.08, doctrine_prior * 0.45):
            return "active"
        if confidence <= 0.34 - min(0.06, doctrine_prior * 0.2):
            return "suspected"
        return "contested"

    @staticmethod
    def _dominant_belief_topics(beliefs: list[AgentBeliefEntry]) -> list[str]:
        ranked: dict[str, float] = {}
        for belief in beliefs:
            ranked[belief.topic] = ranked.get(belief.topic, 0.0) + belief.confidence
        return [
            topic
            for topic, _ in sorted(ranked.items(), key=lambda item: item[1], reverse=True)[:3]
        ]

    def _build_observations(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        *,
        include_live_sources: bool = False,
        belief_state: dict[str, AgentBeliefState] | None = None,
        historical_replay: HistoricalReplayState | None = None,
    ) -> dict[str, AgentObservation]:
        observations: dict[str, AgentObservation] = {}
        public_events = [event for event in world.active_events if event.public][-MAX_PUBLIC_BRIEF_ITEMS:]
        public_brief = self._build_public_brief_from_latent_events(world)

        for agent_id in AGENT_IDS:
            profile = AGENT_PROFILES[agent_id]
            entity_pack = load_entity_pack(agent_id)
            entity_profile = entity_pack.get("profile", {})
            strategic_assets = self._flatten_strategic_assets(
                agent_id=agent_id,
                entity_pack=entity_pack,
                world=world,
            )
            training_source_bundle = AGENT_TRAINING_SOURCE_BUNDLES.get(agent_id, [])
            live_source_bundle = AGENT_LIVE_SOURCE_BUNDLES.get(agent_id, [])
            available_data_sources = self._build_data_source_context(agent_id)
            projection_enabled = self._observation_projection_enabled(agent_id, episode)
            beliefs = (belief_state or {}).get(agent_id, AgentBeliefState(agent_id=agent_id))
            training_source_packets, live_source_packets = self._source_harvester.get_packets_for_agent(
                agent_id,
                include_live=include_live_sources,
            )
            baseline_private_brief = [
                IntelSnippet(
                    source="scenario",
                    category="private_intel",
                    summary=self._clip_summary(summary),
                    confidence=0.72,
                )
                for summary in profile.baseline_private_intel
            ]
            focused_private_brief = self._build_focused_private_brief(
                world=world,
                episode=episode,
                agent_id=agent_id,
                focus_terms=profile.intelligence_focus,
            )
            projected_state, obscured_metric_count = self._project_strategic_state(
                world=world,
                episode=episode,
                agent_id=agent_id,
            )
            projected_assets = self._project_strategic_assets(
                strategic_assets=strategic_assets,
                world=world,
                episode=episode,
                agent_id=agent_id,
            )
            training_source_brief, training_projection = self._source_packets_to_briefs(
                training_source_packets,
                category="training_source",
                world=world,
                episode=episode,
                agent_id=agent_id,
            )
            live_source_brief, live_projection = (
                self._source_packets_to_briefs(
                    live_source_packets,
                    category="live_source",
                    world=world,
                    episode=episode,
                    agent_id=agent_id,
                )
                if include_live_sources
                else ([], {"delayed": 0, "contested": 0, "contradiction_packets": 0, "confidence_sum": 0.0, "contradiction_topics": []})
            )
            private_brief = self._compose_private_brief(
                baseline_private_brief=baseline_private_brief,
                focused_private_brief=focused_private_brief,
                training_source_brief=training_source_brief,
                live_source_brief=live_source_brief,
            )
            projection = self._build_observation_projection(
                agent_id=agent_id,
                projection_enabled=projection_enabled,
                obscured_metric_count=obscured_metric_count,
                delivered_brief_count=len(training_source_brief) + len(live_source_brief),
                delayed_source_count=int(training_projection["delayed"]) + int(live_projection["delayed"]),
                contested_source_count=int(training_projection["contested"]) + int(live_projection["contested"]),
                contradiction_packet_count=int(training_projection["contradiction_packets"])
                + int(live_projection["contradiction_packets"]),
                contradiction_topics=sorted(
                    {
                        *training_projection["contradiction_topics"],
                        *live_projection["contradiction_topics"],
                    }
                ),
                confidence_sum=float(training_projection["confidence_sum"]) + float(live_projection["confidence_sum"]),
            )
            asset_alerts = self._build_asset_alerts(projected_assets)
            historical_brief = self._build_historical_brief(historical_replay)
            decision_prompt = self._build_decision_prompt(
                agent_id=agent_id,
                entity_profile=entity_profile,
                strategic_assets=projected_assets,
                available_data_sources=available_data_sources,
                live_enabled=include_live_sources,
                projection=projection,
                belief_state=beliefs,
                historical_replay=historical_replay,
            )

            observations[agent_id] = AgentObservation(
                public_brief=public_brief,
                private_brief=private_brief,
                belief_brief=[belief.summary for belief in beliefs.beliefs[:4]],
                belief_topics=list(beliefs.dominant_topics),
                perceived_tension=self._perceived_tension(world.tension_level, agent_id, episode.fog_of_war),
                known_coalitions=sorted(world.coalition_graph.get(agent_id, [])),
                event_log=public_events,
                decision_prompt=decision_prompt,
                available_actions=list(AGENT_ALLOWED_ACTIONS.get(agent_id, ())),
                available_data_sources=available_data_sources,
                entity_profile=entity_profile,
                strategic_state=projected_state,
                strategic_assets=projected_assets,
                asset_alerts=asset_alerts,
                source_bundle=training_source_bundle,
                training_source_bundle=training_source_bundle,
                live_source_bundle=live_source_bundle,
                source_packets=training_source_packets + live_source_packets,
                training_source_packets=training_source_packets,
                live_source_packets=live_source_packets,
                historical_brief=historical_brief,
                projection=projection,
            )
        return observations

    def _build_public_brief_from_latent_events(self, world: WorldState) -> list[IntelSnippet]:
        public_events = [
            event
            for event in world.latent_events
            if event.visibility in {"public", "mixed"} and event.status != "resolved"
        ][-MAX_PUBLIC_BRIEF_ITEMS:]
        briefs: list[IntelSnippet] = []
        for event in public_events:
            briefs.append(
                IntelSnippet(
                    source=event.origin,
                    category=f"latent_{event.topic}",
                    summary=self._clip_summary(self._latent_event_public_summary(event)),
                    confidence=round(max(0.3, min(0.92, event.severity * event.reliability + 0.18)), 3),
                )
            )
        return briefs

    def _build_focused_private_brief(
        self,
        *,
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
        focus_terms: tuple[str, ...],
    ) -> list[IntelSnippet]:
        briefs: list[IntelSnippet] = []
        for event in reversed(world.latent_events[-8:]):
            if event.status == "resolved":
                continue
            if agent_id != "oversight" and agent_id not in event.affected_agents:
                event_text = " ".join(narrative.summary for narrative in event.narratives).lower()
                if not any(term in event_text for term in focus_terms):
                    continue
            narrative = self._select_private_event_narrative(event, world=world, episode=episode, agent_id=agent_id)
            briefs.append(
                IntelSnippet(
                    source=event.origin,
                    category=f"latent_{event.topic}",
                    summary=self._clip_summary(narrative.summary),
                    confidence=round(max(0.28, min(0.9, narrative.confidence * event.reliability)), 3),
                )
            )
        return briefs[:3]

    def _source_packets_to_briefs(
        self,
        source_packets: list[SourcePacket],
        category: str,
        *,
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> tuple[list[IntelSnippet], dict[str, float]]:
        briefs: list[IntelSnippet] = []
        delayed_count = 0
        contested_count = 0
        contradiction_packet_count = 0
        contradiction_topics: set[str] = set()
        confidence_sum = 0.0
        sorted_packets = sorted(
            (
                packet
                for packet in source_packets
                if packet.status == "ok"
                and packet.summary
                and FogOfWarDiplomacyEnv._is_source_packet_fresh(packet)
            ),
            key=lambda packet: packet.fetched_at or datetime.fromtimestamp(0, tz=timezone.utc),
            reverse=True,
        )
        for packet in sorted_packets:
            if self._should_delay_source_packet(packet, world=world, episode=episode, agent_id=agent_id):
                delayed_count += 1
                continue
            confidence = self._projected_source_confidence(packet, world=world, episode=episode, agent_id=agent_id)
            relevant_events = self._relevant_latent_events_for_packet(packet, world=world)
            event_context = (
                max(relevant_events, key=lambda event: event.severity * event.reliability)
                if relevant_events
                else None
            )
            contradiction = self._latent_source_contradiction(
                packet,
                world=world,
                episode=episode,
                agent_id=agent_id,
            )
            contested = confidence < 0.52 or (
                packet.kind in LOW_FIDELITY_SOURCE_KINDS
                and self._projection_unit(world, episode, agent_id, f"contested:{packet.source_id}") >= 0.72
            )
            if contradiction["enabled"]:
                contradiction_packet_count += 1
                contradiction_topics.add(str(contradiction["topic"]))
                confidence = max(0.24, round(confidence - float(contradiction["confidence_penalty"]), 3))
            if contested:
                contested_count += 1
            briefs.append(
                IntelSnippet(
                    source=packet.source_name,
                    category=category,
                    summary=self._project_source_summary(
                        packet.summary,
                        confidence=confidence,
                        contested=contested,
                        event_context=event_context,
                        contradiction=contradiction,
                    ),
                    confidence=confidence,
                )
            )
            confidence_sum += confidence
        return briefs, {
            "delayed": float(delayed_count),
            "contested": float(contested_count),
            "contradiction_packets": float(contradiction_packet_count),
            "confidence_sum": round(confidence_sum, 3),
            "contradiction_topics": sorted(contradiction_topics),
        }

    def _build_historical_brief(self, historical_replay: HistoricalReplayState | None) -> list[str]:
        if historical_replay is None or not historical_replay.enabled:
            return []

        visible_events = historical_replay.ground_truth_timeline[: historical_replay.current_event_index + 1]
        lines = [
            (
                f"{event.timestamp.date().isoformat()} {event.topic}: "
                f"{self._clip_summary(event.public_summary or event.summary, 110)}"
            )
            for event in visible_events[-3:]
        ]
        if visible_events:
            lines.append(
                "Visible replay history ends at "
                f"{visible_events[-1].timestamp.isoformat()}. Predict the most likely next event over the next turn."
            )
        return lines

    @staticmethod
    def _clip_summary(summary: str, limit: int = MAX_INTEL_SUMMARY_CHARS) -> str:
        collapsed = " ".join(summary.split())
        if len(collapsed) <= limit:
            return collapsed
        return f"{collapsed[: limit - 3].rstrip()}..."

    @staticmethod
    def _is_source_packet_fresh(packet) -> bool:
        if packet.fetched_at is None:
            return False
        try:
            source = get_source_by_id(packet.source_id)
        except KeyError:
            return False
        return datetime.now(timezone.utc) - packet.fetched_at <= timedelta(seconds=source_ttl_seconds(source))

    @staticmethod
    def _compose_private_brief(
        *,
        baseline_private_brief: list[IntelSnippet],
        focused_private_brief: list[IntelSnippet],
        training_source_brief: list[IntelSnippet],
        live_source_brief: list[IntelSnippet],
        limit: int = MAX_PRIVATE_BRIEF_ITEMS,
    ) -> list[IntelSnippet]:
        # Reserve space for live/training source intel so fresh news always reaches the model.
        primary_groups = [
            training_source_brief[:MAX_TRAINING_SOURCE_BRIEFS],
            live_source_brief[:MAX_LIVE_SOURCE_BRIEFS],
            focused_private_brief[:1],
            baseline_private_brief[:1],
        ]
        overflow_groups = [
            training_source_brief[MAX_TRAINING_SOURCE_BRIEFS:],
            live_source_brief[MAX_LIVE_SOURCE_BRIEFS:],
            focused_private_brief[1:],
            baseline_private_brief[1:],
        ]

        private_brief: list[IntelSnippet] = []
        for group in primary_groups + overflow_groups:
            for brief in group:
                if len(private_brief) >= limit:
                    return private_brief
                private_brief.append(brief)
        return private_brief

    def _select_private_event_narrative(
        self,
        event: LatentEvent,
        *,
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> LatentEventNarrative:
        if not event.narratives:
            return LatentEventNarrative(
                framing="baseline",
                summary=event.topic,
                confidence=max(0.3, event.reliability),
                public=event.visibility != "private",
            )
        if agent_id == "oversight" and event.narratives:
            concealed = next((n for n in event.narratives if n.framing == "concealed"), None)
            return concealed or event.narratives[-1]

        if not self._observation_projection_enabled(agent_id, episode):
            return event.narratives[0]

        narrative_pool = [
            narrative
            for narrative in event.narratives
            if not narrative.public or event.visibility == "private"
        ] or event.narratives
        index = int(self._projection_unit(world, episode, agent_id, f"latent-narrative:{event.event_id}") * len(narrative_pool))
        return narrative_pool[min(index, len(narrative_pool) - 1)]

    def _perceived_tension(self, tension_level: float, agent_id: str, fog_of_war: bool) -> float:
        if agent_id == "oversight" or not fog_of_war:
            return tension_level
        jitter = self._rng.uniform(-4.0, 4.0)
        return round(max(0.0, min(100.0, tension_level + jitter)), 2)

    @staticmethod
    def _observation_projection_enabled(agent_id: str, episode: EpisodeMetadata) -> bool:
        return episode.fog_of_war and agent_id != "oversight"

    def _project_strategic_state(
        self,
        *,
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> tuple[dict[str, float], int]:
        canonical_state = world.latent_state.get(agent_id, {}).copy()
        if not self._observation_projection_enabled(agent_id, episode):
            return canonical_state, 0

        projected_state: dict[str, float] = {}
        obscured_metric_count = 0
        uncertainty_scale = 1.2 + world.risk_scores.get(agent_id, 0.25) * 3.5
        for metric, value in canonical_state.items():
            unit = self._projection_unit(world, episode, agent_id, f"metric:{metric}")
            jitter = (unit - 0.5) * 2.0 * uncertainty_scale
            if any(token in metric for token in ("support", "confidence", "clarity", "resilience")):
                jitter *= 1.2
            observed_value = round(self._clamp_percent(value + jitter), 2)
            if abs(observed_value - value) >= 0.75:
                obscured_metric_count += 1
            projected_state[metric] = observed_value
        return projected_state, obscured_metric_count

    def _project_strategic_assets(
        self,
        *,
        strategic_assets: list[dict[str, object]],
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> list[dict[str, object]]:
        projected_assets = [asset.copy() for asset in strategic_assets]
        if not self._observation_projection_enabled(agent_id, episode):
            return projected_assets

        for asset in projected_assets:
            health = float(asset.get("health", 100.0))
            status = str(asset.get("status", "operational"))
            if status == "operational":
                asset["health"] = round(health / 5.0) * 5.0
                load = float(asset.get("operational_load", 0.0))
                asset["operational_load"] = round(load / 5.0) * 5.0
            else:
                asset["health"] = round(health, 1)
        return projected_assets

    def _build_observation_projection(
        self,
        *,
        agent_id: str,
        projection_enabled: bool,
        obscured_metric_count: int,
        delivered_brief_count: int,
        delayed_source_count: int,
        contested_source_count: int,
        contradiction_packet_count: int,
        contradiction_topics: list[str],
        confidence_sum: float,
    ) -> ObservationProjection:
        if not projection_enabled:
            return ObservationProjection(
                enabled=False,
                mode="direct",
                worldview_reliability=1.0,
            )

        mean_confidence = confidence_sum / max(delivered_brief_count, 1)
        worldview_reliability = max(
            0.32,
            min(0.9, mean_confidence - min(obscured_metric_count, 6) * 0.015),
        )
        notes = [
            "Strategic metrics are estimates under fog-of-war, not privileged ground truth.",
        ]
        if delayed_source_count > 0:
            notes.append("Some fast-moving source packets are lagged before they reach you.")
        if contested_source_count > 0:
            notes.append("At least part of the source picture is contested or fragmentary; cross-check before escalating.")
        if contradiction_packet_count > 0:
            notes.append("Multiple sources may disagree on the same latent development; compare topic framing before acting.")
        return ObservationProjection(
            enabled=True,
            mode="partial",
            worldview_reliability=round(worldview_reliability, 3),
            delayed_source_count=delayed_source_count,
            contested_source_count=contested_source_count,
            contradiction_packet_count=contradiction_packet_count,
            obscured_metric_count=obscured_metric_count,
            contradiction_topics=contradiction_topics,
            notes=notes,
        )

    def _should_delay_source_packet(
        self,
        packet: SourcePacket,
        *,
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> bool:
        if not self._observation_projection_enabled(agent_id, episode):
            return False
        if packet.delivery == "live_demo":
            return False
        turn_lag = 0
        if packet.kind in LOW_FIDELITY_SOURCE_KINDS:
            turn_lag = 1
        if world.turn >= turn_lag:
            return False
        return self._projection_unit(world, episode, agent_id, f"delay:{packet.source_id}") < 0.35

    def _projected_source_confidence(
        self,
        packet: SourcePacket,
        *,
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> float:
        if not self._observation_projection_enabled(agent_id, episode):
            return 0.65 if packet.delivery == "training_core" else 0.55
        base_reliability = SOURCE_KIND_BASE_RELIABILITY.get(packet.kind, 0.58) + SOURCE_DELIVERY_RELIABILITY.get(
            packet.delivery,
            0.0,
        )
        jitter = (self._projection_unit(world, episode, agent_id, f"confidence:{packet.source_id}") - 0.5) * 0.18
        confidence = base_reliability + jitter
        return round(max(0.24, min(0.92, confidence)), 3)

    def _project_source_summary(
        self,
        summary: str,
        *,
        confidence: float,
        contested: bool,
        event_context: LatentEvent | None = None,
        contradiction: dict[str, object] | None = None,
    ) -> str:
        clipped = self._clip_summary(summary)
        if event_context is not None and event_context.status != "resolved":
            event_summary = self._latent_event_public_summary(event_context)
            clipped = self._clip_summary(
                f"This reporting fits a broader {CONTRADICTION_TOPIC_LABELS.get(event_context.topic, event_context.topic)} picture. "
                f"{event_summary} {clipped}"
            )
        if contradiction and contradiction.get("enabled"):
            topic = str(contradiction["topic"])
            framing = str(contradiction["framing"])
            narrative_summary = contradiction.get("narrative_summary")
            if framing == "stabilizing":
                clipped = self._clip_summary(
                    f"Conflicting {topic} reporting: {narrative_summary or 'this source emphasizes partial stabilization around the same development.'} {clipped}"
                )
            else:
                clipped = self._clip_summary(
                    f"Conflicting {topic} reporting: {narrative_summary or 'this source emphasizes renewed deterioration around the same development.'} {clipped}"
                )
        if contested:
            return self._clip_summary(f"Contested reporting: {clipped}")
        if confidence < 0.48:
            return self._clip_summary(f"Unconfirmed reporting: {clipped}")
        if confidence < 0.62:
            return self._clip_summary(f"Preliminary reporting: {clipped}")
        return clipped

    def _latent_source_contradiction(
        self,
        packet: SourcePacket,
        *,
        world: WorldState,
        episode: EpisodeMetadata,
        agent_id: str,
    ) -> dict[str, object]:
        if not self._observation_projection_enabled(agent_id, episode):
            return {"enabled": False}

        relevant_events = self._relevant_latent_events_for_packet(packet, world=world)
        if not relevant_events:
            return {"enabled": False}
        event = max(relevant_events, key=lambda candidate: candidate.severity * candidate.reliability)
        contradiction_strength = min(1.0, event.severity * max(event.reliability, 0.4))
        if contradiction_strength < 0.22:
            return {"enabled": False}

        source_hint = f"{packet.source_id} {packet.source_name}".lower()
        if "rate" in source_hint or "index" in source_hint:
            framing = "stabilizing"
        elif "status" in source_hint or "watch" in source_hint or "disruption" in source_hint:
            framing = "deteriorating"
        else:
            orientation = self._projection_unit(world, episode, agent_id, f"contradiction:{packet.source_id}:{event.event_id}")
            framing = "stabilizing" if orientation < 0.5 else "deteriorating"
        narrative = next(
            (candidate for candidate in event.narratives if candidate.framing == framing),
            None,
        )
        return {
            "enabled": True,
            "topic": CONTRADICTION_TOPIC_LABELS.get(event.topic, event.topic),
            "framing": framing,
            "narrative_summary": narrative.summary if narrative is not None else None,
            "confidence_penalty": round(min(0.18, contradiction_strength * 0.22), 3),
        }

    def _relevant_latent_events_for_packet(
        self,
        packet: SourcePacket,
        *,
        world: WorldState,
    ) -> list[LatentEvent]:
        try:
            source = get_source_by_id(packet.source_id)
            source_text = " ".join(source.tags)
        except KeyError:
            source_text = ""
        text = f"{packet.summary} {' '.join(packet.sample_items)} {packet.source_name} {source_text}".lower()
        topics = set(self._infer_event_topics_from_text(text))
        relevant_events: list[LatentEvent] = []
        for event in world.latent_events:
            if event.status == "resolved":
                continue
            if event.topic in topics:
                relevant_events.append(event)
                continue
            narrative_text = " ".join(narrative.summary for narrative in event.narratives).lower()
            salient_tokens = [token for token in text.split()[:10] if len(token) >= 5]
            if any(token in narrative_text for token in salient_tokens):
                relevant_events.append(event)
        return relevant_events

    @staticmethod
    def _projection_unit(world: WorldState, episode: EpisodeMetadata, agent_id: str, label: str) -> float:
        digest = hashlib.sha256(
            f"{episode.scenario_id}|{world.turn}|{agent_id}|{label}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], byteorder="big") / float(2**64)

    @staticmethod
    def _build_model_bindings() -> dict[str, EntityModelBinding]:
        return build_entity_model_bindings()

    @staticmethod
    def _build_action_log_entries(
        session: SessionState,
        actions: dict[str, AgentAction],
    ) -> list[ActionLogEntry]:
        entries: list[ActionLogEntry] = []
        for agent_id, action in actions.items():
            entries.append(
                ActionLogEntry(
                    turn=session.world.turn,
                    actor=agent_id,
                    action_type=action.type,
                    target=action.target,
                    summary=action.summary,
                    reward_total=session.rewards.get(agent_id, RewardBreakdown()).total,
                    tension_after=session.world.tension_level,
                    market_stress_after=session.world.market_stress,
                    oil_pressure_after=session.world.oil_pressure,
                    metadata=action.metadata.copy(),
                )
            )
        return entries

    @staticmethod
    def _build_reaction_log_entry(
        *,
        session: SessionState,
        signals: list[ExternalSignal],
        latent_event_ids: list[str],
        actions: dict[str, AgentAction],
        oversight: OversightIntervention,
        tension_before: float,
    ) -> ReactionLogEntry:
        actor_outcomes: list[ReactionActorOutcome] = []
        for agent_id, action in actions.items():
            decision_mode = action.metadata.get("mode")
            if decision_mode not in {"heuristic_fallback", "provider_inference"}:
                binding = session.model_bindings.get(agent_id)
                decision_mode = binding.decision_mode if binding is not None else "heuristic_fallback"
            actor_outcomes.append(
                ReactionActorOutcome(
                    agent_id=agent_id,
                    action=action,
                    reward_total=session.rewards.get(agent_id, RewardBreakdown()).total,
                    decision_mode=decision_mode,
                )
            )

        return ReactionLogEntry(
            event_id=str(uuid4()),
            turn=session.world.turn,
            source=signals[0].source if len({signal.source for signal in signals}) == 1 else "public_release",
            latent_event_ids=latent_event_ids,
            signals=[signal.model_copy(deep=True) for signal in signals],
            actor_outcomes=actor_outcomes,
            oversight_triggered=oversight.triggered,
            tension_before=tension_before,
            tension_after=session.world.tension_level,
            market_stress_after=session.world.market_stress,
            oil_pressure_after=session.world.oil_pressure,
        )

    def _actor_metric(self, world: WorldState, agent_id: str, metric: str, default: float = 50.0) -> float:
        return world.latent_state.get(agent_id, {}).get(
            metric,
            AGENT_STATE_BASELINES.get(agent_id, {}).get(metric, default),
        )

    def _bump_actor_metric(self, world: WorldState, agent_id: str, metric: str, delta: float) -> None:
        baseline = AGENT_STATE_BASELINES.get(agent_id, {}).get(metric, 50.0)
        agent_state = world.latent_state.setdefault(agent_id, {})
        current = agent_state.get(metric, baseline)
        agent_state[metric] = round(self._clamp_percent(current + delta), 2)

    def _resync_public_actor_state(self, world: WorldState) -> None:
        public_state: dict[str, dict[str, float]] = {}
        for agent_id in AGENT_IDS:
            latent_metrics = world.latent_state.get(agent_id, {})
            synced_metrics: dict[str, float] = {}
            for metric, latent_value in latent_metrics.items():
                baseline = AGENT_STATE_BASELINES.get(agent_id, {}).get(metric, 50.0)
                previous_public = world.actor_state.get(agent_id, {}).get(metric, baseline)
                sync_factor = self._public_sync_factor(metric)
                target_public = baseline + (latent_value - baseline) * sync_factor
                lagged_public = previous_public + (target_public - previous_public) * 0.7
                synced_metrics[metric] = round(self._clamp_percent(lagged_public), 2)
            public_state[agent_id] = synced_metrics
        world.actor_state = public_state

    @staticmethod
    def _public_sync_factor(metric: str) -> float:
        lowered = metric.lower()
        for token, factor in PUBLIC_STATE_SYNC_FACTORS.items():
            if token != "default" and token in lowered:
                return factor
        return PUBLIC_STATE_SYNC_FACTORS["default"]

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

        if self._signal_mentions(
            text,
            "gold",
            "silver",
            "copper",
            "lithium",
            "nickel",
            "uranium",
            "phosphate",
            "bauxite",
            "rare earth",
            "rare-earth",
            "commodity",
            "mineral",
            "metals",
            "natural gas",
            "lng",
        ):
            self._bump_actor_metric(world, "us", "domestic_support", -1.4 * severity)
            self._bump_actor_metric(world, "gulf", "investor_confidence", -4.8 * severity)
            self._bump_actor_metric(world, "gulf", "diplomatic_flexibility", -1.8 * severity)
            self._bump_actor_metric(world, "iran", "hormuz_leverage", 1.2 * severity)
            self._bump_actor_metric(world, "oversight", "runaway_risk", 1.6 * severity)

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
        world.latent_state.setdefault("oversight", {})["runaway_risk"] = round(runaway_risk, 2)
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

    def _initial_asset_state(self) -> dict[str, dict[str, AssetCondition]]:
        asset_state: dict[str, dict[str, AssetCondition]] = {}
        for agent_id in AGENT_IDS:
            entity_pack = load_entity_pack(agent_id)
            asset_state[agent_id] = {
                asset["asset_id"]: AssetCondition(
                    asset_id=asset["asset_id"],
                    owner=agent_id,
                    name=asset["name"],
                    category=asset["category"],
                    section=asset["section"],
                    latitude=asset.get("latitude"),
                    longitude=asset.get("longitude"),
                    criticality=str(asset.get("status", "tracked")),
                    notes=asset.get("notes"),
                )
                for asset in self._asset_inventory(agent_id, entity_pack)
            }
        return asset_state

    def _build_data_source_context(self, agent_id: str) -> list[DataSourceContext]:
        return [
            DataSourceContext(
                source_id=source.id,
                name=source.name,
                delivery=source.delivery,
                kind=source.kind,
                rationale=source.rationale,
                tags=list(source.tags),
                access_notes=source.notes,
            )
            for source in get_sources_for_agent(agent_id)
        ]

    def _build_decision_prompt(
        self,
        *,
        agent_id: str,
        entity_profile: dict[str, object],
        strategic_assets: list[dict[str, object]],
        available_data_sources: list[DataSourceContext],
        live_enabled: bool,
        projection: ObservationProjection,
        belief_state: AgentBeliefState,
        historical_replay: HistoricalReplayState | None,
    ) -> str:
        profile = AGENT_PROFILES[agent_id]
        source_limit, asset_limit = ASSET_DECISION_SOURCE_LIMITS[profile.model_size]
        objectives = entity_profile.get("strategic_objectives", [])
        protected_interests = entity_profile.get("protected_interests", [])
        priority_fronts = entity_profile.get("priority_fronts", [])
        top_objectives = [str(item) for item in objectives[:3]] if isinstance(objectives, list) else []
        top_interests = [str(item) for item in protected_interests[:3]] if isinstance(protected_interests, list) else []

        source_lines = [
            f"- {source.name} [{source.delivery}/{source.kind}]: {self._clip_summary(source.rationale, 96)}"
            for source in available_data_sources[:source_limit]
        ]
        asset_lines = [
            f"- {asset['name']} [{asset['category']}] @ {asset.get('latitude', '?')}, {asset.get('longitude', '?')} status={asset.get('status')} health={asset.get('health')}"
            for asset in strategic_assets[:asset_limit]
        ]
        damaged_assets = [asset for asset in strategic_assets if str(asset.get("status")) != "operational"]
        damaged_lines = [
            f"- {asset['name']} is {asset.get('status')} ({asset.get('last_change_reason', 'needs attention')})"
            for asset in damaged_assets[:3]
        ]

        front_summary = []
        if isinstance(priority_fronts, list):
            for front in priority_fronts[:2]:
                if isinstance(front, dict) and isinstance(front.get("name"), str):
                    front_summary.append(str(front["name"]))

        prompt_sections = [
            f"You are {profile.display_name}. Role: {profile.role}. Model size: {profile.model_size}.",
            "Choose exactly one allowed action each turn and ground it in current source packets, private/public briefs, strategic state, and asset condition.",
            f"Live mode is {'enabled' if live_enabled else 'disabled'}; prefer the freshest source packets when live mode is on.",
        ]
        if projection.enabled:
            prompt_sections.append(
                f"Observation reliability is partial (estimated reliability {projection.worldview_reliability:.2f}); treat fast-moving or contested reporting cautiously."
            )
            if projection.notes:
                prompt_sections.append("Projection notes:\n" + "\n".join(f"- {note}" for note in projection.notes[:3]))
            if projection.contradiction_topics:
                prompt_sections.append(
                    "Current contradiction topics: " + ", ".join(projection.contradiction_topics[:3]) + "."
                )
        if top_objectives:
            prompt_sections.append(f"Priority objectives: {', '.join(top_objectives)}.")
        if top_interests:
            prompt_sections.append(f"Protected interests: {', '.join(top_interests)}.")
        if front_summary:
            prompt_sections.append(f"Priority fronts: {', '.join(front_summary)}.")
        if belief_state.dominant_topics:
            prompt_sections.append("Dominant remembered belief topics: " + ", ".join(belief_state.dominant_topics[:3]) + ".")
        if belief_state.beliefs:
            prompt_sections.append(
                "Belief memory:\n" + "\n".join(f"- {belief.summary}" for belief in belief_state.beliefs[:3])
            )
        if historical_replay is not None and historical_replay.enabled:
            prompt_sections.append(
                f"Historical replay mode is active for {historical_replay.replay_name}. "
                "You only know the visible timeline shown in the historical brief. "
                "Choose one action and forecast the next event without using future information."
            )
        prompt_sections.append(f"Allowed actions: {', '.join(AGENT_ALLOWED_ACTIONS.get(agent_id, ()))}.")
        prompt_sections.append("Data sources available to you:\n" + ("\n".join(source_lines) if source_lines else "- None configured."))
        prompt_sections.append("Mapped assets under your control:\n" + ("\n".join(asset_lines) if asset_lines else "- No mapped assets available."))
        if damaged_lines:
            prompt_sections.append("Assets currently degraded or malfunctioning:\n" + "\n".join(damaged_lines))
        prompt_sections.append("Use defense or repair-minded choices when critical assets are damaged; use strike, sanction, or deception only when the reward tradeoff is justified by your doctrine and the observed threat.")
        return "\n".join(prompt_sections)

    @staticmethod
    def _build_asset_alerts(strategic_assets: list[dict[str, object]]) -> list[str]:
        alerts = [
            f"{asset['name']} is {asset.get('status')} ({asset.get('last_change_reason', 'operational concern')})"
            for asset in strategic_assets
            if str(asset.get("status")) != "operational"
        ]
        return alerts[:6]

    def _flatten_strategic_assets(
        self,
        *,
        agent_id: str,
        entity_pack: dict[str, object],
        world: WorldState,
    ) -> list[dict[str, object]]:
        inventory = self._asset_inventory(agent_id, entity_pack)
        conditions = world.asset_state.get(agent_id, {})
        flattened: list[dict[str, object]] = []
        for asset in inventory:
            condition = conditions.get(asset["asset_id"])
            flattened.append(
                {
                    **asset,
                    "status": condition.status if condition is not None else asset.get("status", "operational"),
                    "health": round(condition.health, 1) if condition is not None else 100.0,
                    "operational_load": round(condition.operational_load, 1) if condition is not None else 0.0,
                    "last_change_reason": condition.last_change_reason if condition is not None else None,
                }
            )
        return flattened

    def _asset_inventory(self, agent_id: str, entity_pack: dict[str, object]) -> list[dict[str, object]]:
        assets = entity_pack.get("assets", {})
        if not isinstance(assets, dict):
            return []

        inventory: list[dict[str, object]] = []
        location_lookup = self._asset_location_lookup(assets)

        def append_asset(item: dict[str, object], section_name: str) -> None:
            name = item.get("name")
            if not isinstance(name, str) and section_name == "alliance_anchors" and isinstance(item.get("partner"), str):
                name = f"{item['partner']} alliance anchor"
            if not isinstance(name, str):
                name = item.get("location") or item.get("partner")
            if not isinstance(name, str):
                return

            entry: dict[str, object] = {
                "asset_id": self._asset_id(agent_id, section_name, name),
                "name": name,
                "category": item.get("category") or item.get("type") or ("alliance-anchor" if section_name == "alliance_anchors" else section_name),
                "section": section_name,
                "status": item.get("priority") or item.get("importance") or item.get("criticality") or "tracked",
            }
            latitude = item.get("lat", item.get("anchor_lat"))
            longitude = item.get("lon", item.get("anchor_lon"))
            if not (isinstance(latitude, (int, float)) and isinstance(longitude, (int, float))):
                resolved = location_lookup.get(str(item.get("location", name)).strip().lower())
                if resolved is not None:
                    latitude, longitude = resolved
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
                if section_name == "alliance_anchors" and isinstance(item.get("location"), str):
                    entry["notes"] = f"{item['location']}: {item['function']}"
                else:
                    entry["notes"] = item["function"]
            inventory.append(entry)

        for section_name in PHYSICAL_ASSET_SECTIONS:
            section = assets.get(section_name, [])
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict):
                        append_asset(item, section_name)

        return inventory

    @staticmethod
    def _asset_id(agent_id: str, section_name: str, name: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
        return f"{agent_id}-{section_name}-{slug}"

    @staticmethod
    def _asset_location_lookup(assets: dict[str, object]) -> dict[str, tuple[float, float]]:
        lookup: dict[str, tuple[float, float]] = {}
        for section_name in PHYSICAL_ASSET_SECTIONS:
            section = assets.get(section_name, [])
            if not isinstance(section, list):
                continue
            for item in section:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("location")
                latitude = item.get("lat", item.get("anchor_lat"))
                longitude = item.get("lon", item.get("anchor_lon"))
                if isinstance(name, str) and isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
                    lookup[name.strip().lower()] = (latitude, longitude)
        return lookup

    def _apply_asset_action_effects(self, world: WorldState, actor_id: str, action: AgentAction) -> None:
        if action.type == "strike" and action.target in AGENT_IDS:
            self._damage_assets(
                world,
                owner=action.target,
                intensity=42.0,
                reason=f"{actor_id} strike pressure",
                section_bias=ASSET_CATEGORY_BIAS.get(actor_id, {}).get("strike", ()),
                max_assets=2,
            )
        elif action.type == "deceive" and action.target in AGENT_IDS:
            self._damage_assets(
                world,
                owner=action.target,
                intensity=22.0,
                reason=f"{actor_id} deception caused systems malfunction",
                section_bias=ASSET_CATEGORY_BIAS.get(actor_id, {}).get("deceive", ()),
                max_assets=1,
                max_status="malfunctioning",
            )
        elif action.type == "sanction" and action.target in AGENT_IDS:
            self._damage_assets(
                world,
                owner=action.target,
                intensity=18.0,
                reason=f"{actor_id} sanction pressure degraded asset availability",
                section_bias=ASSET_CATEGORY_BIAS.get(actor_id, {}).get("sanction", ()),
                max_assets=1,
                max_status="malfunctioning",
            )
        elif action.type in {"defend", "oversight_review"}:
            defended_owner = actor_id if action.type == "oversight_review" or action.target not in AGENT_IDS else action.target
            self._restore_assets(
                world,
                owner=defended_owner,
                intensity=18.0 if action.type == "defend" else 12.0,
                reason=f"{actor_id} {action.type} hardened critical assets",
                section_bias=ASSET_CATEGORY_BIAS.get(actor_id, {}).get("defend", ()),
                max_assets=2,
            )

    def _apply_signal_asset_effects(self, world: WorldState, signal: ExternalSignal) -> None:
        text = f"{signal.headline} {' '.join(signal.tags)} {(signal.region or '')}".lower()
        severity = max(0.0, min(1.0, signal.severity))
        if self._signal_mentions(text, "strike", "rocket", "missile", "drone", "attack", "explosion", "raid"):
            for owner in self._infer_affected_agents(signal):
                self._damage_assets(
                    world,
                    owner=owner,
                    intensity=24.0 * severity,
                    reason=f"fresh reporting from {signal.source}",
                    section_bias=("front", "airbase", "base", "launch-network", "launch-zone", "energy-port"),
                    max_assets=1,
                    max_status="malfunctioning",
                )
        if self._signal_mentions(text, "shipping", "tanker", "hormuz", "port", "terminal", "oil"):
            for owner in ("us", "gulf", "iran"):
                self._damage_assets(
                    world,
                    owner=owner,
                    intensity=16.0 * severity,
                    reason=f"{signal.source} reported shipping disruption",
                    section_bias=("port", "energy", "energy-port", "chokepoint", "maritime-box"),
                    max_assets=1,
                    max_status="malfunctioning",
                )
        if self._signal_mentions(text, "cyber", "outage", "blackout", "internet"):
            for owner in self._infer_affected_agents(signal):
                self._damage_assets(
                    world,
                    owner=owner,
                    intensity=14.0 * severity,
                    reason=f"{signal.source} reported systems disruption",
                    section_bias=("command", "command-system", "command-network", "civil-center"),
                    max_assets=1,
                    max_status="malfunctioning",
                )

    def _damage_assets(
        self,
        world: WorldState,
        *,
        owner: str,
        intensity: float,
        reason: str,
        section_bias: tuple[str, ...],
        max_assets: int,
        max_status: str | None = None,
    ) -> None:
        if owner not in world.asset_state:
            return
        selected_assets = self._select_assets_for_effect(world, owner, section_bias, max_assets=max_assets, reverse=False)
        for asset in selected_assets:
            current = world.asset_state[owner][asset.asset_id]
            previous_status = current.status
            current.health = round(self._clamp_percent(current.health - intensity), 2)
            current.operational_load = round(min(100.0, current.operational_load + intensity * 0.7), 2)
            current.status = self._derive_asset_status(current.health, current.operational_load, max_status=max_status)
            current.last_change_reason = reason
            self._apply_asset_metric_impacts(world, owner, current, direction="damage")
            if current.status != previous_status:
                self._register_latent_event(
                    world,
                    LatentEvent(
                        event_id=f"asset-{owner}-{asset.asset_id}-{world.turn}-{len(world.latent_events)}",
                        topic="shipping" if any(token in current.section.lower() or token in current.category.lower() for token in ("port", "maritime", "chokepoint", "energy")) else "border",
                        status="active",
                        severity=min(1.0, max(0.35, (100.0 - current.health) / 100.0)),
                        visibility="public",
                        reliability=0.7,
                        origin="asset-state",
                        affected_agents=[owner],
                        affected_assets=[current.asset_id],
                        started_at_turn=world.turn,
                        last_updated_turn=world.turn,
                        decay_rate=0.06,
                        narratives=self._default_latent_event_narratives(
                            "shipping" if any(token in current.section.lower() or token in current.category.lower() for token in ("port", "maritime", "chokepoint", "energy")) else "border",
                            f"{current.name} is now {current.status} after {reason}.",
                        ),
                    ),
                )

    def _restore_assets(
        self,
        world: WorldState,
        *,
        owner: str,
        intensity: float,
        reason: str,
        section_bias: tuple[str, ...],
        max_assets: int,
    ) -> None:
        if owner not in world.asset_state:
            return
        selected_assets = self._select_assets_for_effect(world, owner, section_bias, max_assets=max_assets, reverse=True)
        for asset in selected_assets:
            current = world.asset_state[owner][asset.asset_id]
            current.health = round(self._clamp_percent(current.health + intensity), 2)
            current.operational_load = round(max(0.0, current.operational_load - intensity * 0.8), 2)
            current.status = self._derive_asset_status(current.health, current.operational_load)
            current.last_change_reason = reason
            self._apply_asset_metric_impacts(world, owner, current, direction="repair")

    def _select_assets_for_effect(
        self,
        world: WorldState,
        owner: str,
        section_bias: tuple[str, ...],
        *,
        max_assets: int,
        reverse: bool,
    ) -> list[AssetCondition]:
        assets = list(world.asset_state.get(owner, {}).values())
        if not assets:
            return []

        bias_terms = tuple(term.lower() for term in section_bias)

        def asset_key(asset: AssetCondition) -> tuple[float, float, str]:
            priority = float(ASSET_PRIORITY_SCORES.get(asset.criticality.lower(), 1))
            bias_score = 1.0 if any(term in asset.category.lower() or term in asset.section.lower() for term in bias_terms) else 0.0
            damage_score = 100.0 - asset.health if reverse else asset.health
            return (priority + bias_score, damage_score, asset.name)

        sorted_assets = sorted(assets, key=asset_key, reverse=True)
        if reverse:
            sorted_assets = [asset for asset in sorted_assets if asset.status != "operational"] or sorted_assets
        else:
            sorted_assets = [asset for asset in sorted_assets if asset.status != "destroyed"] or sorted_assets
        return sorted_assets[:max_assets]

    @staticmethod
    def _derive_asset_status(health: float, operational_load: float, max_status: str | None = None) -> str:
        derived = "operational"
        for threshold, status in ASSET_STATUS_DAMAGE_THRESHOLDS:
            if health <= threshold:
                derived = status
                break
        if derived == "operational" and operational_load >= 72.0:
            derived = "malfunctioning"
        if max_status == "malfunctioning" and derived == "destroyed":
            return "malfunctioning"
        return derived

    def _apply_asset_metric_impacts(
        self,
        world: WorldState,
        owner: str,
        asset: AssetCondition,
        *,
        direction: str,
    ) -> None:
        scale = -1.0 if direction == "damage" else 0.6
        category = f"{asset.section} {asset.category}".lower()
        owner_metric_map = {
            "us": {
                "regional_access": ("base", "airbase", "logistics", "command"),
                "shipping_security": ("naval", "port", "maritime", "chokepoint"),
                "force_posture": ("base", "airbase", "command"),
                "domestic_support": ("command", "capital"),
            },
            "israel": {
                "homeland_security": ("front", "civil", "infrastructure", "air-defense"),
                "northern_deterrence": ("front", "launch", "command"),
                "reserve_endurance": ("depth", "logistics", "port"),
                "us_resupply_confidence": ("port", "offshore", "command"),
            },
            "iran": {
                "regime_stability": ("command", "capital", "civil"),
                "proxy_corridor": ("corridor", "logistics", "front"),
                "hormuz_leverage": ("chokepoint", "maritime", "energy-port"),
                "deterrence_credibility": ("front", "command", "maritime"),
            },
            "hezbollah": {
                "launch_survivability": ("launch", "front", "depth"),
                "logistics_depth": ("logistics", "corridor", "reserve"),
                "political_cover": ("political", "command", "civil"),
                "resistance_credibility": ("launch", "front", "command"),
            },
            "gulf": {
                "shipping_continuity": ("port", "chokepoint", "maritime"),
                "infrastructure_security": ("energy", "capital", "port"),
                "investor_confidence": ("energy", "capital", "port"),
                "diplomatic_flexibility": ("capital", "port", "chokepoint"),
            },
            "oversight": {
                "runaway_risk": ("chokepoint", "theater", "civil"),
                "intervention_legitimacy": ("civil", "theater"),
                "autonomy_balance": ("theater", "command"),
                "trace_clarity": ("command", "civil", "theater"),
            },
        }
        for metric, keywords in owner_metric_map.get(owner, {}).items():
            if any(keyword in category for keyword in keywords):
                magnitude = 2.8 if direction == "damage" else 1.4
                self._bump_actor_metric(world, owner, metric, scale * magnitude)

    def _asset_pressure(self, world: WorldState, agent_id: str) -> float:
        assets = list(world.asset_state.get(agent_id, {}).values())
        if not assets:
            return 0.0
        weighted_damage = 0.0
        max_weight = 0.0
        for asset in assets:
            priority = float(ASSET_PRIORITY_SCORES.get(asset.criticality.lower(), 1))
            weighted_damage += priority * max(0.0, 100.0 - asset.health)
            max_weight += priority * 100.0
        if max_weight == 0.0:
            return 0.0
        return min(1.0, weighted_damage / max_weight)

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

    def _reward_metric(self, world: WorldState, agent_id: str, metric: str) -> float:
        config = AGENT_REWARD_METRIC_CONFIGS[agent_id][metric]
        return self._state_score(world, agent_id, metric, config.target, config.tolerance)

    def _action_response_score(self, world: WorldState, agent_id: str, recent_actions: dict[str, AgentAction]) -> float:
        action = recent_actions.get(agent_id)
        if action is None:
            return 0.0

        effects = AGENT_STATE_ACTION_EFFECTS.get(agent_id, {}).get(action.type, {})
        if not effects:
            return -0.25

        weighted_total = 0.0
        total_weight = 0.0
        metric_configs = AGENT_REWARD_METRIC_CONFIGS[agent_id]

        for metric, delta in effects.items():
            config = metric_configs.get(metric)
            if config is None:
                continue
            metric_score = self._reward_metric(world, agent_id, metric)
            direction = 1.0 if delta >= 0 else -1.0
            magnitude = min(abs(delta) / 4.0, 1.0)
            weight = config.weight * magnitude
            weighted_total += direction * metric_score * weight
            total_weight += weight

        if total_weight == 0.0:
            return -0.25

        return self._clamp_unit(weighted_total / total_weight)

    @staticmethod
    def _blend_reward_total(
        metric_weights: dict[str, float],
        metric_scores: dict[str, float],
        behavior: float,
        action_response: float,
    ) -> float:
        metric_total = sum(metric_weights[metric] * metric_scores[metric] for metric in metric_scores)
        total_weight = sum(metric_weights.values()) + 0.10 + 0.08
        return max(-1.0, min(1.0, (metric_total + 0.10 * behavior + 0.08 * action_response) / total_weight))

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
        metric_weights = {
            metric: config.weight for metric, config in AGENT_REWARD_METRIC_CONFIGS["us"].items()
        }
        regional_access = self._reward_metric(world, "us", "regional_access")
        shipping_stability = self._reward_metric(world, "us", "shipping_security")
        domestic_resilience = self._reward_metric(world, "us", "domestic_support")
        force_posture = self._reward_metric(world, "us", "force_posture")
        behavior = self._behavior_score(world, "us", recent_actions)
        action_response = self._action_response_score(world, "us", recent_actions)
        total = self._blend_reward_total(
            metric_weights,
            {
                "regional_access": regional_access,
                "shipping_security": shipping_stability,
                "domestic_support": domestic_resilience,
                "force_posture": force_posture,
            },
            behavior,
            action_response,
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
                "action_response": action_response,
            },
        )

    def _reward_israel(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        metric_weights = {
            metric: config.weight for metric, config in AGENT_REWARD_METRIC_CONFIGS["israel"].items()
        }
        homeland_security = self._reward_metric(world, "israel", "homeland_security")
        northern_deterrence = self._reward_metric(world, "israel", "northern_deterrence")
        reserve_endurance = self._reward_metric(world, "israel", "reserve_endurance")
        us_backstop = self._reward_metric(world, "israel", "us_resupply_confidence")
        behavior = self._behavior_score(world, "israel", recent_actions)
        action_response = self._action_response_score(world, "israel", recent_actions)
        total = self._blend_reward_total(
            metric_weights,
            {
                "homeland_security": homeland_security,
                "northern_deterrence": northern_deterrence,
                "us_resupply_confidence": us_backstop,
                "reserve_endurance": reserve_endurance,
            },
            behavior,
            action_response,
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
                "action_response": action_response,
            },
        )

    def _reward_iran(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        metric_weights = {
            metric: config.weight for metric, config in AGENT_REWARD_METRIC_CONFIGS["iran"].items()
        }
        regime_survival = self._reward_metric(world, "iran", "regime_stability")
        proxy_axis_integrity = self._reward_metric(world, "iran", "proxy_corridor")
        chokepoint_leverage = self._reward_metric(world, "iran", "hormuz_leverage")
        deterrence_credibility = self._reward_metric(world, "iran", "deterrence_credibility")
        behavior = self._behavior_score(world, "iran", recent_actions)
        action_response = self._action_response_score(world, "iran", recent_actions)
        total = self._blend_reward_total(
            metric_weights,
            {
                "regime_stability": regime_survival,
                "proxy_corridor": proxy_axis_integrity,
                "hormuz_leverage": chokepoint_leverage,
                "deterrence_credibility": deterrence_credibility,
            },
            behavior,
            action_response,
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
                "action_response": action_response,
            },
        )

    def _reward_hezbollah(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        metric_weights = {
            metric: config.weight for metric, config in AGENT_REWARD_METRIC_CONFIGS["hezbollah"].items()
        }
        launch_survivability = self._reward_metric(world, "hezbollah", "launch_survivability")
        logistics_depth = self._reward_metric(world, "hezbollah", "logistics_depth")
        political_cover = self._reward_metric(world, "hezbollah", "political_cover")
        resistance_credibility = self._reward_metric(world, "hezbollah", "resistance_credibility")
        iran_backing = self._clamp_unit(0.6 * self._alliance_score(world, "hezbollah") + 0.4 * logistics_depth)
        behavior = self._behavior_score(world, "hezbollah", recent_actions)
        action_response = self._action_response_score(world, "hezbollah", recent_actions)
        total = self._blend_reward_total(
            metric_weights,
            {
                "launch_survivability": launch_survivability,
                "logistics_depth": logistics_depth,
                "resistance_credibility": resistance_credibility,
                "political_cover": political_cover,
            },
            behavior,
            action_response,
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
                "action_response": action_response,
            },
        )

    def _reward_gulf(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        metric_weights = {
            metric: config.weight for metric, config in AGENT_REWARD_METRIC_CONFIGS["gulf"].items()
        }
        shipping_continuity = self._reward_metric(world, "gulf", "shipping_continuity")
        infrastructure_security = self._reward_metric(world, "gulf", "infrastructure_security")
        investor_confidence = self._reward_metric(world, "gulf", "investor_confidence")
        diplomatic_flexibility = self._reward_metric(world, "gulf", "diplomatic_flexibility")
        behavior = self._behavior_score(world, "gulf", recent_actions)
        action_response = self._action_response_score(world, "gulf", recent_actions)
        total = self._blend_reward_total(
            metric_weights,
            {
                "shipping_continuity": shipping_continuity,
                "investor_confidence": investor_confidence,
                "infrastructure_security": infrastructure_security,
                "diplomatic_flexibility": diplomatic_flexibility,
            },
            behavior,
            action_response,
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
                "action_response": action_response,
            },
        )

    def _reward_oversight(
        self,
        world: WorldState,
        episode: EpisodeMetadata,
        recent_actions: dict[str, AgentAction],
    ) -> RewardBreakdown:
        metric_weights = {
            metric: config.weight for metric, config in AGENT_REWARD_METRIC_CONFIGS["oversight"].items()
        }
        risk_reduction = self._reward_metric(world, "oversight", "runaway_risk")
        intervention_legitimacy = self._reward_metric(world, "oversight", "intervention_legitimacy")
        autonomy_preservation = self._reward_metric(world, "oversight", "autonomy_balance")
        trace_clarity = self._reward_metric(world, "oversight", "trace_clarity")
        behavior = self._behavior_score(world, "oversight", recent_actions)
        action_response = self._action_response_score(world, "oversight", recent_actions)
        total = self._blend_reward_total(
            metric_weights,
            {
                "runaway_risk": risk_reduction,
                "autonomy_balance": autonomy_preservation,
                "intervention_legitimacy": intervention_legitimacy,
                "trace_clarity": trace_clarity,
            },
            behavior,
            action_response,
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
                "action_response": action_response,
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
