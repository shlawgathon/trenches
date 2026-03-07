from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field
from trenches_env.rl import ALGORITHM_HINTS, DEFAULT_MAX_TURNS, DEFAULT_TRAINING_STAGE

ActionType = Literal[
    "hold",
    "negotiate",
    "sanction",
    "strike",
    "defend",
    "intel_query",
    "mobilize",
    "deceive",
    "oversight_review",
]

TrainingStage = Literal["stage_1_dense", "stage_2_partial", "stage_3_sparse"]
SourcePacketStatus = Literal["pending", "ok", "error"]
SourceMonitorStatus = Literal["healthy", "degraded", "blocked"]
SourceMonitorIssueSeverity = Literal["warning", "error"]
AssetConditionStatus = Literal["operational", "degraded", "malfunctioning", "destroyed"]
DecisionMode = Literal["heuristic_fallback", "provider_ready", "provider_inference"]
ModelProviderName = Literal["none", "openai", "anthropic", "openrouter", "ollama", "vllm", "custom"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IntelSnippet(BaseModel):
    source: str
    category: str
    summary: str
    confidence: float = 0.5


class SourcePacket(BaseModel):
    source_id: str
    source_name: str
    delivery: Literal["training_core", "live_demo"]
    kind: str
    endpoint_kind: str
    status: SourcePacketStatus = "pending"
    fetched_at: datetime | None = None
    probe_url: str | None = None
    summary: str = ""
    sample_items: list[str] = Field(default_factory=list)
    error: str | None = None


class DataSourceContext(BaseModel):
    source_id: str
    name: str
    delivery: Literal["training_core", "live_demo"]
    kind: str
    rationale: str
    tags: list[str] = Field(default_factory=list)
    access_notes: str | None = None


class AssetCondition(BaseModel):
    asset_id: str
    owner: str
    name: str
    category: str
    section: str
    latitude: float | None = None
    longitude: float | None = None
    status: AssetConditionStatus = "operational"
    health: float = 100.0
    operational_load: float = 0.0
    criticality: str = "tracked"
    notes: str | None = None
    last_change_reason: str | None = None


class ExternalSignal(BaseModel):
    source: str
    headline: str
    region: str | None = None
    tags: list[str] = Field(default_factory=list)
    severity: float = 0.5


class BlackSwanEvent(BaseModel):
    id: str
    summary: str
    source: str
    severity: float = 0.5
    public: bool = True
    affected_agents: list[str] = Field(default_factory=list)


class LatentEventNarrative(BaseModel):
    framing: Literal["baseline", "stabilizing", "deteriorating", "concealed"] = "baseline"
    summary: str
    confidence: float = 0.5
    public: bool = True


class LatentEvent(BaseModel):
    event_id: str
    topic: str
    status: Literal["emerging", "active", "contained", "resolved"] = "emerging"
    severity: float = 0.5
    visibility: Literal["public", "mixed", "private"] = "mixed"
    reliability: float = 0.6
    origin: str
    affected_agents: list[str] = Field(default_factory=list)
    affected_assets: list[str] = Field(default_factory=list)
    started_at_turn: int = 0
    last_updated_turn: int = 0
    decay_rate: float = 0.08
    linked_event_ids: list[str] = Field(default_factory=list)
    narratives: list[LatentEventNarrative] = Field(default_factory=list)


class AgentAction(BaseModel):
    actor: str
    type: ActionType
    summary: str
    target: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardBreakdown(BaseModel):
    coalition_stability: float = 0.0
    escalation_penalty: float = 0.0
    market_gain: float = 0.0
    behavioral_consistency: float = 0.0
    goal_terms: dict[str, float] = Field(default_factory=dict)
    total: float = 0.0


class OversightIntervention(BaseModel):
    triggered: bool = False
    risk_score: float = 0.0
    reason: str = ""
    affected_agents: list[str] = Field(default_factory=list)
    action_override: dict[str, AgentAction] = Field(default_factory=dict)


class ObservationProjection(BaseModel):
    enabled: bool = False
    mode: Literal["direct", "partial"] = "direct"
    worldview_reliability: float = 1.0
    delayed_source_count: int = 0
    contested_source_count: int = 0
    contradiction_packet_count: int = 0
    obscured_metric_count: int = 0
    contradiction_topics: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EntityModelBinding(BaseModel):
    agent_id: str
    provider: ModelProviderName = "none"
    model_name: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    configured: bool = False
    ready_for_inference: bool = False
    decision_mode: DecisionMode = "heuristic_fallback"
    supports_tool_calls: bool = False
    supports_structured_output: bool = False
    action_tools: list[str] = Field(default_factory=list)
    observation_tools: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AgentBeliefEntry(BaseModel):
    belief_id: str
    topic: str
    summary: str
    confidence: float = 0.5
    status: Literal["suspected", "active", "contested", "confirmed", "disconfirmed"] = "suspected"
    source: str = "latent_event"
    suspected_agents: list[str] = Field(default_factory=list)
    related_event_ids: list[str] = Field(default_factory=list)
    confirmation_count: int = 0
    contradiction_count: int = 0
    last_confirmed_turn: int | None = None
    last_updated_turn: int = 0


class AgentBeliefState(BaseModel):
    agent_id: str
    dominant_topics: list[str] = Field(default_factory=list)
    beliefs: list[AgentBeliefEntry] = Field(default_factory=list)
    last_revision_turn: int = 0


class AgentObservation(BaseModel):
    public_brief: list[IntelSnippet] = Field(default_factory=list)
    private_brief: list[IntelSnippet] = Field(default_factory=list)
    belief_brief: list[str] = Field(default_factory=list)
    belief_topics: list[str] = Field(default_factory=list)
    perceived_tension: float = 50.0
    known_coalitions: list[str] = Field(default_factory=list)
    event_log: list[BlackSwanEvent] = Field(default_factory=list)
    decision_prompt: str = ""
    available_actions: list[str] = Field(default_factory=list)
    available_data_sources: list[DataSourceContext] = Field(default_factory=list)
    entity_profile: dict[str, Any] = Field(default_factory=dict)
    strategic_state: dict[str, float] = Field(default_factory=dict)
    strategic_assets: list[dict[str, Any]] = Field(default_factory=list)
    asset_alerts: list[str] = Field(default_factory=list)
    source_bundle: list[str] = Field(default_factory=list)
    training_source_bundle: list[str] = Field(default_factory=list)
    live_source_bundle: list[str] = Field(default_factory=list)
    source_packets: list[SourcePacket] = Field(default_factory=list)
    training_source_packets: list[SourcePacket] = Field(default_factory=list)
    live_source_packets: list[SourcePacket] = Field(default_factory=list)
    projection: ObservationProjection = Field(default_factory=ObservationProjection)


class WorldState(BaseModel):
    turn: int = 0
    tension_level: float = 50.0
    market_stress: float = 30.0
    oil_pressure: float = 40.0
    latent_state: dict[str, dict[str, float]] = Field(default_factory=dict)
    latent_events: list[LatentEvent] = Field(default_factory=list)
    actor_state: dict[str, dict[str, float]] = Field(default_factory=dict)
    asset_state: dict[str, dict[str, AssetCondition]] = Field(default_factory=dict)
    coalition_graph: dict[str, list[str]] = Field(default_factory=dict)
    active_events: list[BlackSwanEvent] = Field(default_factory=list)
    hidden_intents: dict[str, str] = Field(default_factory=dict)
    behavioral_consistency: dict[str, float] = Field(default_factory=dict)
    ema_tension: dict[str, float] = Field(default_factory=dict)
    risk_scores: dict[str, float] = Field(default_factory=dict)
    last_actions: list[AgentAction] = Field(default_factory=list)


class LiveSessionConfig(BaseModel):
    enabled: bool = False
    auto_step: bool = False
    poll_interval_ms: int = 30_000
    started_at: datetime | None = None
    last_source_sync_at: datetime | None = None
    last_auto_step_at: datetime | None = None
    source_queue_sizes: dict[str, int] = Field(default_factory=dict)
    reacted_packet_fetched_at: dict[str, datetime] = Field(default_factory=dict)


class EpisodeMetadata(BaseModel):
    max_turns: int = DEFAULT_MAX_TURNS
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    scenario_id: str = "baseline_alert"
    scenario_name: str = "Baseline Alert Posture"
    scenario_description: str = ""
    scenario_tags: list[str] = Field(default_factory=list)
    algorithm_hints: dict[str, str] = Field(default_factory=lambda: ALGORITHM_HINTS.copy())
    dense_rewards: bool = False
    sparse_rewards: bool = True
    fog_of_war: bool = True
    oversight_enabled: bool = True
    credit_assignment: str = "CTDE"
    live_mode_capable: bool = True
    live_mode_inference_only: bool = True


class StepTrace(BaseModel):
    turn: int
    tension_before: float
    tension_after: float
    actions: dict[str, AgentAction] = Field(default_factory=dict)
    rewards: dict[str, RewardBreakdown] = Field(default_factory=dict)
    oversight: OversightIntervention
    created_at: datetime = Field(default_factory=utc_now)


class ActionLogEntry(BaseModel):
    turn: int
    actor: str
    action_type: ActionType
    summary: str
    target: str | None = None
    reward_total: float = 0.0
    tension_after: float = 0.0
    market_stress_after: float = 0.0
    oil_pressure_after: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ReactionActorOutcome(BaseModel):
    agent_id: str
    action: AgentAction
    reward_total: float = 0.0
    decision_mode: DecisionMode = "heuristic_fallback"


class ReactionLogEntry(BaseModel):
    event_id: str
    turn: int
    source: str = "public_release"
    latent_event_ids: list[str] = Field(default_factory=list)
    signals: list[ExternalSignal] = Field(default_factory=list)
    actor_outcomes: list[ReactionActorOutcome] = Field(default_factory=list)
    oversight_triggered: bool = False
    tension_before: float = 0.0
    tension_after: float = 0.0
    market_stress_after: float = 0.0
    oil_pressure_after: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)


class SessionState(BaseModel):
    session_id: str
    seed: int | None = None
    world: WorldState
    observations: dict[str, AgentObservation] = Field(default_factory=dict)
    belief_state: dict[str, AgentBeliefState] = Field(default_factory=dict)
    rewards: dict[str, RewardBreakdown] = Field(default_factory=dict)
    model_bindings: dict[str, EntityModelBinding] = Field(default_factory=dict)
    episode: EpisodeMetadata = Field(default_factory=EpisodeMetadata)
    recent_traces: list[StepTrace] = Field(default_factory=list)
    action_log: list[ActionLogEntry] = Field(default_factory=list)
    reaction_log: list[ReactionLogEntry] = Field(default_factory=list)
    live: LiveSessionConfig = Field(default_factory=LiveSessionConfig)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SourceMonitorIssue(BaseModel):
    severity: SourceMonitorIssueSeverity
    message: str


class AgentSourceMonitor(BaseModel):
    agent_id: str
    display_name: str
    status: SourceMonitorStatus = "healthy"
    configured_training_sources: int = 0
    configured_live_sources: int = 0
    active_source_count: int = 0
    ok_packet_count: int = 0
    pending_packet_count: int = 0
    error_packet_count: int = 0
    available_training_packet_count: int = 0
    available_live_packet_count: int = 0
    delivered_training_brief_count: int = 0
    delivered_live_brief_count: int = 0
    missing_training_sources: list[str] = Field(default_factory=list)
    missing_live_sources: list[str] = Field(default_factory=list)
    unbundled_training_sources: list[str] = Field(default_factory=list)
    unbundled_live_sources: list[str] = Field(default_factory=list)
    missing_packet_sources: list[str] = Field(default_factory=list)
    sources_without_probe_targets: list[str] = Field(default_factory=list)
    stale_sources: list[str] = Field(default_factory=list)
    error_sources: list[str] = Field(default_factory=list)
    pending_sources: list[str] = Field(default_factory=list)
    delivered_source_names: list[str] = Field(default_factory=list)
    issues: list[SourceMonitorIssue] = Field(default_factory=list)


class SourceMonitorSummary(BaseModel):
    healthy_agents: int = 0
    degraded_agents: int = 0
    blocked_agents: int = 0
    active_source_count: int = 0
    ok_packet_count: int = 0
    delivered_source_brief_count: int = 0


class SourceMonitorReport(BaseModel):
    session_id: str
    live_enabled: bool = False
    generated_at: datetime = Field(default_factory=utc_now)
    summary: SourceMonitorSummary = Field(default_factory=SourceMonitorSummary)
    agents: list[AgentSourceMonitor] = Field(default_factory=list)


class CreateSessionRequest(BaseModel):
    seed: int | None = None
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    max_turns: int | None = None
    scenario_id: str | None = None


class ResetSessionRequest(BaseModel):
    seed: int | None = None
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    max_turns: int | None = None
    scenario_id: str | None = None


class LiveControlRequest(BaseModel):
    enabled: bool
    auto_step: bool = False
    poll_interval_ms: int = 30_000


class StepSessionRequest(BaseModel):
    actions: dict[str, AgentAction] = Field(default_factory=dict)
    external_signals: list[ExternalSignal] = Field(default_factory=list)


class StepSessionResponse(BaseModel):
    session: SessionState
    oversight: OversightIntervention
    done: bool = False


class IngestNewsRequest(BaseModel):
    signals: list[ExternalSignal] = Field(default_factory=list)
    agent_ids: list[str] = Field(default_factory=list)


class IngestNewsResponse(BaseModel):
    session: SessionState
    oversight: OversightIntervention
    reaction: ReactionLogEntry | None = None
    done: bool = False


class ProviderAgentDiagnostics(BaseModel):
    agent_id: str
    provider: ModelProviderName = "none"
    model_name: str = ""
    configured: bool = False
    ready_for_inference: bool = False
    decision_mode: DecisionMode = "heuristic_fallback"
    status: Literal["idle", "healthy", "degraded", "fallback_only"] = "idle"
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    consecutive_failures: int = 0
    last_latency_ms: float | None = None
    avg_latency_ms: float | None = None
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    last_error: str | None = None


class ProviderDiagnosticsResponse(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    agents: list[ProviderAgentDiagnostics] = Field(default_factory=list)


class ResetEnvRequest(BaseModel):
    seed: int | None = None
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    max_turns: int | None = None
    scenario_id: str | None = None


class ResetEnvResponse(BaseModel):
    observations: dict[str, AgentObservation] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)


class StepEnvRequest(BaseModel):
    actions: dict[str, AgentAction] = Field(default_factory=dict)
    external_signals: list[ExternalSignal] = Field(default_factory=list)


class StepEnvResponse(BaseModel):
    observations: dict[str, AgentObservation] = Field(default_factory=dict)
    rewards: dict[str, RewardBreakdown] = Field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class ScenarioSummary(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    benchmark_turns: int = 0
    benchmark_enabled: bool = True


class BenchmarkEntityScorecard(BaseModel):
    agent_id: str
    total_reward: float = 0.0
    mean_reward: float = 0.0
    final_reward: float = 0.0
    final_goal_terms: dict[str, float] = Field(default_factory=dict)
    aggregated_goal_terms: dict[str, float] = Field(default_factory=dict)
    final_state: dict[str, float] = Field(default_factory=dict)
    damaged_asset_count: int = 0
    asset_pressure: float = 0.0
    action_counts: dict[str, int] = Field(default_factory=dict)
    dominant_action: str | None = None
    warnings: list[str] = Field(default_factory=list)


class BenchmarkScenarioResult(BaseModel):
    scenario_id: str
    scenario_name: str
    seed: int | None = None
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    turns_executed: int = 0
    done: bool = False
    done_reason: str | None = None
    oversight_trigger_count: int = 0
    final_tension: float = 0.0
    final_market_stress: float = 0.0
    final_oil_pressure: float = 0.0
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    scorecards: dict[str, BenchmarkEntityScorecard] = Field(default_factory=dict)


class BenchmarkRunRequest(BaseModel):
    scenario_ids: list[str] = Field(default_factory=list)
    seed: int | None = None
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    steps_per_scenario: int | None = None


class BenchmarkRunResponse(BaseModel):
    seed: int | None = None
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    scenario_ids: list[str] = Field(default_factory=list)
    scenario_count: int = 0
    results: list[BenchmarkScenarioResult] = Field(default_factory=list)
    aggregate_mean_total_rewards: dict[str, float] = Field(default_factory=dict)
