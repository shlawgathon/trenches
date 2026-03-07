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
    action_override: dict[str, str] = Field(default_factory=dict)


class AgentObservation(BaseModel):
    public_brief: list[IntelSnippet] = Field(default_factory=list)
    private_brief: list[IntelSnippet] = Field(default_factory=list)
    perceived_tension: float = 50.0
    known_coalitions: list[str] = Field(default_factory=list)
    event_log: list[BlackSwanEvent] = Field(default_factory=list)
    entity_profile: dict[str, Any] = Field(default_factory=dict)
    strategic_state: dict[str, float] = Field(default_factory=dict)
    strategic_assets: list[dict[str, Any]] = Field(default_factory=list)
    source_bundle: list[str] = Field(default_factory=list)
    training_source_bundle: list[str] = Field(default_factory=list)
    live_source_bundle: list[str] = Field(default_factory=list)
    source_packets: list[SourcePacket] = Field(default_factory=list)
    training_source_packets: list[SourcePacket] = Field(default_factory=list)
    live_source_packets: list[SourcePacket] = Field(default_factory=list)


class WorldState(BaseModel):
    turn: int = 0
    tension_level: float = 50.0
    market_stress: float = 30.0
    oil_pressure: float = 40.0
    actor_state: dict[str, dict[str, float]] = Field(default_factory=dict)
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
    source_queue_sizes: dict[str, int] = Field(default_factory=dict)


class EpisodeMetadata(BaseModel):
    max_turns: int = DEFAULT_MAX_TURNS
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
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


class SessionState(BaseModel):
    session_id: str
    seed: int | None = None
    world: WorldState
    observations: dict[str, AgentObservation] = Field(default_factory=dict)
    rewards: dict[str, RewardBreakdown] = Field(default_factory=dict)
    episode: EpisodeMetadata = Field(default_factory=EpisodeMetadata)
    recent_traces: list[StepTrace] = Field(default_factory=list)
    live: LiveSessionConfig = Field(default_factory=LiveSessionConfig)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CreateSessionRequest(BaseModel):
    seed: int | None = None
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    max_turns: int | None = None


class ResetSessionRequest(BaseModel):
    seed: int | None = None
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    max_turns: int | None = None


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


class ResetEnvRequest(BaseModel):
    seed: int | None = None
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    max_turns: int | None = None


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
