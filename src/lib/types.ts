export type IntelSnippet = {
  source: string;
  category: string;
  summary: string;
  confidence: number;
};

export type SourcePacket = {
  source_id: string;
  source_name: string;
  delivery: "training_core" | "live_demo";
  kind: string;
  endpoint_kind: string;
  status: "pending" | "ok" | "error";
  fetched_at?: string | null;
  probe_url?: string | null;
  summary: string;
  sample_items: string[];
  error?: string | null;
};

export type BlackSwanEvent = {
  id: string;
  summary: string;
  source: string;
  severity: number;
  public: boolean;
  affected_agents: string[];
};

export type AgentAction = {
  actor: string;
  type:
    | "hold"
    | "negotiate"
    | "sanction"
    | "strike"
    | "defend"
    | "intel_query"
    | "mobilize"
    | "deceive"
    | "oversight_review";
  summary: string;
  target?: string | null;
  metadata?: Record<string, unknown>;
};

export type ExternalSignal = {
  source: string;
  headline: string;
  region?: string | null;
  tags?: string[];
  severity?: number;
};

export type RewardBreakdown = {
  coalition_stability: number;
  escalation_penalty: number;
  market_gain: number;
  behavioral_consistency: number;
  goal_terms: Record<string, number>;
  total: number;
};

export type OversightIntervention = {
  triggered: boolean;
  risk_score: number;
  reason: string;
  affected_agents: string[];
  action_override: Record<string, string>;
};

export type AgentObservation = {
  public_brief: IntelSnippet[];
  private_brief: IntelSnippet[];
  perceived_tension: number;
  known_coalitions: string[];
  event_log: BlackSwanEvent[];
  entity_profile: Record<string, unknown>;
  strategic_state: Record<string, number>;
  strategic_assets: Record<string, unknown>[];
  source_bundle: string[];
  training_source_bundle: string[];
  live_source_bundle: string[];
  source_packets: SourcePacket[];
  training_source_packets: SourcePacket[];
  live_source_packets: SourcePacket[];
};

export type WorldState = {
  turn: number;
  tension_level: number;
  market_stress: number;
  oil_pressure: number;
  actor_state: Record<string, Record<string, number>>;
  coalition_graph: Record<string, string[]>;
  active_events: BlackSwanEvent[];
  hidden_intents: Record<string, string>;
  behavioral_consistency: Record<string, number>;
  ema_tension: Record<string, number>;
  risk_scores: Record<string, number>;
  last_actions: AgentAction[];
};

export type LiveSessionConfig = {
  enabled: boolean;
  auto_step: boolean;
  poll_interval_ms: number;
  started_at?: string | null;
  last_source_sync_at?: string | null;
  source_queue_sizes: Record<string, number>;
};

export type EpisodeMetadata = {
  max_turns: number;
  training_stage: "stage_1_dense" | "stage_2_partial" | "stage_3_sparse";
  algorithm_hints: Record<string, string>;
  dense_rewards: boolean;
  sparse_rewards: boolean;
  fog_of_war: boolean;
  oversight_enabled: boolean;
  credit_assignment: string;
  live_mode_capable: boolean;
  live_mode_inference_only: boolean;
};

export type StepTrace = {
  turn: number;
  tension_before: number;
  tension_after: number;
  actions: Record<string, AgentAction>;
  rewards: Record<string, RewardBreakdown>;
  oversight: OversightIntervention;
  created_at: string;
};

export type SessionState = {
  session_id: string;
  seed?: number | null;
  world: WorldState;
  observations: Record<string, AgentObservation>;
  rewards: Record<string, RewardBreakdown>;
  episode: EpisodeMetadata;
  recent_traces: StepTrace[];
  live: LiveSessionConfig;
  created_at: string;
  updated_at: string;
};

export type CreateSessionRequest = {
  seed?: number;
  training_stage?: "stage_1_dense" | "stage_2_partial" | "stage_3_sparse";
  max_turns?: number;
};

export type ResetSessionRequest = {
  seed?: number;
  training_stage?: "stage_1_dense" | "stage_2_partial" | "stage_3_sparse";
  max_turns?: number;
};

export type LiveControlRequest = {
  enabled: boolean;
  auto_step?: boolean;
  poll_interval_ms?: number;
};

export type StepSessionRequest = {
  actions: Record<string, AgentAction>;
  external_signals?: ExternalSignal[];
};

export type StepSessionResponse = {
  session: SessionState;
  oversight: OversightIntervention;
  done: boolean;
};
