export type MonitoringTone = "stable" | "watch" | "critical" | "idle";

export type MonitoringTool = {
  id: string;
  label: string;
  category?: string;
  readiness?: "ready" | "degraded" | "offline";
};

export type MonitoringIntelSnippet = {
  source: string;
  summary: string;
  confidence?: number | null;
};

export type MonitoringSourcePacket = {
  sourceId: string;
  sourceName: string;
  delivery: "training_core" | "live_demo" | string;
  status: "ok" | "pending" | "error" | string;
  kind?: string;
  fetchedAt?: string | null;
};

export type MonitoringSourceCounters = {
  ok?: number;
  pending?: number;
  error?: number;
  trainingCore?: number;
  liveDemo?: number;
  total?: number;
};

export type MonitoringStrategicAsset = {
  name: string;
  category?: string;
  status?: string;
  latitude?: number;
  longitude?: number;
  notes?: string;
};

export type MonitoringObservationInput = {
  perceivedTension?: number | null;
  knownCoalitions?: string[];
  publicBrief?: MonitoringIntelSnippet[];
  privateBrief?: MonitoringIntelSnippet[];
  strategicAssets?: MonitoringStrategicAsset[];
  sourcePackets?: MonitoringSourcePacket[];
  sourcePacketCounts?: MonitoringSourceCounters;
  sourceBundle?: string[];
  trainingSourceBundle?: string[];
  liveSourceBundle?: string[];
  tools?: MonitoringTool[];
};

export type MonitoringRewardInput = {
  total: number;
  coalitionStability?: number;
  escalationPenalty?: number;
  marketGain?: number;
  behavioralConsistency?: number;
  goalTerms?: Record<string, number>;
};

export type MonitoringActionTrace = {
  id?: string;
  turn?: number;
  createdAt?: string | null;
  actor: string;
  type: string;
  summary: string;
  target?: string | null;
  rewardTotal?: number | null;
  tensionDelta?: number | null;
  oversightTriggered?: boolean;
};

export type MonitoringModelMeta = {
  family?: string;
  variant?: string;
  sizeLabel?: string;
  contextWindow?: string;
  trainingStack?: string;
};

export type MonitoringAgentSnapshot = {
  id: string;
  displayName: string;
  color?: string;
  accent?: string;
  doctrine?: string;
  currentObjective?: string;
  postureLabel?: string;
  statusTone?: MonitoringTone;
  statusLabel?: string;
  lastDecisionAt?: string | null;
  confidence?: number | null;
  model?: MonitoringModelMeta;
  observation: MonitoringObservationInput;
  reward: MonitoringRewardInput;
  recentActions?: MonitoringActionTrace[];
};

export type MonitoringDeckProps = {
  agents: MonitoringAgentSnapshot[];
  selectedAgentId?: string | null;
  onSelectAgent?: (agentId: string) => void;
  title?: string;
  eyebrow?: string;
  headline?: string;
  summary?: string;
  statusChip?: string;
};

export type MonitoringPanelSharedProps = {
  agents: MonitoringAgentSnapshot[];
  selectedAgentId?: string | null;
  onSelectAgent?: (agentId: string) => void;
  title?: string;
  eyebrow?: string;
};
