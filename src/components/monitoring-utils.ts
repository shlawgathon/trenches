import type {
  MonitoringActionTrace,
  MonitoringAgentSnapshot,
  MonitoringObservationInput,
  MonitoringSourceCounters,
  MonitoringTone,
} from "./monitoring-types";

export type ResolvedSourceHealth = {
  ok: number;
  pending: number;
  error: number;
  trainingCore: number;
  liveDemo: number;
  total: number;
  lastFetchedAt?: string | null;
};

export function clamp(value: number, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

export function formatMaybe(value: number | null | undefined, digits = 1) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

export function formatSigned(value: number | null | undefined, digits = 2) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  const normalized = value >= 0 ? `+${value.toFixed(digits)}` : value.toFixed(digits);
  return normalized;
}

export function formatPercent(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${Math.round(clamp(value) * 100)}%`;
}

export function formatTimeLabel(value: string | null | undefined) {
  if (!value) {
    return "No sync yet";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function titleCase(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function summarizeSourceHealth(observation: MonitoringObservationInput | undefined): ResolvedSourceHealth {
  const counts: MonitoringSourceCounters = observation?.sourcePacketCounts ?? {};
  const packets = observation?.sourcePackets ?? [];
  const fromPackets = {
    ok: packets.filter((packet) => packet.status === "ok").length,
    pending: packets.filter((packet) => packet.status === "pending").length,
    error: packets.filter((packet) => packet.status === "error").length,
    trainingCore: packets.filter((packet) => packet.delivery === "training_core").length,
    liveDemo: packets.filter((packet) => packet.delivery === "live_demo").length,
  };

  const timestamps = packets
    .map((packet) => packet.fetchedAt ?? null)
    .filter((value): value is string => typeof value === "string" && value.length > 0)
    .sort();

  const trainingFallback = observation?.trainingSourceBundle?.length ?? 0;
  const liveFallback = observation?.liveSourceBundle?.length ?? 0;
  const bundledTotal = observation?.sourceBundle?.length ?? 0;
  const inferredTotal = trainingFallback > 0 || liveFallback > 0 ? trainingFallback + liveFallback : 0;
  const totalFallback =
    counts.total ??
    (packets.length > 0 ? packets.length : bundledTotal > 0 ? bundledTotal : inferredTotal);

  return {
    ok: counts.ok ?? fromPackets.ok,
    pending: counts.pending ?? fromPackets.pending,
    error: counts.error ?? fromPackets.error,
    trainingCore: counts.trainingCore ?? (fromPackets.trainingCore || trainingFallback),
    liveDemo: counts.liveDemo ?? (fromPackets.liveDemo || liveFallback),
    total: totalFallback,
    lastFetchedAt: timestamps[timestamps.length - 1] ?? null,
  };
}

export function resolveAgentTone(agent: MonitoringAgentSnapshot): MonitoringTone {
  if (agent.statusTone) {
    return agent.statusTone;
  }
  const health = summarizeSourceHealth(agent.observation);
  const tensionValue = agent.observation.perceivedTension ?? 0;
  const tension = tensionValue > 1 ? clamp(tensionValue / 100) : tensionValue;
  const confidence = agent.confidence ?? 0.5;
  if (health.error > 0 || tension >= 0.75 || agent.reward.total <= -0.35) {
    return "critical";
  }
  if (health.pending > 0 || tension >= 0.45 || confidence <= 0.45) {
    return "watch";
  }
  if (health.total === 0) {
    return "idle";
  }
  return "stable";
}

export function getToneLabel(tone: MonitoringTone) {
  switch (tone) {
    case "stable":
      return "Stable";
    case "watch":
      return "Watch";
    case "critical":
      return "Critical";
    default:
      return "Idle";
  }
}

export function getFocusedAgent(agents: MonitoringAgentSnapshot[], selectedAgentId?: string | null) {
  if (selectedAgentId) {
    const selected = agents.find((agent) => agent.id === selectedAgentId);
    if (selected) {
      return selected;
    }
  }
  return agents[0];
}

export function getLatestAction(agent: MonitoringAgentSnapshot) {
  const actions = agent.recentActions ?? [];
  return [...actions].sort(compareActions)[0];
}

export function flattenRecentActions(agents: MonitoringAgentSnapshot[]) {
  const actions: Array<MonitoringActionTrace & { displayName: string; color?: string }> = [];
  for (const agent of agents) {
    for (const action of agent.recentActions ?? []) {
      actions.push({
        ...action,
        displayName: agent.displayName,
        color: agent.color,
      });
    }
  }
  return actions.sort(compareActions);
}

export function compareActions(left: MonitoringActionTrace, right: MonitoringActionTrace) {
  const leftTime = left.createdAt ? Date.parse(left.createdAt) : Number.NEGATIVE_INFINITY;
  const rightTime = right.createdAt ? Date.parse(right.createdAt) : Number.NEGATIVE_INFINITY;
  if (leftTime !== rightTime) {
    return rightTime - leftTime;
  }
  return (right.turn ?? 0) - (left.turn ?? 0);
}

export function getModelLabel(agent: MonitoringAgentSnapshot) {
  const parts = [agent.model?.family, agent.model?.variant, agent.model?.sizeLabel].filter(Boolean);
  if (parts.length === 0) {
    return "Unspecified model";
  }
  return parts.join(" / ");
}

export function rewardEntries(agent: MonitoringAgentSnapshot) {
  const entries = [
    {
      key: "coalitionStability",
      label: "Coalition Stability",
      value: agent.reward.coalitionStability ?? 0,
    },
    {
      key: "escalationPenalty",
      label: "Escalation Penalty",
      value: agent.reward.escalationPenalty ?? 0,
    },
    {
      key: "marketGain",
      label: "Market Gain",
      value: agent.reward.marketGain ?? 0,
    },
    {
      key: "behavioralConsistency",
      label: "Behavioral Consistency",
      value: agent.reward.behavioralConsistency ?? 0,
    },
  ];

  const goalTerms = Object.entries(agent.reward.goalTerms ?? {}).map(([key, value]) => ({
    key,
    label: titleCase(key),
    value,
  }));

  return {
    baseline: entries,
    doctrine: goalTerms,
  };
}

export function topIntelSnippets(agent: MonitoringAgentSnapshot, limit = 4) {
  return [...(agent.observation.privateBrief ?? []), ...(agent.observation.publicBrief ?? [])].slice(0, limit);
}
