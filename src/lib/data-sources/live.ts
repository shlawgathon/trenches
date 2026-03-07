import { getLiveSourcesForAgent } from "./registry";
import type { AgentId, DataSourceSpec, LiveSourcePlanItem } from "./types";

function defaultPollIntervalMs(source: DataSourceSpec): number {
  switch (source.kind) {
    case "telegram":
      return 15_000;
    case "structured":
    case "api":
      return 30_000;
    case "rss":
      return 60_000;
    case "scrape":
      return 120_000;
    case "video":
      return 180_000;
  }
}

function defaultMaxItemsPerPull(source: DataSourceSpec): number {
  switch (source.kind) {
    case "telegram":
      return 20;
    case "structured":
    case "api":
      return 10;
    case "rss":
      return 8;
    case "scrape":
      return 5;
    case "video":
      return 1;
  }
}

function isWarmStartSource(source: DataSourceSpec): boolean {
  return source.kind === "telegram" || source.kind === "structured" || source.tags.includes("official");
}

export function buildLiveSourcePlan(agentId: AgentId): LiveSourcePlanItem[] {
  return getLiveSourcesForAgent(agentId).map((source) => ({
    sourceId: source.id,
    pollIntervalMs: defaultPollIntervalMs(source),
    warmStart: isWarmStartSource(source),
    maxItemsPerPull: defaultMaxItemsPerPull(source),
  }));
}
