import "./monitoring.css";

import { ModelStatusPanel } from "./ModelStatusPanel";
import { RecentActionsPanel } from "./RecentActionsPanel";
import { RewardSummaryPanel } from "./RewardSummaryPanel";
import { SourceHealthPanel } from "./SourceHealthPanel";
import type { MonitoringDeckProps } from "./monitoring-types";
import { getFocusedAgent } from "./monitoring-utils";

export type {
  MonitoringActionTrace,
  MonitoringAgentSnapshot,
  MonitoringDeckProps,
  MonitoringIntelSnippet,
  MonitoringModelMeta,
  MonitoringObservationInput,
  MonitoringRewardInput,
  MonitoringSourceCounters,
  MonitoringSourcePacket,
  MonitoringStrategicAsset,
  MonitoringTool,
} from "./monitoring-types";

export function MonitoringDeck({
  agents,
  selectedAgentId,
  onSelectAgent,
  title = "Model Monitoring",
  eyebrow = "Operational Supervision",
  headline = "Viewer-side command deck for model health, reward pressure, source integrity, and recent behavior.",
  summary = "Designed to sit beside the command map: the map shows the theater, this layer shows how each model is reading it and responding.",
  statusChip,
}: MonitoringDeckProps) {
  const focused = getFocusedAgent(agents, selectedAgentId);
  const activeToolCount = focused?.observation.tools?.length ?? 0;
  const activeAssetCount = focused?.observation.strategicAssets?.length ?? 0;

  return (
    <section className="monitoring-deck">
      <header className="monitoring-deck__header">
        <section className="monitoring-deck__hero">
          <p>{eyebrow}</p>
          <h2>{title}</h2>
          <p className="monitoring-deck__lede">{headline}</p>
        </section>

        <section className="monitoring-deck__summary">
          <span className="monitoring-chip">{statusChip ?? `${agents.length} agents online`}</span>
          <p className="monitoring-copy">{summary}</p>
          <div className="monitoring-kpi-grid">
            <div className="monitoring-kpi">
              <span>Focused Actor</span>
              <strong>{focused?.displayName ?? "--"}</strong>
            </div>
            <div className="monitoring-kpi">
              <span>Tracked Assets</span>
              <strong>{activeAssetCount}</strong>
            </div>
            <div className="monitoring-kpi">
              <span>Operational Tools</span>
              <strong>{activeToolCount}</strong>
            </div>
          </div>
        </section>
      </header>

      <div className="monitoring-deck__grid">
        <ModelStatusPanel agents={agents} selectedAgentId={selectedAgentId} onSelectAgent={onSelectAgent} />
        <div className="monitoring-deck__stack">
          <RewardSummaryPanel agents={agents} selectedAgentId={selectedAgentId} />
          <SourceHealthPanel agents={agents} selectedAgentId={selectedAgentId} />
        </div>
      </div>

      <RecentActionsPanel agents={agents} selectedAgentId={selectedAgentId} />
    </section>
  );
}
