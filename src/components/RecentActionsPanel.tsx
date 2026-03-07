import type { MonitoringPanelSharedProps } from "./monitoring-types";
import { flattenRecentActions, formatSigned, formatTimeLabel, getFocusedAgent } from "./monitoring-utils";

type RecentActionsPanelProps = MonitoringPanelSharedProps & {
  maxItems?: number;
};

export function RecentActionsPanel({
  agents,
  selectedAgentId,
  title = "Recent Actions",
  eyebrow = "Trace Timeline",
  maxItems = 10,
}: RecentActionsPanelProps) {
  const focused = getFocusedAgent(agents, selectedAgentId);
  const feed =
    focused && selectedAgentId
      ? [...(focused.recentActions ?? [])].slice(0, maxItems).map((action) => ({
          ...action,
          displayName: focused.displayName,
        }))
      : flattenRecentActions(agents).slice(0, maxItems);

  return (
    <section className="monitoring-panel">
      <header className="monitoring-panel__header">
        <div>
          <p className="monitoring-panel__eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
          <p>Action-by-action view of model behavior, reward impact, and oversight intervention markers.</p>
        </div>
        <aside>
          <span className="monitoring-chip">{selectedAgentId ? focused?.displayName ?? selectedAgentId : "All Actors"}</span>
          <span className="monitoring-label">{feed.length} entries shown</span>
        </aside>
      </header>

      {feed.length === 0 ? (
        <div className="monitoring-empty">No recent traces supplied.</div>
      ) : (
        <div className="monitoring-action-list">
          {feed.map((action, index) => (
            <article key={`${action.actor}-${action.turn ?? index}-${action.createdAt ?? "none"}`} className="monitoring-action-card">
              <div className="monitoring-action-head">
                <div>
                  <strong>{action.displayName ?? action.actor}</strong>
                  <p className="monitoring-action-summary">{action.summary}</p>
                </div>
                <div className="monitoring-action-badges">
                  <span className="monitoring-pill">{action.type}</span>
                  {action.target ? <span className="monitoring-pill">Target: {action.target}</span> : null}
                  {action.oversightTriggered ? <span className="monitoring-pill monitoring-pill--hot">Oversight</span> : null}
                </div>
              </div>

              <div className="monitoring-metric-strip">
                {typeof action.turn === "number" ? <span className="monitoring-pill">Turn {action.turn}</span> : null}
                <span className="monitoring-pill">{formatTimeLabel(action.createdAt)}</span>
                {typeof action.rewardTotal === "number" ? (
                  <span className="monitoring-pill">Reward {formatSigned(action.rewardTotal)}</span>
                ) : null}
                {typeof action.tensionDelta === "number" ? (
                  <span className="monitoring-pill">Tension {formatSigned(action.tensionDelta)}</span>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
