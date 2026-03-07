import "./monitoring.css";

import type { MonitoringPanelSharedProps } from "./monitoring-types";
import { formatTimeLabel, getFocusedAgent, resolveAgentTone, summarizeSourceHealth } from "./monitoring-utils";

export function SourceHealthPanel({
  agents,
  selectedAgentId,
  title = "Source Health",
  eyebrow = "Intel Ingest",
}: MonitoringPanelSharedProps) {
  const focused = getFocusedAgent(agents, selectedAgentId);
  const focusedHealth = focused ? summarizeSourceHealth(focused.observation) : null;
  const focusedTone = focused ? resolveAgentTone(focused) : "idle";
  const sourceNames = focused
    ? [...(focused.observation.trainingSourceBundle ?? []), ...(focused.observation.liveSourceBundle ?? [])].slice(0, 8)
    : [];

  return (
    <section className="monitoring-panel">
      <header className="monitoring-panel__header">
        <div>
          <p className="monitoring-panel__eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
          <p>Packet delivery, training/live split, and sync freshness for the selected actor and the wider stack.</p>
        </div>
        {focused ? (
          <aside>
            <span className={`monitoring-tone-pill monitoring-tone--${focusedTone}`}>{focused.displayName}</span>
            <span className="monitoring-label">{formatTimeLabel(focusedHealth?.lastFetchedAt)}</span>
          </aside>
        ) : null}
      </header>

      {agents.length === 0 ? (
        <div className="monitoring-empty">No source packets supplied.</div>
      ) : (
        <div className="monitoring-source-layout">
          {focused && focusedHealth ? (
            <div className="monitoring-source-topline">
              <section className="monitoring-source-card">
                <span className="monitoring-label">Focused Actor</span>
                <strong>{focusedHealth.total}</strong>
                <p>
                  {focused.displayName} is reading {focusedHealth.trainingCore} training-core feeds and {focusedHealth.liveDemo}{" "}
                  live/demo streams.
                </p>
              </section>

              <section className="monitoring-source-card">
                <span className="monitoring-label">Delivery Split</span>
                <div className="monitoring-source-health-grid">
                  <div className="monitoring-mini-card">
                    <span>Healthy</span>
                    <strong>{focusedHealth.ok}</strong>
                  </div>
                  <div className="monitoring-mini-card">
                    <span>Pending</span>
                    <strong>{focusedHealth.pending}</strong>
                  </div>
                  <div className="monitoring-mini-card">
                    <span>Error</span>
                    <strong>{focusedHealth.error}</strong>
                  </div>
                  <div className="monitoring-mini-card">
                    <span>Last Sync</span>
                    <strong>{formatTimeLabel(focusedHealth.lastFetchedAt)}</strong>
                  </div>
                </div>
              </section>
            </div>
          ) : null}

          {focusedHealth ? (
            <div className="monitoring-source-bar">
              <span>Focused Packet Integrity</span>
              <div className="monitoring-source-track" aria-hidden="true">
                <div
                  className="monitoring-source-segment monitoring-source-segment--ok"
                  style={{ width: `${focusedHealth.total > 0 ? (focusedHealth.ok / focusedHealth.total) * 100 : 0}%` }}
                />
                <div
                  className="monitoring-source-segment monitoring-source-segment--pending"
                  style={{
                    width: `${focusedHealth.total > 0 ? (focusedHealth.pending / focusedHealth.total) * 100 : 0}%`,
                  }}
                />
                <div
                  className="monitoring-source-segment monitoring-source-segment--error"
                  style={{
                    width: `${focusedHealth.total > 0 ? (focusedHealth.error / focusedHealth.total) * 100 : 0}%`,
                  }}
                />
              </div>
            </div>
          ) : null}

          <section className="monitoring-source-table">
            <div className="monitoring-table-head">
              <span>Agent</span>
              <span>Ok</span>
              <span>Pending</span>
              <span>Error</span>
              <span>Total</span>
            </div>
            {agents.map((agent) => {
              const health = summarizeSourceHealth(agent.observation);
              return (
                <div
                  key={agent.id}
                  className={`monitoring-source-row ${focused?.id === agent.id ? "is-focused" : ""}`}
                >
                  <div>
                    <strong>{agent.displayName}</strong>
                    <p className="monitoring-copy">
                      {health.trainingCore} training / {health.liveDemo} live
                    </p>
                  </div>
                  <strong>{health.ok}</strong>
                  <strong>{health.pending}</strong>
                  <strong>{health.error}</strong>
                  <strong>{health.total}</strong>
                </div>
              );
            })}
          </section>

          <section>
            <p className="monitoring-panel__eyebrow">Focused Feed Set</p>
            {sourceNames.length > 0 ? (
              <ul className="monitoring-source-list">
                {sourceNames.map((source) => (
                  <li key={source}>{source}</li>
                ))}
              </ul>
            ) : (
              <div className="monitoring-empty">No bundle labels supplied.</div>
            )}
          </section>
        </div>
      )}
    </section>
  );
}
