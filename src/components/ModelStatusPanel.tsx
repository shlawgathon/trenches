import type { CSSProperties } from "react";

import type { MonitoringAgentSnapshot, MonitoringPanelSharedProps } from "./monitoring-types";
import {
  formatMaybe,
  formatPercent,
  formatTimeLabel,
  getFocusedAgent,
  getLatestAction,
  getModelLabel,
  getToneLabel,
  resolveAgentTone,
  summarizeSourceHealth,
  topIntelSnippets,
} from "./monitoring-utils";

type CardProps = {
  agent: MonitoringAgentSnapshot;
  selected: boolean;
  onSelectAgent?: (agentId: string) => void;
};

function AgentStatusCard({ agent, selected, onSelectAgent }: CardProps) {
  const tone = resolveAgentTone(agent);
  const health = summarizeSourceHealth(agent.observation);
  const latestAction = getLatestAction(agent);
  const style = {
    "--monitoring-accent": agent.accent ?? agent.color ?? "#88f2c3",
  } as CSSProperties;

  const content = (
    <>
      <header className="monitoring-agent-card__header">
        <div>
          <h4>{agent.displayName}</h4>
          <p>{agent.postureLabel ?? agent.currentObjective ?? "Operational posture unspecified."}</p>
        </div>
        <span className={`monitoring-tone-pill monitoring-tone--${tone}`}>
          {agent.statusLabel ?? getToneLabel(tone)}
        </span>
      </header>

      <div className="monitoring-row">
        <span className="monitoring-model-label">{getModelLabel(agent)}</span>
        <span className="monitoring-label">{formatTimeLabel(agent.lastDecisionAt)}</span>
      </div>

      <div className="monitoring-stats-row">
        <div className="monitoring-mini-card">
          <span>Perceived Tension</span>
          <strong>{formatMaybe(agent.observation.perceivedTension)}</strong>
        </div>
        <div className="monitoring-mini-card">
          <span>Confidence</span>
          <strong>{formatPercent(agent.confidence)}</strong>
        </div>
        <div className="monitoring-mini-card">
          <span>Assets</span>
          <strong>{agent.observation.strategicAssets?.length ?? 0}</strong>
        </div>
        <div className="monitoring-mini-card">
          <span>Tools</span>
          <strong>{agent.observation.tools?.length ?? 0}</strong>
        </div>
      </div>

      <div className="monitoring-source-bar">
        <span>Source Health</span>
        <div className="monitoring-source-track" aria-hidden="true">
          <div
            className="monitoring-source-segment monitoring-source-segment--ok"
            style={{ width: `${health.total > 0 ? (health.ok / health.total) * 100 : 0}%` }}
          />
          <div
            className="monitoring-source-segment monitoring-source-segment--pending"
            style={{ width: `${health.total > 0 ? (health.pending / health.total) * 100 : 0}%` }}
          />
          <div
            className="monitoring-source-segment monitoring-source-segment--error"
            style={{ width: `${health.total > 0 ? (health.error / health.total) * 100 : 0}%` }}
          />
        </div>
      </div>

      <div className="monitoring-copy">
        {latestAction ? (
          <>
            <strong>{latestAction.type}</strong> {latestAction.summary}
          </>
        ) : (
          "No recent action trace."
        )}
      </div>
    </>
  );

  if (onSelectAgent) {
    return (
      <button
        type="button"
        className={`monitoring-agent-card monitoring-agent-card--button ${selected ? "is-selected" : ""}`}
        style={style}
        onClick={() => onSelectAgent(agent.id)}
      >
        {content}
      </button>
    );
  }

  return (
    <article className={`monitoring-agent-card ${selected ? "is-selected" : ""}`} style={style}>
      {content}
    </article>
  );
}

export function ModelStatusPanel({
  agents,
  selectedAgentId,
  onSelectAgent,
  title = "Model Status Matrix",
  eyebrow = "Agent Runtime",
}: MonitoringPanelSharedProps) {
  const focused = getFocusedAgent(agents, selectedAgentId);
  const focusedTone = focused ? resolveAgentTone(focused) : "idle";
  const intel = focused ? topIntelSnippets(focused) : [];

  return (
    <section className="monitoring-panel">
      <header className="monitoring-panel__header">
        <div>
          <p className="monitoring-panel__eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
          <p>Operator-facing readout of each model's posture, confidence, source integrity, and active objective.</p>
        </div>
        {focused ? (
          <aside>
            <span className={`monitoring-tone-pill monitoring-tone--${focusedTone}`}>
              {focused.statusLabel ?? getToneLabel(focusedTone)}
            </span>
            <span className="monitoring-label">{focused.displayName}</span>
          </aside>
        ) : null}
      </header>

      {agents.length === 0 ? (
        <div className="monitoring-empty">No model snapshots supplied.</div>
      ) : (
        <div className="monitoring-status-layout">
          <div className="monitoring-status-grid">
            {agents.map((agent) => (
              <AgentStatusCard
                key={agent.id}
                agent={agent}
                selected={focused?.id === agent.id}
                onSelectAgent={onSelectAgent}
              />
            ))}
          </div>

          {focused ? (
            <aside className="monitoring-status-focus">
              <section className="monitoring-focus-hero">
                <span className="monitoring-chip">{getModelLabel(focused)}</span>
                <h4>{focused.displayName}</h4>
                <p>{focused.doctrine ?? focused.currentObjective ?? "No doctrine summary supplied."}</p>
                <div className="monitoring-focus-hero__meta">
                  <span className="monitoring-tag">Objective: {focused.currentObjective ?? "Unspecified"}</span>
                  <span className="monitoring-tag">
                    Coalition Links: {focused.observation.knownCoalitions?.length ?? 0}
                  </span>
                  <span className="monitoring-tag">
                    Assets: {focused.observation.strategicAssets?.length ?? 0}
                  </span>
                </div>
              </section>

              <section className="monitoring-mini-grid">
                <div className="monitoring-mini-card">
                  <span>Reward Total</span>
                  <strong>{focused.reward.total.toFixed(2)}</strong>
                </div>
                <div className="monitoring-mini-card">
                  <span>Last Decision</span>
                  <strong>{formatTimeLabel(focused.lastDecisionAt)}</strong>
                </div>
              </section>

              <section>
                <p className="monitoring-panel__eyebrow">Operational Tools</p>
                <div className="monitoring-tags">
                  {(focused.observation.tools ?? []).slice(0, 8).map((tool) => (
                    <span key={tool.id} className="monitoring-tag">
                      {tool.label}
                    </span>
                  ))}
                  {(focused.observation.tools ?? []).length === 0 ? (
                    <span className="monitoring-empty">No tool inventory supplied.</span>
                  ) : null}
                </div>
              </section>

              <section>
                <p className="monitoring-panel__eyebrow">Top Intel</p>
                <div className="monitoring-intel-list">
                  {intel.map((item, index) => (
                    <article key={`${item.source}-${index}`} className="monitoring-intel-card">
                      <strong>{item.source}</strong>
                      <p>{item.summary}</p>
                    </article>
                  ))}
                  {intel.length === 0 ? <div className="monitoring-empty">No intel snippets supplied.</div> : null}
                </div>
              </section>
            </aside>
          ) : null}
        </div>
      )}
    </section>
  );
}
