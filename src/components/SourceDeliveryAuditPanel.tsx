import "./monitoring.css";

import type { AgentSourceMonitor, SourceMonitorReport } from "../lib/types";

function toTone(status: AgentSourceMonitor["status"]) {
  switch (status) {
    case "healthy":
      return "stable";
    case "degraded":
      return "watch";
    default:
      return "critical";
  }
}

type Props = {
  report: SourceMonitorReport | null;
  selectedAgentId?: string | null;
  loading?: boolean;
  error?: string | null;
};

export function SourceDeliveryAuditPanel({ report, selectedAgentId, loading = false, error }: Props) {
  const focused =
    report?.agents.find((agent) => agent.agent_id === selectedAgentId) ??
    report?.agents[0] ??
    null;

  return (
    <section className="monitoring-panel">
      <header className="monitoring-panel__header">
        <div>
          <p className="monitoring-panel__eyebrow">Delivery Audit</p>
          <h3>Source-to-Model Verifier</h3>
          <p>Checks that each entity has configured sources, active packets, and source-derived brief items reaching the model.</p>
        </div>
        {report ? (
          <aside>
            <span className={`monitoring-tone-pill monitoring-tone--${report.summary.blocked_agents > 0 ? "critical" : report.summary.degraded_agents > 0 ? "watch" : "stable"}`}>
              {report.summary.blocked_agents > 0
                ? `${report.summary.blocked_agents} blocked`
                : report.summary.degraded_agents > 0
                  ? `${report.summary.degraded_agents} degraded`
                  : "All paths ready"}
            </span>
            <span className="monitoring-label">
              {report.summary.delivered_source_brief_count} source briefs delivered
            </span>
          </aside>
        ) : null}
      </header>

      {loading && !report ? <div className="monitoring-empty">Loading source-delivery audit.</div> : null}
      {error ? <div className="monitoring-empty">{error}</div> : null}
      {!loading && !error && !report ? <div className="monitoring-empty">No source-delivery audit available.</div> : null}

      {report ? (
        <div className="monitoring-audit-layout">
          <section className="monitoring-mini-grid">
            <div className="monitoring-mini-card">
              <span>Healthy Agents</span>
              <strong>{report.summary.healthy_agents}</strong>
            </div>
            <div className="monitoring-mini-card">
              <span>Degraded Agents</span>
              <strong>{report.summary.degraded_agents}</strong>
            </div>
            <div className="monitoring-mini-card">
              <span>Blocked Agents</span>
              <strong>{report.summary.blocked_agents}</strong>
            </div>
            <div className="monitoring-mini-card">
              <span>Active Sources</span>
              <strong>{report.summary.active_source_count}</strong>
            </div>
          </section>

          <section className="monitoring-audit-table">
            <div className="monitoring-audit-head">
              <span>Agent</span>
              <span>Configured</span>
              <span>Active</span>
              <span>Briefs</span>
              <span>Issues</span>
            </div>
            {report.agents.map((agent) => {
              const tone = toTone(agent.status);
              return (
                <div
                  key={agent.agent_id}
                  className={`monitoring-audit-row ${focused?.agent_id === agent.agent_id ? "is-focused" : ""}`}
                >
                  <div>
                    <strong>{agent.display_name}</strong>
                    <p className="monitoring-copy">
                      <span className={`monitoring-tone-pill monitoring-tone--${tone}`}>{agent.status}</span>
                    </p>
                  </div>
                  <strong>
                    {agent.configured_training_sources}
                    {agent.configured_live_sources > 0 ? ` / ${agent.configured_live_sources}` : ""}
                  </strong>
                  <strong>{agent.active_source_count}</strong>
                  <strong>
                    {agent.delivered_training_brief_count}
                    {report.live_enabled ? ` / ${agent.delivered_live_brief_count}` : ""}
                  </strong>
                  <strong>{agent.issues.length}</strong>
                </div>
              );
            })}
          </section>

          {focused ? (
            <section className="monitoring-audit-focus">
              <div className="monitoring-source-topline">
                <section className="monitoring-source-card">
                  <span className="monitoring-label">Focused Actor</span>
                  <strong>{focused.display_name}</strong>
                  <p>
                    {focused.ok_packet_count} ok, {focused.pending_packet_count} pending, {focused.error_packet_count} error
                    packets across {focused.active_source_count} active feeds.
                  </p>
                </section>
                <section className="monitoring-source-card">
                  <span className="monitoring-label">Model Feed</span>
                  <strong>
                    {focused.delivered_training_brief_count}
                    {report.live_enabled ? ` / ${focused.delivered_live_brief_count}` : ""}
                  </strong>
                  <p>
                    Training/live source snippets currently reaching the model-facing private brief.
                  </p>
                </section>
              </div>

              <section>
                <p className="monitoring-panel__eyebrow">Delivered Source Names</p>
                {focused.delivered_source_names.length > 0 ? (
                  <div className="monitoring-tags">
                    {focused.delivered_source_names.map((source) => (
                      <span key={source} className="monitoring-tag">
                        {source}
                      </span>
                    ))}
                  </div>
                ) : (
                  <div className="monitoring-empty">No source-derived brief items are currently visible to the model.</div>
                )}
              </section>

              <section>
                <p className="monitoring-panel__eyebrow">Issues</p>
                {focused.issues.length > 0 ? (
                  <div className="monitoring-issue-list">
                    {focused.issues.map((issue, index) => (
                      <article
                        key={`${focused.agent_id}-${index}`}
                        className={`monitoring-issue monitoring-issue--${issue.severity}`}
                      >
                        <strong>{issue.severity}</strong>
                        <p>{issue.message}</p>
                      </article>
                    ))}
                  </div>
                ) : (
                  <div className="monitoring-empty">Manifest, packet coverage, and model delivery are aligned for this actor.</div>
                )}
              </section>
            </section>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
