"use client";

import { useDashboard } from "./dashboard-context";

export function WorldPage() {
  const { runtime, session } = useDashboard();

  if (!session || !runtime) {
    return (
      <section className="panel dashboard-empty-state">
        <div className="panel-header">
          <h2>World State Trace</h2>
          <span>Waiting for runtime session</span>
        </div>
        <p>The world trace and source plan matrix unlock as soon as the backend produces a live session snapshot.</p>
      </section>
    );
  }

  return (
    <section className="panel-grid">
      <article className="panel panel-wide">
        <div className="panel-header">
          <h2>World State Trace</h2>
          <span>{session.live.enabled ? "Live intelligence sync armed" : "Scenario-only state"}</span>
        </div>
        <div className="world-stats">
          <div>
            <label>Coalition Graph</label>
            <pre>{JSON.stringify(session.world.coalition_graph, null, 2)}</pre>
          </div>
          <div>
            <label>Recent Traces</label>
            <pre>{JSON.stringify(session.recent_traces.slice(-5), null, 2)}</pre>
          </div>
        </div>
      </article>

      <article className="panel">
        <div className="panel-header">
          <h2>Source Plan Matrix</h2>
          <span>Non-shared source stacks</span>
        </div>
        <div className="list-block">
          {(["us", "israel", "iran", "hezbollah", "gulf", "oversight"] as const).map((agentId) => (
            <div key={agentId} className="list-row">
              <strong>{agentId}</strong>
              <span>{runtime.liveSourcePlans[agentId]?.length ?? 0} sources</span>
            </div>
          ))}
        </div>
      </article>
    </section>
  );
}
