"use client";

import type { MapSelection } from "../lib/viewer-map";
import { MonitoringDeck } from "../components/MonitoringDeck";

import { useDashboard } from "./dashboard-context";

export function MonitoringPage() {
  const { monitoringAgents, selectedMonitoringAgent, session, setSelectedMapEntity } = useDashboard();

  if (!session) {
    return (
      <section className="panel dashboard-empty-state">
        <div className="panel-header">
          <h2>Model Supervision Deck</h2>
          <span>Waiting for session telemetry</span>
        </div>
        <p>
          Monitoring cards appear after the backend session is created. The navigation stays separate so the operator
          always knows where model-health details live once the run is active.
        </p>
      </section>
    );
  }

  return (
    <MonitoringDeck
      agents={monitoringAgents}
      selectedAgentId={selectedMonitoringAgent}
      onSelectAgent={(agentId) => setSelectedMapEntity(agentId as MapSelection)}
      title="Model Supervision Deck"
      eyebrow="Training and Inference Oversight"
      headline="Black-box monitoring surface for reward pressure, source integrity, recent actions, and actor posture across the full regional simulation."
      summary="This layer is for the human operator only. It sits beside the map and explains how each doctrine-specific model is reading the environment and being rewarded."
      statusChip={session.live.enabled ? "Live post-training session" : "Scenario replay session"}
    />
  );
}
