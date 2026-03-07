import type { MonitoringPanelSharedProps } from "./monitoring-types";
import { formatSigned, getFocusedAgent, resolveAgentTone, rewardEntries } from "./monitoring-utils";

function widthFor(value: number, maxValue: number) {
  if (maxValue <= 0) {
    return "0%";
  }
  return `${Math.max((Math.abs(value) / maxValue) * 100, 4)}%`;
}

export function RewardSummaryPanel({
  agents,
  selectedAgentId,
  title = "Reward Summary",
  eyebrow = "Training Feedback",
}: MonitoringPanelSharedProps) {
  const focused = getFocusedAgent(agents, selectedAgentId);
  const focusedTone = focused ? resolveAgentTone(focused) : "idle";
  const entries = focused ? rewardEntries(focused) : { baseline: [], doctrine: [] };
  const maxMagnitude = Math.max(
    0.1,
    ...entries.baseline.map((entry) => Math.abs(entry.value)),
    ...entries.doctrine.map((entry) => Math.abs(entry.value)),
  );

  return (
    <section className="monitoring-panel">
      <header className="monitoring-panel__header">
        <div>
          <p className="monitoring-panel__eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
          <p>Actor-specific reward surfaces, including shared baseline terms and doctrine-specific goal terms.</p>
        </div>
        {focused ? (
          <aside>
            <span className={`monitoring-tone-pill monitoring-tone--${focusedTone}`}>{focused.displayName}</span>
            <span className="monitoring-label">Comparative reward vector</span>
          </aside>
        ) : null}
      </header>

      {agents.length === 0 ? (
        <div className="monitoring-empty">No reward signals supplied.</div>
      ) : (
        <div className="monitoring-reward-layout">
          {focused ? (
            <div className="monitoring-reward-topline">
              <section className="monitoring-reward-score">
                <span className="monitoring-label">Focused Actor Total</span>
                <strong>{focused.reward.total.toFixed(2)}</strong>
                <p>{focused.currentObjective ?? focused.doctrine ?? "No doctrine note supplied."}</p>
              </section>

              <section className="monitoring-reward-compare">
                {agents.map((agent) => (
                  <div key={agent.id} className="monitoring-mini-card">
                    <span>{agent.displayName}</span>
                    <strong>{agent.reward.total.toFixed(2)}</strong>
                  </div>
                ))}
              </section>
            </div>
          ) : null}

          {entries.baseline.length > 0 ? (
            <section className="monitoring-reward-grid">
              <p className="monitoring-panel__eyebrow">Shared Terms</p>
              {entries.baseline.map((entry) => (
                <div key={entry.key} className="monitoring-reward-bar">
                  <span>{entry.label}</span>
                  <div className="monitoring-bar-track">
                    <div
                      className={`monitoring-bar-fill ${entry.value >= 0 ? "is-positive" : "is-negative"}`}
                      style={{ width: widthFor(entry.value, maxMagnitude) }}
                    />
                  </div>
                  <strong>{formatSigned(entry.value)}</strong>
                </div>
              ))}
            </section>
          ) : null}

          {entries.doctrine.length > 0 ? (
            <section className="monitoring-reward-grid">
              <p className="monitoring-panel__eyebrow">Doctrine Terms</p>
              {entries.doctrine.map((entry) => (
                <div key={entry.key} className="monitoring-reward-bar">
                  <span>{entry.label}</span>
                  <div className="monitoring-bar-track">
                    <div
                      className={`monitoring-bar-fill ${entry.value >= 0 ? "is-positive" : "is-negative"}`}
                      style={{ width: widthFor(entry.value, maxMagnitude) }}
                    />
                  </div>
                  <strong>{formatSigned(entry.value)}</strong>
                </div>
              ))}
            </section>
          ) : null}
        </div>
      )}
    </section>
  );
}
