import { startTransition, useEffect, useMemo, useState } from "react";

import { bootstrapPlatform } from "./app/bootstrap";
import type { PlatformRuntime } from "./lib/platform";
import type { AgentAction, AgentObservation, SessionState } from "./lib/types";

const AGENT_ORDER = ["us", "israel", "iran", "hezbollah", "gulf", "oversight"] as const;

function buildDefaultActions(): Record<string, AgentAction> {
  return {
    us: { actor: "us", type: "negotiate", summary: "Probe for deconfliction with Gulf partners.", target: "gulf" },
    israel: { actor: "israel", type: "defend", summary: "Raise air-defense readiness on the northern front." },
    iran: { actor: "iran", type: "intel_query", summary: "Query proxy and strike-damage reporting." },
    hezbollah: { actor: "hezbollah", type: "hold", summary: "Hold position and reassess launch timing." },
    gulf: { actor: "gulf", type: "intel_query", summary: "Poll shipping, oil, and port disruption risk." },
    oversight: { actor: "oversight", type: "oversight_review", summary: "Assess drift and escalation probability." },
  };
}

function toLines(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item));
  }
  return [];
}

function toAssets(observation: AgentObservation): Array<{ name?: string; category?: string; notes?: string }> {
  return observation.strategic_assets as Array<{ name?: string; category?: string; notes?: string }>;
}

function getTrainingSourceBundle(observation: AgentObservation | undefined): string[] {
  if (!observation) {
    return [];
  }
  if (observation.training_source_bundle.length > 0) {
    return observation.training_source_bundle;
  }
  return observation.source_bundle;
}

function getLiveSourceBundle(observation: AgentObservation | undefined): string[] {
  return observation?.live_source_bundle ?? [];
}

function getCollectedTrainingPackets(observation: AgentObservation | undefined) {
  return observation?.training_source_packets ?? [];
}

export default function App() {
  const [runtime, setRuntime] = useState<PlatformRuntime | null>(null);
  const [session, setSession] = useState<SessionState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function initialize() {
      try {
        const nextRuntime = await bootstrapPlatform();
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setRuntime(nextRuntime);
        });

        if (nextRuntime.backendStatus === "healthy") {
          const nextSession = await nextRuntime.sessionClient.createSession({ seed: 7 });
          if (!cancelled) {
            startTransition(() => {
              setSession(nextSession);
            });
          }
        }
      } catch (nextError) {
        if (!cancelled) {
          setError(nextError instanceof Error ? nextError.message : "Failed to bootstrap runtime.");
        }
      }
    }

    void initialize();
    return () => {
      cancelled = true;
    };
  }, []);

  const summary = useMemo(() => {
    if (!session) {
      return null;
    }
    return [
      { label: "Turn", value: String(session.world.turn) },
      { label: "Tension", value: session.world.tension_level.toFixed(1) },
      { label: "Market Stress", value: session.world.market_stress.toFixed(1) },
      { label: "Oil Pressure", value: session.world.oil_pressure.toFixed(1) },
      { label: "Live", value: session.live.enabled ? "On" : "Off" },
      { label: "Backend", value: runtime?.backendStatus ?? "unknown" },
    ];
  }, [runtime?.backendStatus, session]);

  async function createFreshSession() {
    if (!runtime) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const nextSession = await runtime.sessionClient.createSession({ seed: 7 });
      setSession(nextSession);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to create session.");
    } finally {
      setBusy(false);
    }
  }

  async function toggleLive(enabled: boolean) {
    if (!runtime || !session) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const nextSession = await runtime.sessionClient.setLiveMode(session.session_id, {
        enabled,
        auto_step: false,
        poll_interval_ms: 15000,
      });
      setSession(nextSession);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to update live mode.");
    } finally {
      setBusy(false);
    }
  }

  async function stepSession() {
    if (!runtime || !session) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await runtime.sessionClient.stepSession(session.session_id, {
        actions: buildDefaultActions(),
        external_signals: session.live.enabled
          ? [
              {
                source: "live-monitor",
                headline: "Shipping risk increases near Hormuz after fresh source bundle pull.",
                region: "gulf",
                tags: ["shipping", "oil", "hormuz"],
                severity: 0.4,
              },
            ]
          : [],
      });
      setSession(result.session);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to step session.");
    } finally {
      setBusy(false);
    }
  }

  async function refreshSources() {
    if (!runtime || !session) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const nextSession = await runtime.sessionClient.refreshSources(session.session_id);
      setSession(nextSession);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to refresh sources.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Trenches Monitoring Deck</p>
          <h1>Fog-of-War Live Session Dashboard</h1>
          <p className="lede">
            Monitor all six agents, inspect what each model sees, and control whether the live post-training session is
            running.
          </p>
        </div>
        <div className="status-card">
          <span>Runtime Booted</span>
          <strong>{runtime?.bootedAt ?? "pending"}</strong>
          <span>Source Validation</span>
          <strong>{runtime ? `${runtime.sourceValidation.duplicateKeys.length} duplicate keys` : "pending"}</strong>
        </div>
      </header>

      <section className="control-strip">
        <button onClick={createFreshSession} disabled={busy}>
          New Session
        </button>
        <button onClick={() => toggleLive(true)} disabled={busy || !session}>
          Start Live RL Session
        </button>
        <button onClick={() => toggleLive(false)} disabled={busy || !session}>
          Stop Live RL Session
        </button>
        <button onClick={refreshSources} disabled={busy || !session}>
          Refresh Sources
        </button>
        <button onClick={stepSession} disabled={busy || !session}>
          Advance Turn
        </button>
      </section>

      {error ? <section className="banner error">{error}</section> : null}
      {!session ? <section className="banner">No backend session yet. Create one or start the backend.</section> : null}

      {summary ? (
        <section className="summary-grid">
          {summary.map((item) => (
            <article key={item.label} className="metric">
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </article>
          ))}
        </section>
      ) : null}

      {session && runtime ? (
        <>
          <section className="panel-grid">
            <article className="panel panel-wide">
              <div className="panel-header">
                <h2>Global Monitor</h2>
                <span>{session.live.enabled ? "Live ingest armed" : "Static session"}</span>
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
                <h2>Live Source Plans</h2>
                <span>Agent-specific and non-shared</span>
              </div>
              <div className="list-block">
                {AGENT_ORDER.map((agentId) => (
                  <div key={agentId} className="list-row">
                    <strong>{agentId}</strong>
                    <span>{runtime.liveSourcePlans[agentId]?.length ?? 0} sources</span>
                  </div>
                ))}
              </div>
            </article>
          </section>

          <section className="agents-grid">
            {AGENT_ORDER.map((agentId) => {
              const observation = session.observations[agentId];
              const profile = observation?.entity_profile as Record<string, unknown> | undefined;
              const objectives = toLines(profile?.strategic_objectives);
              const assets = observation ? toAssets(observation) : [];
              const trainingSources = getTrainingSourceBundle(observation);
              const liveSources = getLiveSourceBundle(observation);
              const collectedPackets = getCollectedTrainingPackets(observation);

              return (
                <article key={agentId} className="agent-card">
                  <div className="panel-header">
                    <h2>{profile?.display_name ? String(profile.display_name) : agentId}</h2>
                    <span>{observation?.perceived_tension?.toFixed(1) ?? "0.0"} perceived tension</span>
                  </div>

                  <div className="agent-block">
                    <label>Training Core Sources</label>
                    <ul>
                      {trainingSources.map((source) => (
                        <li key={source}>{source}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="agent-block">
                    <label>Live / Demo Sources</label>
                    <ul>
                      {liveSources.length > 0 ? (
                        liveSources.map((source) => <li key={source}>{source}</li>)
                      ) : (
                        <li>None configured</li>
                      )}
                    </ul>
                  </div>

                  <div className="agent-block">
                    <label>Strategic Objectives</label>
                    <ul>
                      {objectives.slice(0, 4).map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="agent-block">
                    <label>Strategic Assets</label>
                    <ul>
                      {assets.slice(0, 5).map((asset) => (
                        <li key={`${asset.name}-${asset.category}`}>
                          {asset.name ?? "Unknown"} {asset.category ? `(${asset.category})` : ""}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="agent-block">
                    <label>Private Brief</label>
                    <ul>
                      {(observation?.private_brief ?? []).map((brief) => (
                        <li key={`${brief.source}-${brief.summary}`}>{brief.summary}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="agent-block">
                    <label>Collected Source Snapshots</label>
                    <ul>
                      {collectedPackets.slice(0, 3).map((packet) => (
                        <li key={packet.source_id}>
                          {packet.source_name}: {packet.summary || packet.status}
                        </li>
                      ))}
                    </ul>
                  </div>
                </article>
              );
            })}
          </section>
        </>
      ) : null}
    </div>
  );
}
