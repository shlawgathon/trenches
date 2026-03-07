import { startTransition, useEffect, useMemo, useState } from "react";

import { bootstrapPlatform } from "./app/bootstrap";
import { CommandMap } from "./components/CommandMap";
import { MonitoringDeck, type MonitoringAgentSnapshot } from "./components/MonitoringDeck";
import { buildViewerMapState, type MapSelection } from "./lib/viewer-map";
import type { PlatformRuntime } from "./lib/platform";
import type { AgentAction, AgentObservation, SessionState } from "./lib/types";

const AGENT_ORDER = ["us", "israel", "iran", "hezbollah", "gulf", "oversight"] as const;
const AGENT_MODEL_META: Record<
  (typeof AGENT_ORDER)[number],
  {
    family: string;
    variant: string;
    sizeLabel: string;
  }
> = {
  us: { family: "Doctrine Policy", variant: "CENTCOM theater lead", sizeLabel: "large" },
  israel: { family: "Doctrine Policy", variant: "IDF northern-front lead", sizeLabel: "medium-large" },
  iran: { family: "Doctrine Policy", variant: "IRGC asymmetric lead", sizeLabel: "medium-large" },
  hezbollah: { family: "Doctrine Policy", variant: "proxy pressure lead", sizeLabel: "medium" },
  gulf: { family: "Doctrine Policy", variant: "shipping stability lead", sizeLabel: "medium" },
  oversight: { family: "Meta Policy", variant: "risk and intervention lead", sizeLabel: "medium-large" },
};

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

function toAssets(observation: AgentObservation) {
  return observation.strategic_assets as Array<{
    name?: string;
    category?: string;
    status?: string;
    latitude?: number;
    longitude?: number;
    notes?: string;
  }>;
}

function getSourcePacketCounts(observation: AgentObservation | undefined) {
  const packets = observation?.source_packets ?? [];
  return {
    ok: packets.filter((packet) => packet.status === "ok").length,
    pending: packets.filter((packet) => packet.status === "pending").length,
    error: packets.filter((packet) => packet.status === "error").length,
    trainingCore: packets.filter((packet) => packet.delivery === "training_core").length,
    liveDemo: packets.filter((packet) => packet.delivery === "live_demo").length,
    total: packets.length,
  };
}

function toConfidence(value: number | undefined) {
  if (typeof value !== "number") {
    return 0.5;
  }
  return Math.max(0, Math.min(1, (value + 1) / 2));
}

export default function App() {
  const [runtime, setRuntime] = useState<PlatformRuntime | null>(null);
  const [session, setSession] = useState<SessionState | null>(null);
  const [selectedMapEntity, setSelectedMapEntity] = useState<MapSelection>("all");
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

  const viewerMapState = useMemo(() => {
    if (!session) {
      return null;
    }
    return buildViewerMapState(session);
  }, [session]);

  const commandMapEntities = useMemo(() => {
    return viewerMapState?.entities ?? [];
  }, [viewerMapState]);

  const commandMapFeatures = useMemo(() => {
    return viewerMapState?.features ?? [];
  }, [viewerMapState]);

  const commandMapLinks = useMemo(() => viewerMapState?.links ?? [], [viewerMapState]);

  const commandMapWorldSummary = useMemo(() => {
    if (!viewerMapState || !session) {
      return null;
    }
    return {
      ...viewerMapState.worldSummary,
      activeEventCount: session.world.active_events.length,
      lastUpdatedLabel: new Date(session.updated_at).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
  }, [session, viewerMapState]);

  const monitoringAgents = useMemo<MonitoringAgentSnapshot[]>(() => {
    if (!session) {
      return [];
    }

    const entityLookup = new Map(commandMapEntities.map((entity) => [entity.id, entity]));
    return AGENT_ORDER.map((agentId) => {
      const observation = session.observations[agentId];
      const reward = session.rewards[agentId];
      const profile = (observation?.entity_profile ?? {}) as Record<string, unknown>;
      const viewerEntity = entityLookup.get(agentId);
      const recentActions = [...session.recent_traces]
        .reverse()
        .filter((trace) => trace.actions[agentId])
        .slice(0, 8)
        .map((trace) => {
          const action = trace.actions[agentId];
          return {
            id: `${agentId}-${trace.turn}`,
            turn: trace.turn,
            createdAt: trace.created_at,
            actor: action.actor,
            type: action.type,
            summary: action.summary,
            target: action.target,
            rewardTotal: trace.rewards[agentId]?.total ?? null,
            tensionDelta: trace.tension_after - trace.tension_before,
            oversightTriggered: trace.oversight.triggered,
          };
        });

      return {
        id: agentId,
        displayName: typeof profile.display_name === "string" ? profile.display_name : agentId,
        color: viewerEntity?.color,
        accent: viewerEntity?.accent,
        doctrine:
          (profile.decision_doctrine as { escalation_bias?: string } | undefined)?.escalation_bias ??
          (profile.military_posture as { style?: string } | undefined)?.style,
        currentObjective: toLines(profile.strategic_objectives)[0] ?? "Objective pending",
        postureLabel: (profile.military_posture as { style?: string } | undefined)?.style ?? "Doctrinal posture",
        statusLabel: session.live.enabled ? "Streaming" : "Scenario",
        lastDecisionAt: recentActions[0]?.createdAt ?? session.updated_at,
        confidence: toConfidence(reward?.behavioral_consistency),
        model: {
          ...AGENT_MODEL_META[agentId],
          contextWindow: "128K",
          trainingStack: session.episode.algorithm_hints.post_training,
        },
        observation: {
          perceivedTension: observation?.perceived_tension ?? null,
          knownCoalitions: observation?.known_coalitions ?? [],
          publicBrief: (observation?.public_brief ?? []).map((item) => ({
            source: item.source,
            summary: item.summary,
            confidence: item.confidence,
          })),
          privateBrief: (observation?.private_brief ?? []).map((item) => ({
            source: item.source,
            summary: item.summary,
            confidence: item.confidence,
          })),
          strategicAssets: (observation ? toAssets(observation) : []).map((asset) => ({
            name: asset.name ?? "Unknown asset",
            category: asset.category,
            status: asset.status,
            latitude: asset.latitude,
            longitude: asset.longitude,
            notes: asset.notes,
          })),
          sourcePackets: (observation?.source_packets ?? []).map((packet) => ({
            sourceId: packet.source_id,
            sourceName: packet.source_name,
            delivery: packet.delivery,
            status: packet.status,
            kind: packet.kind,
            fetchedAt: packet.fetched_at ?? null,
          })),
          sourcePacketCounts: getSourcePacketCounts(observation),
          sourceBundle: observation?.source_bundle ?? [],
          trainingSourceBundle: observation?.training_source_bundle ?? [],
          liveSourceBundle: observation?.live_source_bundle ?? [],
        },
        reward: {
          total: reward?.total ?? 0,
          coalitionStability: reward?.coalition_stability,
          escalationPenalty: reward?.escalation_penalty,
          marketGain: reward?.market_gain,
          behavioralConsistency: reward?.behavioral_consistency,
          goalTerms: reward?.goal_terms ?? {},
        },
        recentActions,
      };
    });
  }, [commandMapEntities, session]);

  const selectedMonitoringAgent = selectedMapEntity === "all" ? null : selectedMapEntity;

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
          {commandMapWorldSummary ? (
            <CommandMap
              entities={commandMapEntities}
              features={commandMapFeatures}
              links={commandMapLinks}
              selectedEntity={selectedMapEntity}
              onSelectEntity={(entityId) => setSelectedMapEntity(entityId as MapSelection)}
              worldSummary={commandMapWorldSummary}
            />
          ) : null}

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
                {AGENT_ORDER.map((agentId) => (
                  <div key={agentId} className="list-row">
                    <strong>{agentId}</strong>
                    <span>{runtime.liveSourcePlans[agentId]?.length ?? 0} sources</span>
                  </div>
                ))}
              </div>
            </article>
          </section>
        </>
      ) : null}
    </div>
  );
}
