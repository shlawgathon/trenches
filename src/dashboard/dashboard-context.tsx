"use client";

import {
  createContext,
  startTransition,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { bootstrapPlatform } from "../app/bootstrap";
import { type MonitoringAgentSnapshot } from "../components/MonitoringDeck";
import { buildViewerMapState, type MapSelection } from "../lib/viewer-map";
import type { PlatformRuntime } from "../lib/platform";
import type { AgentAction, AgentObservation, SessionState } from "../lib/types";

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

type DashboardContextValue = {
  runtime: PlatformRuntime | null;
  session: SessionState | null;
  error: string | null;
  busy: boolean;
  summary: Array<{ label: string; value: string }> | null;
  selectedMapEntity: MapSelection;
  selectedMonitoringAgent: string | null;
  commandMapEntities: ReturnType<typeof buildViewerMapState>["entities"];
  commandMapFeatures: ReturnType<typeof buildViewerMapState>["features"];
  commandMapLinks: ReturnType<typeof buildViewerMapState>["links"];
  commandMapWorldSummary: {
    turn: number;
    tension: number;
    marketStress: number;
    oilPressure: number;
    liveMode: boolean;
    activeEventCount?: number;
    lastUpdatedLabel?: string;
  };
  monitoringAgents: MonitoringAgentSnapshot[];
  isInitializingSession: boolean;
  isBackendUnavailable: boolean;
  setSelectedMapEntity: (entityId: MapSelection) => void;
  createFreshSession: () => Promise<void>;
  toggleLive: (enabled: boolean) => Promise<void>;
  stepSession: () => Promise<void>;
  refreshSources: () => Promise<void>;
};

const DashboardContext = createContext<DashboardContextValue | null>(null);

export function DashboardProvider({ children }: { children: ReactNode }) {
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
      return {
        turn: 0,
        tension: 0,
        marketStress: 0,
        oilPressure: 0,
        liveMode: false,
        activeEventCount: 0,
        lastUpdatedLabel: runtime
          ? new Date(runtime.bootedAt).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })
          : "Awaiting sync",
      };
    }
    return {
      ...viewerMapState.worldSummary,
      activeEventCount: session.world.active_events.length,
      lastUpdatedLabel: new Date(session.updated_at).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
  }, [runtime, session, viewerMapState]);

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
  const isInitializingSession = runtime?.backendStatus === "healthy" && !session && !error;
  const isBackendUnavailable = runtime?.backendStatus === "unreachable";

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

  const value = useMemo<DashboardContextValue>(
    () => ({
      runtime,
      session,
      error,
      busy,
      summary,
      selectedMapEntity,
      selectedMonitoringAgent,
      commandMapEntities,
      commandMapFeatures,
      commandMapLinks,
      commandMapWorldSummary,
      monitoringAgents,
      isInitializingSession,
      isBackendUnavailable,
      setSelectedMapEntity,
      createFreshSession,
      toggleLive,
      stepSession,
      refreshSources,
    }),
    [
      runtime,
      session,
      error,
      busy,
      summary,
      selectedMapEntity,
      selectedMonitoringAgent,
      commandMapEntities,
      commandMapFeatures,
      commandMapLinks,
      commandMapWorldSummary,
      monitoringAgents,
      isInitializingSession,
      isBackendUnavailable,
    ],
  );

  return <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>;
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error("useDashboard must be used within a DashboardProvider.");
  }
  return context;
}
