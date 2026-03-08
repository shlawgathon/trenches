// @refresh reset
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import gsap from "gsap";
import {
  Activity,
  Boxes,
  BrainCircuit,
  ChevronDown,
  Clock3,
  Columns2,
  Droplets,
  Gauge,
  History,
  ListFilter,
  LucideIcon,
  Maximize,
  MessageSquare,
  Radar,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Signal,
  TrendingUp,
  Users,
  Workflow,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import mapboxgl from "mapbox-gl";
import { NewsFeed } from "@/src/components/NewsFeed";
import { ActivityLog } from "@/src/components/ActivityLog";
import { ChatPanel, type SyntheticEvent } from "@/src/components/ChatPanel";
import { EventTimeline, type TimelineInteractionFocus } from "@/src/components/EventTimeline";

import { cn } from "@/src/lib/utils";
import { GlowingEffect } from "@/src/components/ui/glowing-effect";
import type { AgentAction, EntityModelBinding, ExternalSignal, SessionState } from "@/src/lib/types";
import { bootstrapPlatform } from "@/src/app/bootstrap";
import { getMapboxToken } from "@/src/lib/env";

const MAP_STYLE = "mapbox://styles/mapbox/dark-v11";
const DEFAULT_CENTER: [number, number] = [41.8, 27.8];
const DEFAULT_ZOOM = 1.8;
// Side panel widths: NewsFeed left (16px margin + 340/48px) · ActivityLog right (16px margin + 360/48px)
const LEFT_PANEL_OPEN_W = 356;   // 16 + 340
const LEFT_PANEL_COLLAPSED_W = 64; // 16 + 48
const RIGHT_PANEL_OPEN_W = 376;  // 16 + 360
const RIGHT_PANEL_COLLAPSED_W = 64; // 16 + 48
const TOP_BAR_GAP = 6;          // gap between panel edge and top bar
const TOP_BAR_MAX_WIDTH = 1600;
const TOP_BAR_MIN_WIDTH = 420;
const TOP_BAR_COMPACT_BREAKPOINT = 900;


const INTEL_HIDDEN_LAYERS = [
  "road-label",
  "settlement-label",
  "settlement-subdivision-label",
  "airport-label",
  "poi-label",
  "transit-label",
  "natural-label",
  "road-primary",
  "road-secondary-tertiary",
];

const AGENT_META: Record<string, { defaultLabel: string; color: string; secondaryColor: string; flag: string }> = {
  us: { defaultLabel: "United States", color: "#b22234", secondaryColor: "#3c3b6e", flag: "🇺🇸" },
  israel: { defaultLabel: "Israel", color: "#005eb8", secondaryColor: "#ffffff", flag: "🇮🇱" },
  iran: { defaultLabel: "Iran", color: "#239f40", secondaryColor: "#da0000", flag: "🇮🇷" },
  hezbollah: { defaultLabel: "Hezbollah", color: "#f4c542", secondaryColor: "#1f7a1f", flag: "🇱🇧" },
  gulf: { defaultLabel: "Gulf", color: "#00732f", secondaryColor: "#ce1126", flag: "🇦🇪" },
  oversight: { defaultLabel: "Oversight", color: "#1f4e79", secondaryColor: "#f4c542", flag: "🏳️‍⚖️" },
};
const MAP_NODE_IDS = Object.keys(AGENT_META).filter((id) => id !== "oversight");


const ACTION_HEAT: Record<AgentAction["type"], number> = {
  hold: 2,
  negotiate: 8,
  sanction: 40,
  strike: 90,
  defend: 14,
  intel_query: 10,
  mobilize: 55,
  deceive: 48,
  oversight_review: 6,
};
type SessionEntityAsset = {
  id: string;
  entityId: string;
  label: string;
  category: string;
  section: string;
  status: string;
  health: number | null;
  lngLat: [number, number];
  latitude: number;
  longitude: number;
  notes: string | null;
  lastChangeReason: string | null;
};

type AgentMapNode = {
  label: string;
  lngLat: [number, number];
  color: string;
  secondaryColor: string;
  flag: string;
};

type UnknownRecord = Record<string, unknown>;

export default function GlobePage() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const popupRef = useRef<mapboxgl.Popup | null>(null);
  const nodeMarkersRef = useRef<mapboxgl.Marker[]>([]);
  const assetMarkersRef = useRef<mapboxgl.Marker[]>([]);
  const topBarRef = useRef<HTMLDivElement>(null);
  const topBarAnimatedRef = useRef(false);

  const [session, setSession] = useState<SessionState | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [mapReady, setMapReady] = useState(false);
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [panelsCollapsed, setPanelsCollapsed] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [timelineTurn, setTimelineTurn] = useState(0);
  const timelineTurnRef = useRef(0);
  const [interactionFocus, setInteractionFocus] = useState<TimelineInteractionFocus | null>(null);
  const [timelineCollapsed, setTimelineCollapsed] = useState(false);
  const pendingSessionRef = useRef<SessionState | null>(null);
  const [viewportWidth, setViewportWidth] = useState(0);
  const [syntheticEvents, setSyntheticEvents] = useState<SyntheticEvent[]>([]);
  const warnedProviderStatesRef = useRef<Set<string>>(new Set());
  // Snapshot cache: stores SessionState *before* each injection for true rewind
  const snapshotCacheRef = useRef<Map<string, SessionState>>(new Map());

  const handleInjectEvent = async (signal: ExternalSignal) => {
    const rt = window.__trenches;
    const sid = session?.session_id;
    if (!rt || !sid) throw new Error("No active session");
    // Snapshot current state before injection
    const preInjectSnapshot = session;
    const result = await rt.sessionClient.ingestNews(sid, { signals: [signal] });
    applySessionSnapshot(result.session);
    const eventId = `synth-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    // Cache the pre-injection state for rewind
    if (preInjectSnapshot) {
      snapshotCacheRef.current.set(eventId, preInjectSnapshot);
    }
    setSyntheticEvents((prev) => [
      ...prev,
      {
        id: eventId,
        signal,
        injectedAtTurn: result.session.world.turn,
        timestamp: Date.now(),
      },
    ]);
  };

  const handleRemoveEvent = (eventId: string) => {
    setSyntheticEvents((prev) => prev.map((e) => (e.id === eventId ? { ...e, removed: true } : e)));
    snapshotCacheRef.current.delete(eventId);
  };

  const handleRewindToEvent = (eventId: string) => {
    const evt = syntheticEvents.find((e) => e.id === eventId);
    if (!evt) return;
    // Restore the exact pre-injection session state
    const snapshot = snapshotCacheRef.current.get(eventId);
    if (snapshot) {
      setSession(snapshot);
      setTimelineTurn(snapshot.world.turn);
    } else {
      // Fallback: just rewind the scrubber
      setTimelineTurn(Math.max(0, evt.injectedAtTurn - 1));
    }
    // Mark this and all later events as removed + clean up their snapshots
    setSyntheticEvents((prev) =>
      prev.map((e) => {
        if (e.injectedAtTurn >= evt.injectedAtTurn) {
          snapshotCacheRef.current.delete(e.id);
          return { ...e, removed: true };
        }
        return e;
      }),
    );
  };

  // Bottom offset for side panels: just clear the timeline + bottom padding + small gap
  const sideBarBottom = timelineCollapsed ? 72 : 244;

  const [visible, setVisible] = useState({
    topBar: true,
    mapControls: true,
    newsFeed: true,
    activityLog: true,
    timeline: true,
    agentContext: true,
  });
  const togglePanelsRef = useRef<Array<(collapsed: boolean) => void>>([]);

  const activityItems = useMemo(() => deriveActivityItems(session), [session]);
  const simNewsItems = useMemo(() => deriveNewsItems(session), [session]);
  const entityAssets = useMemo(() => deriveSessionEntityAssets(session), [session]);
  const agentMapNodes = useMemo(() => deriveAgentMapNodes(session, entityAssets), [session, entityAssets]);
  const selectedAgentMeta = selectedAgent ? agentMapNodes[selectedAgent] ?? null : null;
  const topBarStats = useMemo(() => deriveTopBarStats(session, activityItems.length), [session, activityItems.length]);
  const [selectedExtraStatKey, setSelectedExtraStatKey] = useState<string>("");
  const selectedExtraStat = useMemo(
    () => topBarStats.extra.find((stat) => stat.key === selectedExtraStatKey) ?? topBarStats.extra[0] ?? null,
    [selectedExtraStatKey, topBarStats.extra],
  );
  const topBarLayout = useMemo(() => {
    const leftW = leftPanelOpen ? LEFT_PANEL_OPEN_W : LEFT_PANEL_COLLAPSED_W;
    const rightW = rightPanelOpen ? RIGHT_PANEL_OPEN_W : RIGHT_PANEL_COLLAPSED_W;
    const available = Math.max(TOP_BAR_MIN_WIDTH, viewportWidth - leftW - rightW - TOP_BAR_GAP * 2);
    const width = Math.min(TOP_BAR_MAX_WIDTH, available);
    // Center between the two panel edges, not the viewport center
    const leftEdge = leftW + TOP_BAR_GAP;
    const rightEdge = viewportWidth - rightW - TOP_BAR_GAP;
    const centerX = (leftEdge + rightEdge) / 2;
    const offsetFromCenter = centerX - viewportWidth / 2;
    return { width, offsetX: offsetFromCenter };
  }, [leftPanelOpen, rightPanelOpen, viewportWidth]);
  const topBarCompact = topBarLayout.width < TOP_BAR_COMPACT_BREAKPOINT;

  const applySessionSnapshot = (incoming: SessionState) => {
    setSession((current) => {
      if (!current) return incoming;
      if (incoming.world.turn > current.world.turn) return incoming;
      if (incoming.world.turn === current.world.turn && incoming.updated_at >= current.updated_at) return incoming;
      return current;
    });
    setTimelineTurn((current) => {
      const updated = Math.max(current, incoming.world.turn);
      timelineTurnRef.current = updated;
      return updated;
    });
  };

  // If user is rewound, buffer poll updates; when they catch up, flush the buffer
  const isFollowing = !session || timelineTurn >= (session?.world.turn ?? 0);

  useEffect(() => {
    if (isFollowing && pendingSessionRef.current) {
      applySessionSnapshot(pendingSessionRef.current);
      pendingSessionRef.current = null;
    }
  }, [isFollowing]);

  useEffect(() => {
    if (!topBarStats.extra.some((stat) => stat.key === selectedExtraStatKey) && topBarStats.extra[0]) {
      setSelectedExtraStatKey(topBarStats.extra[0].key);
    }
  }, [selectedExtraStatKey, topBarStats.extra]);

  useEffect(() => {
    const updateViewportWidth = () => setViewportWidth(window.innerWidth);
    updateViewportWidth();
    window.addEventListener("resize", updateViewportWidth);
    return () => window.removeEventListener("resize", updateViewportWidth);
  }, []);

  useEffect(() => {
    const topBar = topBarRef.current;
    if (!topBar) return;

    if (!topBarAnimatedRef.current) {
      gsap.set(topBar, { width: topBarLayout.width, x: topBarLayout.offsetX });
      topBarAnimatedRef.current = true;
      return;
    }

    const tween = gsap.to(topBar, {
      width: topBarLayout.width,
      x: topBarLayout.offsetX,
      duration: 0.45,
      ease: "power2.inOut",
      overwrite: "auto",
    });
    return () => {
      tween.kill();
    };
  }, [topBarLayout]);

  const newsItems = useMemo(() => simNewsItems, [simNewsItems]);

  const intensityByAgent = useMemo(() => {
    const intensity = new Map<string, number>();
    for (const [agent] of Object.entries(AGENT_META)) intensity.set(agent, 10);

    if (!session) return intensity;

    for (const trace of session.recent_traces) {
      for (const [agentId, action] of Object.entries(trace.actions)) {
        if (!action) continue;
        const current = intensity.get(agentId) ?? 10;
        const delta = trace.tension_after - trace.tension_before;
        intensity.set(agentId, Math.min(100, current + ACTION_HEAT[action.type] * 0.25 + Math.max(0, delta)));
      }
    }

    return intensity;
  }, [session]);



  const selectedAgentActions = useMemo(() => {
    if (!selectedAgent || !session) return [];
    return activityItems
      .filter((item) => item.agent === selectedAgent && item.turn <= timelineTurn)
      .slice(0, 8);
  }, [selectedAgent, session, activityItems, timelineTurn]);

  const selectedAgentContext = useMemo(() => {
    if (!selectedAgent || !session) return null;
    return session.observations[selectedAgent];
  }, [selectedAgent, session]);



  useEffect(() => {
    const token = getMapboxToken();
    if (!token) {
      setMapError("NEXT_PUBLIC_MAPBOX_TOKEN / NEXT_PUBLIC_MAP_KEY / MAP_KEY not set");
      return;
    }

    const container = mapContainerRef.current;
    if (!container || mapRef.current) return;

    mapboxgl.accessToken = token;
    const map = new mapboxgl.Map({
      container,
      style: MAP_STYLE,
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      projection: "globe",
      attributionControl: false,
    });

    mapRef.current = map;
    popupRef.current = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: true,
      anchor: "left",
      offset: [10, 0],
      className: "asset-name-popup",
    });

    map.on("load", () => {
      setMapReady(true);
    });

    map.on("style.load", () => {
      map.setFog({
        color: "rgba(20, 20, 20, 0.95)",
        "high-color": "rgba(30, 30, 30, 0.4)",
        "space-color": "rgba(10, 10, 10, 1)",
        "star-intensity": 0.06,
        range: [-1, 2],
      });

      for (const layerId of INTEL_HIDDEN_LAYERS) {
        if (map.getLayer(layerId)) map.setLayoutProperty(layerId, "visibility", "none");
      }
    });

    map.on("error", (e) => {
      console.error("[Mapbox error]", e.error?.message || e);
    });

    return () => {
      nodeMarkersRef.current.forEach((marker) => marker.remove());
      nodeMarkersRef.current = [];
      assetMarkersRef.current.forEach((marker) => marker.remove());
      assetMarkersRef.current = [];
      popupRef.current?.remove();
      popupRef.current = null;
      map.remove();
      mapRef.current = null;
      setMapReady(false);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        const rt = await bootstrapPlatform();
        if (cancelled || rt.backendStatus !== "healthy") return;
        const sess = await rt.sessionClient.createSession();
        if (!cancelled) {
          applySessionSnapshot(sess);
        }
        // Enable live mode with auto_step so backend steps when RSS signals arrive
        const liveSession = await rt.sessionClient.setLiveMode(sess.session_id, {
          enabled: true,
          auto_step: true,
          poll_interval_ms: 5000,
        });
        if (!cancelled) {
          applySessionSnapshot(liveSession);
        }
      } catch (err) {
        console.warn("[Session bootstrap failed]", err);
      }
    }
    void init();
    return () => {
      cancelled = true;
    };
  }, []);

  /* ── Step loop: advance only when following AND real providers exist ── */
  useEffect(() => {
    if (!session) return;
    const sessionId = session.session_id;

    // Check if any agent has a real provider (not pure heuristic fallback)
    const hasRealProviders = Object.values(session.model_bindings ?? {}).some(
      (b) => b.decision_mode !== "heuristic_fallback"
    );

    let stopped = false;
    let busy = false;
    const POLL_MS = 5_000;

    const interval = setInterval(async () => {
      if (stopped || busy) return;
      busy = true;
      try {
        const rt = window.__trenches;
        if (!rt) return;

        const isFollowing = timelineTurnRef.current >= (session.world.turn ?? 0);

        if (isFollowing && hasRealProviders) {
          // Step the simulation forward (real providers available)
          const result = await rt.sessionClient.stepSession(sessionId, {
            actions: {},
            external_signals: [],
          });
          if (!stopped) {
            applySessionSnapshot(result.session);
          }
        } else if (!isFollowing) {
          // User is rewound — just poll current state, don't advance
          const latest = await rt.sessionClient.getSession(sessionId);
          if (!stopped) {
            pendingSessionRef.current = latest;
          }
        }
        // else: following but no real providers → do nothing (idle)
      } catch (err) {
        console.warn("[Step loop error]", err);
      } finally {
        busy = false;
      }
    }, POLL_MS);

    return () => {
      stopped = true;
      clearInterval(interval);
    };
  }, [session?.session_id]);

  useEffect(() => {
    if (!session) return;

    for (const [agentId, binding] of Object.entries(session.model_bindings ?? {})) {
      if (!isFallbackBinding(binding)) continue;
      const warningKey = `${session.session_id}:${session.updated_at}:${agentId}:${binding.provider}:${binding.decision_mode}`;
      if (warnedProviderStatesRef.current.has(warningKey)) continue;
      warnedProviderStatesRef.current.add(warningKey);
      console.warn("[Trenches provider fallback]", {
        sessionId: session.session_id,
        agentId,
        provider: binding.provider,
        decisionMode: binding.decision_mode,
        model: binding.model_name,
        baseUrl: binding.base_url ?? null,
        notes: binding.notes,
      });
    }

    for (const action of session.action_log ?? []) {
      const mode = typeof action.metadata?.mode === "string" ? action.metadata.mode : null;
      const provider = typeof action.metadata?.provider === "string" ? action.metadata.provider : null;
      if (mode !== "heuristic_fallback" && provider !== "openrouter") continue;
      const warningKey = `${action.created_at}:${action.actor}:${mode ?? "unknown"}:${provider ?? "unknown"}`;
      if (warnedProviderStatesRef.current.has(warningKey)) continue;
      warnedProviderStatesRef.current.add(warningKey);
      console.warn("[Trenches action fallback]", {
        sessionId: session.session_id,
        actor: action.actor,
        turn: action.turn,
        actionType: action.action_type,
        summary: action.summary,
        mode,
        provider,
        providerError: action.metadata?.provider_error ?? null,
      });
    }
  }, [session]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    nodeMarkersRef.current.forEach((marker) => marker.remove());
    nodeMarkersRef.current = [];

    for (const agentId of MAP_NODE_IDS) {
      const meta = agentMapNodes[agentId];
      if (!meta) continue;
      const intensity = intensityByAgent.get(agentId) ?? 10;
      const markerEl = document.createElement("button");
      markerEl.type = "button";
      markerEl.style.background = "transparent";
      markerEl.style.border = "none";
      markerEl.style.padding = "0";
      markerEl.style.cursor = "pointer";
      markerEl.title = `${meta.label}`;
      markerEl.setAttribute("aria-label", `${meta.flag} ${meta.label} model intensity ${Math.round(intensity)}`);

      const pulse = document.createElement("div");
      pulse.style.position = "relative";
      pulse.style.width = `${20 + intensity * 0.25}px`;
      pulse.style.height = `${20 + intensity * 0.25}px`;
      pulse.style.borderRadius = "999px";
      pulse.style.background = `${meta.color}22`;
      pulse.style.boxShadow = `0 0 ${4 + intensity * 0.1}px ${meta.color}55`;
      pulse.style.animation = `pulse-${agentId} ${Math.max(1.8, 3.8 - intensity / 70)}s ease-out infinite`;

      const core = document.createElement("div");
      core.style.position = "absolute";
      core.style.top = "50%";
      core.style.left = "50%";
      core.style.transform = "translate(-50%, -50%)";
      core.style.width = "14px";
      core.style.height = "14px";
      core.style.borderRadius = "999px";
      core.style.display = "flex";
      core.style.alignItems = "center";
      core.style.justifyContent = "center";
      core.style.fontSize = "10px";
      core.textContent = meta.flag;

      pulse.appendChild(core);
      markerEl.appendChild(pulse);

      const styleTag = document.createElement("style");
      styleTag.textContent = `
        @keyframes pulse-${agentId} {
          0% { transform: scale(0.85); opacity: 0.9; }
          70% { transform: scale(1.35); opacity: 0.25; }
          100% { transform: scale(1.55); opacity: 0; }
        }
      `;
      markerEl.appendChild(styleTag);

      markerEl.onclick = () => {
        setSelectedAgent(agentId);
        popupRef.current
          ?.setLngLat(meta.lngLat)
          .setHTML(
            `<div style="background:${meta.color};color:#071014;padding:4px 8px;border-radius:999px;border:1px solid ${meta.color};font:600 10px ui-monospace, SFMono-Regular, Menlo, monospace;letter-spacing:0.08em;text-transform:uppercase;box-shadow:0 4px 16px rgba(0,0,0,0.25);white-space:nowrap;">${escapeHtml(meta.label)}</div>`,
          )
          .addTo(map);
      };

      const marker = new mapboxgl.Marker({ element: markerEl, anchor: "center" })
        .setLngLat(meta.lngLat)
        .addTo(map);

      nodeMarkersRef.current.push(marker);
    }
  }, [agentMapNodes, intensityByAgent]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    assetMarkersRef.current.forEach((marker) => marker.remove());
    assetMarkersRef.current = [];

    for (const asset of entityAssets) {
      const entityMeta = agentMapNodes[asset.entityId];
      if (!entityMeta) continue;

      const layerStyle = getAssetMarkerStyle(asset);
      const markerEl = document.createElement("button");
      markerEl.type = "button";
      markerEl.style.width = `${layerStyle.size}px`;
      markerEl.style.height = `${layerStyle.size}px`;
      markerEl.style.borderRadius = "999px";
      markerEl.style.cursor = "pointer";
      markerEl.style.padding = "0";
      markerEl.style.background = entityMeta.color;
      markerEl.style.opacity = String(layerStyle.opacity);
      markerEl.style.boxShadow = `0 0 7px ${entityMeta.color}`;
      markerEl.style.border = `1px solid ${layerStyle.borderColor}`;
      markerEl.title = `${entityMeta.flag} ${entityMeta.label}: ${asset.label} (${asset.category}/${asset.section}) status=${asset.status} | lat ${asset.latitude.toFixed(2)}, lon ${asset.longitude.toFixed(2)}`;
      markerEl.setAttribute("aria-label", `${entityMeta.label} asset ${asset.label}`);
      markerEl.onclick = () => {
        setSelectedAgent(null);
        popupRef.current
          ?.setLngLat(asset.lngLat)
          .setHTML(
            `<div style="background:${entityMeta.color};color:#071014;padding:4px 8px;border-radius:999px;border:1px solid ${entityMeta.color};font:600 10px ui-monospace, SFMono-Regular, Menlo, monospace;letter-spacing:0.08em;text-transform:uppercase;box-shadow:0 4px 16px rgba(0,0,0,0.25);white-space:nowrap;">${escapeHtml(asset.label)} · ${escapeHtml(asset.status)}</div>`,
          )
          .addTo(map);
      };

      const marker = new mapboxgl.Marker({ element: markerEl, anchor: "center" })
        .setLngLat(asset.lngLat)
        .addTo(map);

      assetMarkersRef.current.push(marker);
    }

    return () => {
      assetMarkersRef.current.forEach((marker) => marker.remove());
      assetMarkersRef.current = [];
    };
  }, [agentMapNodes, entityAssets]);






  return (
    <div className="relative h-screen w-screen overflow-hidden bg-background">
      <div ref={mapContainerRef} className="absolute inset-0" style={{ width: "100vw", height: "100vh" }} />

      {visible.topBar && <div ref={topBarRef} className="pointer-events-none absolute top-4 left-1/2 z-20 -translate-x-1/2">
        <div className="relative h-full rounded-md border-[0.75px] border-border/30">
          <GlowingEffect spread={40} glow={true} disabled={false} proximity={64} inactiveZone={0.01} borderWidth={2} />
          <div
            className="pointer-events-auto flex h-9 w-full select-none items-center gap-3 overflow-hidden rounded-md bg-card/25 px-4 font-sans backdrop-blur-lg"
            style={{
              boxShadow:
                "0 0 8px rgba(0,0,0,0.03), 0 2px 6px rgba(0,0,0,0.08), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 12px rgba(0,0,0,0.15)",
            }}
          >
          <div className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-primary" />
          <span className={cn("shrink-0 text-xs font-semibold tracking-[0.2em] text-foreground/80 uppercase", topBarCompact && "tracking-[0.16em]")}>Trenches</span>
          <span className="shrink-0 text-[10px] font-mono text-muted-foreground">{session ? `T${timelineTurn}` : "IDLE"}</span>
          {session && (
            <>
              <div className="mx-0.5 h-4 w-px shrink-0 bg-border/40" />
              <div className="flex min-w-0 flex-1 items-center justify-between gap-3 font-mono text-xs">
                <div className="flex min-w-0 items-center gap-3 overflow-hidden whitespace-nowrap">
                {topBarStats.primary.map((stat) => (
                  <StatusPill key={stat.key} label={stat.label} value={stat.value} warn={stat.warn} icon={stat.icon} compact={topBarCompact} />
                ))}
                </div>
                {selectedExtraStat && (
                  <div className="flex shrink-0 items-center gap-2">
                    {topBarCompact ? (
                      <ListFilter className="h-3.5 w-3.5 text-muted-foreground" />
                    ) : (
                      <label htmlFor="topbar-extra-stat" className="text-muted-foreground">STAT</label>
                    )}
                    <div className="relative shrink-0">
                      <select
                        id="topbar-extra-stat"
                        value={selectedExtraStat.key}
                        onChange={(event) => setSelectedExtraStatKey(event.target.value)}
                        className={cn(
                          "appearance-none border border-border/30 bg-card/30 py-1 text-[10px] uppercase tracking-[0.14em] text-foreground outline-none",
                          topBarCompact ? "w-[148px] pl-2.5 pr-8" : "w-[190px] pl-3 pr-9",
                        )}
                      >
                        {topBarStats.extra.map((stat) => (
                          <option key={stat.key} value={stat.key}>
                            {stat.label}
                          </option>
                        ))}
                      </select>
                      <ChevronDown className="pointer-events-none absolute top-1/2 right-2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                    </div>
                    <StatusPill label="" value={selectedExtraStat.value} warn={selectedExtraStat.warn} icon={selectedExtraStat.icon} compact={true} />
                  </div>
                )}
              </div>
            </>
          )}
          </div>
        </div>
      </div>}



      {selectedAgent && visible.agentContext && (
        <div className="absolute bottom-52 left-1/2 z-20 w-[420px] -translate-x-1/2 border border-border/30 bg-card/70 p-3 backdrop-blur-xl">
          <div className="mb-2 flex items-center justify-between">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-foreground/80">{selectedAgentMeta?.flag ?? ""} {selectedAgentMeta?.label ?? selectedAgent}</div>
              <div className="text-[9px] font-mono text-muted-foreground">Model context and action history</div>
            </div>
            <button
              onClick={() => setSelectedAgent(null)}
              className="border border-border/40 px-2 py-1 text-[9px] font-mono uppercase tracking-wider text-muted-foreground hover:text-foreground"
            >
              Close
            </button>
          </div>
          <p className="mb-2 line-clamp-2 text-xs text-foreground/80">
            {(selectedAgentContext?.event_log[0]?.summary ?? "No active context events for this actor in current replay window.")}
          </p>
          <div className="space-y-1">
            {selectedAgentActions.length === 0 ? (
              <div className="text-[10px] font-mono text-muted-foreground">No actions up to this turn.</div>
            ) : (
              selectedAgentActions.map((item) => (
                <div key={item.id} className="border border-border/30 px-2 py-1 text-[10px]">
                  <span className="font-mono text-muted-foreground">T{item.turn}</span> · {item.actionType.toUpperCase()} · {item.summary}
                </div>
              ))
            )}
          </div>
        </div>
      )}


      {/* Bottom container: chat + controls bar + timeline grouped together */}
      <div className="absolute right-4 bottom-4 left-4 z-30 flex flex-col items-center gap-2 pointer-events-none">
        <ChatPanel
          open={chatOpen}
          onClose={() => setChatOpen(false)}
          sessionId={session?.session_id ?? null}
          session={session}
          syntheticEvents={syntheticEvents}
          onInjectEvent={handleInjectEvent}
          onRemoveEvent={handleRemoveEvent}
          onRewindToEvent={handleRewindToEvent}
        />
        {visible.mapControls && <div className="pointer-events-auto">
          <div className="relative rounded-md border-[0.75px] border-border/30">
            <GlowingEffect spread={40} glow={true} disabled={false} proximity={64} inactiveZone={0.01} borderWidth={2} />
            <div
              className="flex select-none items-center gap-1 rounded-md bg-card/30 px-3 py-2 backdrop-blur-xl"
              style={{
                boxShadow:
                  "0 0 8px rgba(0,0,0,0.03), 0 2px 6px rgba(0,0,0,0.08), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 12px rgba(0,0,0,0.15)",
              }}
            >
            <button onClick={() => mapRef.current?.easeTo({ zoom: (mapRef.current?.getZoom() ?? DEFAULT_ZOOM) + 0.8, duration: 300 })} disabled={!mapReady} className="flex h-7 w-7 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40" title="Zoom in">
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
            <button onClick={() => mapRef.current?.easeTo({ zoom: (mapRef.current?.getZoom() ?? DEFAULT_ZOOM) - 0.8, duration: 300 })} disabled={!mapReady} className="flex h-7 w-7 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40" title="Zoom out">
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <div className="mx-1 h-4 w-px bg-border/40" />
            <button
              onClick={() => setChatOpen((prev) => !prev)}
              className={cn("flex h-7 w-7 cursor-pointer items-center justify-center transition-colors", chatOpen ? "text-primary" : "text-muted-foreground hover:text-foreground")}
              title="AI Chat"
            >
              <MessageSquare className="h-3.5 w-3.5" />
            </button>
            <div className="mx-1 h-4 w-px bg-border/40" />
            <button
              onClick={() =>
                mapRef.current?.flyTo({
                  center: DEFAULT_CENTER,
                  zoom: DEFAULT_ZOOM,
                  duration: 1200,
                })
              }
              className="flex h-7 w-7 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
              title="Reset view"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => {
                const next = !panelsCollapsed;
                setPanelsCollapsed(next);
                setLeftPanelOpen(!next);
                setRightPanelOpen(!next);
                togglePanelsRef.current.forEach((fn) => fn(next));
              }}
              className="flex h-7 w-7 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
              title={panelsCollapsed ? "Expand panels" : "Collapse panels"}
            >
              {panelsCollapsed ? <Columns2 className="h-3.5 w-3.5" /> : <Maximize className="h-3.5 w-3.5" />}
            </button>
            </div>
          </div>
        </div>}

        {visible.timeline && <EventTimeline session={session} onTurnChange={(t) => { setTimelineTurn(t); timelineTurnRef.current = t; }} interactionFocus={interactionFocus} onInteractionFocus={setInteractionFocus} onRegisterToggle={(fn) => { togglePanelsRef.current[2] = fn; }} embedded onCollapsedChange={setTimelineCollapsed} />}
      </div>

      {visible.newsFeed && <NewsFeed items={newsItems} hydration={session ? getLiveHydration(session.live) : null} interactionFocus={interactionFocus} onInteractionFocus={setInteractionFocus} onRegisterToggle={(fn) => { togglePanelsRef.current[0] = fn; }} onCollapsedChange={(c) => setLeftPanelOpen(!c)} bottomOffset={sideBarBottom} />}
      {visible.activityLog && <ActivityLog items={activityItems} focusTurn={timelineTurn} interactionFocus={interactionFocus} onInteractionFocus={setInteractionFocus} onRegisterToggle={(fn) => { togglePanelsRef.current[1] = fn; }} onCollapsedChange={(c) => setRightPanelOpen(!c)} bottomOffset={sideBarBottom} />}

      {mapError && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-background">
          <div className="border border-border/40 bg-card/80 p-8 text-center backdrop-blur-xl">
            <p className="text-sm font-mono text-muted-foreground uppercase tracking-wider">
              Map error: <code className="text-primary">{mapError}</code>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function StatusPill({
  label,
  value,
  warn,
  icon: Icon,
  compact = false,
}: {
  label: string;
  value: string;
  warn?: boolean;
  icon: LucideIcon;
  compact?: boolean;
}) {
  return (
    <div className="flex shrink-0 items-center gap-2 whitespace-nowrap" title={`${label} ${value}`}>
      {compact ? <Icon className="h-3.5 w-3.5 text-muted-foreground" /> : <span className="text-muted-foreground">{label}</span>}
      <span className={warn ? "text-primary font-bold" : "text-foreground"}>{value}</span>
    </div>
  );
}

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function getStringValue(record: UnknownRecord, key: string): string | null {
  const value = record[key];
  return typeof value === "string" && value.trim() ? value : null;
}

function getNumberValue(record: UnknownRecord, key: string): number | null {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function deriveSessionEntityAssets(session: SessionState | null): SessionEntityAsset[] {
  if (!session) return [];

  const items: SessionEntityAsset[] = [];

  const worldAssetState = session.world.asset_state;
  if (worldAssetState) {
    for (const [entityId, assetsById] of Object.entries(worldAssetState)) {
      for (const asset of Object.values(assetsById)) {
        const latitude = typeof asset.latitude === "number" ? asset.latitude : null;
        const longitude = typeof asset.longitude === "number" ? asset.longitude : null;
        if (latitude === null || longitude === null) continue;

        items.push({
          id: asset.asset_id,
          entityId,
          label: asset.name,
          category: asset.category,
          section: asset.section,
          status: asset.status,
          health: asset.health,
          notes: asset.notes ?? null,
          lastChangeReason: asset.last_change_reason ?? null,
          lngLat: [longitude, latitude],
          longitude,
          latitude,
        });
      }
    }

    if (items.length > 0) {
      return items;
    }
  }

  for (const [entityId, observation] of Object.entries(session.observations)) {
    for (const rawAsset of observation.strategic_assets) {
      if (!rawAsset || typeof rawAsset !== "object") continue;
      const asset = rawAsset as UnknownRecord;
      const latitude = getNumberValue(asset, "latitude");
      const longitude = getNumberValue(asset, "longitude");
      if (latitude === null || longitude === null) continue;

      const label = getStringValue(asset, "name") ?? `${entityId} asset`;
      const category = getStringValue(asset, "category") ?? "tracked";
      const section = getStringValue(asset, "section") ?? "theater";
      const status = getStringValue(asset, "status") ?? "operational";
      const health = getNumberValue(asset, "health");
      const notes = getStringValue(asset, "notes");
      const lastChangeReason = getStringValue(asset, "last_change_reason");
      const assetId = getStringValue(asset, "asset_id") ?? `${entityId}-${label}-${latitude}-${longitude}`;

      items.push({
        id: assetId,
        entityId,
        label,
        category,
        section,
        status,
        health,
        notes,
        lastChangeReason,
        lngLat: [longitude, latitude],
        longitude,
        latitude,
      });
    }
  }

  return items;
}

function deriveAgentMapNodes(
  session: SessionState | null,
  entityAssets: SessionEntityAsset[],
): Record<string, AgentMapNode> {
  const nodes: Record<string, AgentMapNode> = {};

  for (const [agentId, meta] of Object.entries(AGENT_META)) {
    const assets = entityAssets.filter((asset) => asset.entityId === agentId);
    if (assets.length === 0) continue;

    const longitude = assets.reduce((sum, asset) => sum + asset.longitude, 0) / assets.length;
    const latitude = assets.reduce((sum, asset) => sum + asset.latitude, 0) / assets.length;
    const profile = session?.observations[agentId]?.entity_profile as UnknownRecord | undefined;

    nodes[agentId] = {
      label: getStringValue(profile ?? {}, "display_name") ?? meta.defaultLabel,
      lngLat: [longitude, latitude],
      color: meta.color,
      secondaryColor: meta.secondaryColor,
      flag: meta.flag,
    };
  }

  return nodes;
}

function getAssetMarkerStyle(asset: SessionEntityAsset): { size: number; opacity: number; borderColor: string } {
  const isStressed = asset.status !== "operational" || (asset.health !== null && asset.health < 80);
  return {
    size: isStressed ? 9 : 6,
    opacity: isStressed ? 1 : 0.82,
    borderColor: isStressed ? "#ffe082" : "#ffffff",
  };
}

type TopBarExtraStat = {
  key: string;
  label: string;
  value: string;
  icon: LucideIcon;
  warn?: boolean;
};

type LiveHydrationView = NonNullable<SessionState["live"]["hydration"]>;

const DEFAULT_LIVE_HYDRATION: LiveHydrationView = {
  phase: "steady",
  total: 0,
  ready: 0,
  pending: 0,
  error: 0,
};

function getLiveHydration(
  live: SessionState["live"] | null | undefined,
): LiveHydrationView {
  const raw = live?.hydration as Partial<LiveHydrationView> | undefined;
  return {
    phase: raw?.phase ?? DEFAULT_LIVE_HYDRATION.phase,
    total: raw?.total ?? DEFAULT_LIVE_HYDRATION.total,
    ready: raw?.ready ?? DEFAULT_LIVE_HYDRATION.ready,
    pending: raw?.pending ?? DEFAULT_LIVE_HYDRATION.pending,
    error: raw?.error ?? DEFAULT_LIVE_HYDRATION.error,
  };
}

function deriveTopBarStats(session: SessionState | null, activityCount: number): {
  primary: TopBarExtraStat[];
  extra: TopBarExtraStat[];
} {
  if (!session) return { primary: [], extra: [] };

  const world = session.world as Record<string, unknown>;
  const riskScores = asNumberMap(world.risk_scores);
  const behavior = asNumberMap(world.behavioral_consistency);
  const emaTension = asNumberMap(world.ema_tension);
  const coalitionGraph = asArrayMap(world.coalition_graph);
  const actorState = asObjectMap(world.actor_state);
  const assetState = asObjectMap(world.asset_state);
  const activeEvents = asObjectArray(world.active_events);
  const latentEvents = asObjectArray(world.latent_events);
  const lastActions = asObjectArray(world.last_actions);
  const hiddenIntents = asObjectMap(world.hidden_intents);
  const hydration = getLiveHydration(session.live);
  const liveQueue = hydration.pending;
  const coalitionLinks = Object.values(coalitionGraph).reduce((sum, targets) => sum + targets.length, 0);
  const assetCount = Object.values(assetState).reduce((sum, assets) => sum + Object.keys(assets).length, 0);
  const hydrationValue = hydration.total > 0
    ? `${hydration.ready}/${hydration.total}`
    : "0/0";

  const topLevelPrimary = Object.entries(world)
    .filter(([key, value]) => key !== "turn" && typeof value === "number")
    .map(([key, value]) => ({
      key,
      label: humanizeStatKey(key),
      value: formatWorldMetric(key, value as number),
      icon: iconForStatKey(key),
      warn: inferWorldMetricWarning(key, value as number),
    }));

  const activityStat = {
    key: "activity",
    label: "ACTIVITY",
    value: String(activityCount),
    icon: iconForStatKey("activity"),
    warn: activityCount > 24,
  };

  const primary = [...topLevelPrimary, activityStat];
  const extra = [
    { key: "active_events", label: "ACTIVE EVENTS", value: String(activeEvents.length), icon: iconForStatKey("active_events"), warn: activeEvents.length > 8 },
    { key: "latent_events", label: "LATENT EVENTS", value: String(latentEvents.length), icon: iconForStatKey("latent_events"), warn: latentEvents.length > 12 },
    { key: "last_actions", label: "LAST ACTIONS", value: String(lastActions.length), icon: iconForStatKey("last_actions"), warn: lastActions.length > 5 },
    { key: "agents", label: "AGENTS", value: String(Object.keys(actorState).length), icon: iconForStatKey("agents") },
    { key: "assets", label: "ASSETS", value: String(assetCount), icon: iconForStatKey("assets"), warn: assetCount === 0 },
    { key: "coalitions", label: "COALITIONS", value: String(coalitionLinks), icon: iconForStatKey("coalitions"), warn: coalitionLinks === 0 },
    { key: "avg_risk", label: "AVG RISK", value: formatAverage(riskScores), icon: iconForStatKey("avg_risk"), warn: averageOfMap(riskScores) > 0.6 },
    { key: "max_risk", label: "MAX RISK", value: formatPercent(maxOfMap(riskScores)), icon: iconForStatKey("max_risk"), warn: maxOfMap(riskScores) > 0.75 },
    { key: "avg_consistency", label: "AVG CONSISTENCY", value: formatAverage(behavior), icon: iconForStatKey("avg_consistency"), warn: averageOfMap(behavior) < 0.45 },
    { key: "avg_ema", label: "AVG EMA", value: formatPercent(averageOfMap(emaTension) / 100), icon: iconForStatKey("avg_ema"), warn: averageOfMap(emaTension) > 60 },
    { key: "hidden_intents", label: "HIDDEN INTENTS", value: String(Object.keys(hiddenIntents).length), icon: iconForStatKey("hidden_intents") },
    { key: "hydration", label: "HYDRATION", value: hydrationValue, icon: iconForStatKey("hydration"), warn: hydration.phase !== "steady" },
    { key: "live_queue", label: "LIVE QUEUE", value: String(liveQueue), icon: iconForStatKey("live_queue"), warn: liveQueue > 0 },
    { key: "recent_traces", label: "RECENT TRACES", value: String(session.recent_traces.length), icon: iconForStatKey("recent_traces") },
  ].filter((stat) => !primary.some((primaryStat) => primaryStat.key === stat.key));

  return { primary, extra };
}

function iconForStatKey(key: string): LucideIcon {
  switch (key) {
    case "tension":
    case "tension_level":
    case "avg_risk":
    case "max_risk":
      return ShieldAlert;
    case "market":
    case "market_stress":
      return TrendingUp;
    case "oil":
    case "oil_pressure":
      return Droplets;
    case "activity":
      return Activity;
    case "active_events":
      return Radar;
    case "latent_events":
      return Clock3;
    case "last_actions":
    case "recent_traces":
      return History;
    case "agents":
      return Users;
    case "assets":
      return Boxes;
    case "coalitions":
      return Workflow;
    case "avg_consistency":
      return ShieldCheck;
    case "avg_ema":
      return Gauge;
    case "hidden_intents":
      return BrainCircuit;
    case "hydration":
    case "live_queue":
      return Signal;
    default:
      return Gauge;
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function asNumberMap(value: unknown): Record<string, number> {
  if (!value || typeof value !== "object") return {};
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, number] => typeof entry[1] === "number"),
  );
}

function asArrayMap(value: unknown): Record<string, string[]> {
  if (!value || typeof value !== "object") return {};
  return Object.fromEntries(
    Object.entries(value).map(([key, raw]) => [
      key,
      Array.isArray(raw) ? raw.filter((item): item is string => typeof item === "string") : [],
    ]),
  );
}

function asObjectMap(value: unknown): Record<string, Record<string, unknown>> {
  if (!value || typeof value !== "object") return {};
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, Record<string, unknown>] => typeof entry[1] === "object" && entry[1] !== null),
  );
}

function asObjectArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
}

function averageOfMap(values: Record<string, number>): number {
  const entries = Object.values(values);
  if (entries.length === 0) return 0;
  return entries.reduce((sum, value) => sum + value, 0) / entries.length;
}

function maxOfMap(values: Record<string, number>): number {
  const entries = Object.values(values);
  if (entries.length === 0) return 0;
  return Math.max(...entries);
}

function formatAverage(values: Record<string, number>): string {
  return formatPercent(averageOfMap(values));
}

function formatPercent(unitValue: number): string {
  return String(Math.round(unitValue * 100));
}

function humanizeStatKey(key: string): string {
  return key.replaceAll("_", " ").toUpperCase();
}

function formatWorldMetric(key: string, value: number): string {
  if (key.endsWith("_level") || key.endsWith("_stress") || key.endsWith("_pressure")) {
    return String(Math.round(value));
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function inferWorldMetricWarning(key: string, value: number): boolean {
  if (key.includes("tension") || key.includes("stress") || key.includes("pressure")) {
    return value > 60;
  }
  return false;
}

export type NewsItem = {
  id: string;
  source: string;
  summary: string;
  severity: number;
  timestamp: string;
  turn: number;
  agent: string | null;
  type: "event" | "intel";
  url: string;
  translateUrl?: string | null;
  bootstrapOnly?: boolean;
};

export type ActivityItem = {
  id: string;
  turn: number;
  agent: string;
  actionType: string;
  summary: string;
  target?: string | null;
  rewardTotal: number | null;
  tensionDelta: number;
  oversightTriggered: boolean;
  timestamp: string;
};

function newsFingerprint(item: Pick<NewsItem, "source" | "summary">): string {
  return `${item.source.toLowerCase()}::${item.summary.toLowerCase().replace(/\s+/g, " ").trim()}`;
}

function dedupeNewsItems(items: NewsItem[]): NewsItem[] {
  const byFingerprint = new Map<string, NewsItem>();

  for (const item of items) {
    const fingerprint = newsFingerprint(item);
    const existing = byFingerprint.get(fingerprint);
    if (!existing) {
      byFingerprint.set(fingerprint, item);
      continue;
    }
    const existingTs = new Date(existing.timestamp).getTime() || 0;
    const nextTs = new Date(item.timestamp).getTime() || 0;
    const preferNext =
      (existing.bootstrapOnly && !item.bootstrapOnly)
      || (existing.bootstrapOnly === item.bootstrapOnly && nextTs >= existingTs);
    if (preferNext) {
      byFingerprint.set(fingerprint, item);
    }
  }

  return [...byFingerprint.values()].sort((a, b) => {
    const aTs = new Date(a.timestamp).getTime() || 0;
    const bTs = new Date(b.timestamp).getTime() || 0;
    return bTs - aTs;
  });
}

function deriveNewsItems(session: SessionState | null): NewsItem[] {
  if (!session) return [];

  const items: NewsItem[] = [];
  const seenPacketIds = new Set<string>();
  for (const [agentId, obs] of Object.entries(session.observations)) {
    for (const packet of obs.source_packets) {
      if (packet.status !== "ok" || !packet.summary || seenPacketIds.has(packet.source_id)) continue;
      seenPacketIds.add(packet.source_id);
      items.push({
        id: `packet-${packet.source_id}`,
        source: packet.source_name,
        summary: packet.summary,
        severity: packet.delivery === "live_demo" ? 0.78 : 0.62,
        timestamp: packet.fetched_at ?? session.updated_at,
        turn: session.world.turn,
        agent: agentId,
        type: "intel",
        url: `https://news.google.com/search?q=${encodeURIComponent(`${packet.source_name} ${packet.summary}`.slice(0, 80))}`,
      });
    }
  }

  for (const event of session.world.active_events) {
    items.push({
      id: event.id,
      source: event.source,
      summary: event.summary,
      severity: event.severity,
      timestamp: session.updated_at,
      turn: session.world.turn,
      agent: event.affected_agents[0] ?? null,
      type: "event",
      url: `https://news.google.com/search?q=${encodeURIComponent(event.summary.slice(0, 80))}`,
    });
  }

  for (const [agentId, obs] of Object.entries(session.observations)) {
    for (const brief of obs.public_brief) {
      items.push({
        id: `${agentId}-pub-${brief.source}-${brief.summary.slice(0, 20)}`,
        source: brief.source,
        summary: brief.summary,
        severity: brief.confidence,
        timestamp: session.updated_at,
        turn: session.world.turn,
        agent: agentId,
        type: "intel",
        url: `https://news.google.com/search?q=${encodeURIComponent(brief.summary.slice(0, 80))}`,
      });
    }
  }

  return dedupeNewsItems(items).slice(0, 40);
}

function isFallbackBinding(binding: EntityModelBinding): boolean {
  return binding.decision_mode === "heuristic_fallback" || binding.provider === "openrouter";
}

function deriveActivityItems(session: SessionState | null): ActivityItem[] {
  if (!session) return [];

  const items: ActivityItem[] = [];
  for (const trace of [...session.recent_traces].reverse()) {
    for (const [agentId, action] of Object.entries(trace.actions)) {
      if (!action) continue;
      items.push({
        id: `${agentId}-${trace.turn}`,
        turn: trace.turn,
        agent: action.actor,
        actionType: action.type,
        summary: action.summary,
        target: action.target,
        rewardTotal: trace.rewards[agentId]?.total ?? null,
        tensionDelta: trace.tension_after - trace.tension_before,
        oversightTriggered: trace.oversight.triggered,
        timestamp: trace.created_at,
      });
    }
  }
  return items.slice(0, 120);
}
