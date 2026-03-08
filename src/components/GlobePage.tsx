"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ZoomIn, ZoomOut, RotateCcw, Columns2, Maximize, MessageSquare } from "lucide-react";
import mapboxgl from "mapbox-gl";
import { NewsFeed } from "@/src/components/NewsFeed";
import { ActivityLog } from "@/src/components/ActivityLog";
import { ChatPanel } from "@/src/components/ChatPanel";
import { EventTimeline, type TimelineInteractionFocus } from "@/src/components/EventTimeline";

import { cn } from "@/src/lib/utils";
import type { AgentAction, SessionState } from "@/src/lib/types";
import { bootstrapPlatform } from "@/src/app/bootstrap";
import { deriveTimelineEvents } from "@/src/lib/timeline-types";
import { getMapboxToken } from "@/src/lib/env";
import usAssets from "../../entities/us/assets.json";
import israelAssets from "../../entities/israel/assets.json";
import iranAssets from "../../entities/iran/assets.json";
import hezbollahAssets from "../../entities/hezbollah/assets.json";
import gulfAssets from "../../entities/gulf/assets.json";
import oversightAssets from "../../entities/oversight/assets.json";

const MAP_STYLE = "mapbox://styles/mapbox/dark-v11";
const DEFAULT_CENTER: [number, number] = [41.8, 27.8];
const DEFAULT_ZOOM = 1.8;


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

const AGENT_MAP_NODES: Record<string, { label: string; lngLat: [number, number]; color: string; secondaryColor: string; flag: string }> = {
  us: { label: "United States", lngLat: [-77.0369, 38.9072], color: "#b22234", secondaryColor: "#3c3b6e", flag: "🇺🇸" },
  israel: { label: "Israel", lngLat: [35.2137, 31.7683], color: "#005eb8", secondaryColor: "#ffffff", flag: "🇮🇱" },
  iran: { label: "Iran", lngLat: [51.389, 35.6892], color: "#239f40", secondaryColor: "#da0000", flag: "🇮🇷" },
  hezbollah: { label: "Hezbollah", lngLat: [35.5018, 33.8938], color: "#f4c542", secondaryColor: "#1f7a1f", flag: "🇱🇧" },
  gulf: { label: "Gulf", lngLat: [54.3773, 24.4539], color: "#00732f", secondaryColor: "#ce1126", flag: "🇦🇪" },
  oversight: { label: "Oversight", lngLat: [2.3522, 48.8566], color: "#1f4e79", secondaryColor: "#f4c542", flag: "🏳️‍⚖️" },
};
const MAP_NODE_IDS = Object.keys(AGENT_MAP_NODES).filter((id) => id !== "oversight");


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

type AssetRecord = Record<string, unknown>;

const ENTITY_ASSET_PACKS = {
  us: usAssets,
  israel: israelAssets,
  iran: iranAssets,
  hezbollah: hezbollahAssets,
  gulf: gulfAssets,
  oversight: oversightAssets,
} as const;

const ASSET_LAYERS = [
  "locations",
  "fronts",
  "infrastructure",
  "strategic_sites",
  "alliance_anchors",
  "chokepoints",
  "geospatial_anchors",
] as const;


const ASSET_LAYER_STYLE: Record<(typeof ASSET_LAYERS)[number], { size: number; opacity: number }> = {
  locations: { size: 7, opacity: 0.95 },
  fronts: { size: 8, opacity: 0.92 },
  infrastructure: { size: 6, opacity: 0.9 },
  strategic_sites: { size: 9, opacity: 1 },
  alliance_anchors: { size: 8, opacity: 0.95 },
  chokepoints: { size: 10, opacity: 1 },
  geospatial_anchors: { size: 7, opacity: 0.95 },
};

type HardCodedEntityAsset = {
  id: string;
  entityId: string;
  label: string;
  layer: (typeof ASSET_LAYERS)[number];
  lngLat: [number, number];
  latitude: number;
  longitude: number;
  hasDirectCoordinates: boolean;
};

const HARD_CODED_ENTITY_ASSETS: HardCodedEntityAsset[] = Object.entries(ENTITY_ASSET_PACKS).flatMap(([entityId, pack]) => {
  const locationLookup = new Map<string, [number, number]>();
  for (const record of (pack.locations ?? []) as AssetRecord[]) {
    const name = typeof record.name === "string" ? record.name : undefined;
    const lat = typeof record.lat === "number" ? record.lat : undefined;
    const lon = typeof record.lon === "number" ? record.lon : undefined;
    if (name && lat !== undefined && lon !== undefined) {
      locationLookup.set(name, [lon, lat]);
    }
  }

  return ASSET_LAYERS.flatMap((layer) => {
    const rawRecords = (pack as unknown as Record<string, unknown>)[layer];
    const records = Array.isArray(rawRecords) ? (rawRecords.filter((item): item is AssetRecord => typeof item === "object" && item !== null)) : [];
    return records
      .map((record, index) => {
        const lat = typeof record.lat === "number" ? record.lat : undefined;
        const lon = typeof record.lon === "number" ? record.lon : undefined;
        const anchorLat = typeof record.anchor_lat === "number" ? record.anchor_lat : undefined;
        const anchorLon = typeof record.anchor_lon === "number" ? record.anchor_lon : undefined;
        const linkedLocation = typeof record.location === "string" ? locationLookup.get(record.location) : undefined;

        const resolved =
          lat !== undefined && lon !== undefined
            ? { lngLat: [lon, lat] as [number, number], hasDirectCoordinates: true }
            : anchorLat !== undefined && anchorLon !== undefined
              ? { lngLat: [anchorLon, anchorLat] as [number, number], hasDirectCoordinates: true }
              : linkedLocation
                ? { lngLat: linkedLocation, hasDirectCoordinates: false }
                : null;

        if (!resolved) {
          return null;
        }

        const label = typeof record.name === "string" ? record.name : `${entityId} ${layer} ${index + 1}`;
        return {
          id: `${entityId}-${layer}-${index}`,
          entityId,
          layer,
          label,
          lngLat: resolved.lngLat,
          longitude: resolved.lngLat[0],
          latitude: resolved.lngLat[1],
          hasDirectCoordinates: resolved.hasDirectCoordinates,
        };
      })
      .filter(Boolean) as HardCodedEntityAsset[];
  });
});

export default function GlobePage() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const popupRef = useRef<mapboxgl.Popup | null>(null);
  const nodeMarkersRef = useRef<mapboxgl.Marker[]>([]);
  const assetMarkersRef = useRef<mapboxgl.Marker[]>([]);

  const [session, setSession] = useState<SessionState | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [mapReady, setMapReady] = useState(false);
  const [panelsCollapsed, setPanelsCollapsed] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [timelineTurn, setTimelineTurn] = useState(0);
  const [interactionFocus, setInteractionFocus] = useState<TimelineInteractionFocus | null>(null);
  const [timelineCollapsed, setTimelineCollapsed] = useState(false);

  // Bottom offset for side panels: just clear the timeline + bottom padding + small gap
  const sideBarBottom = timelineCollapsed ? 72 : 204;

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
  const newsItems = useMemo(() => deriveNewsItems(session), [session]);

  const intensityByAgent = useMemo(() => {
    const intensity = new Map<string, number>();
    for (const [agent] of Object.entries(AGENT_MAP_NODES)) intensity.set(agent, 10);

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
        const sess = await rt.sessionClient.createSession({ seed: 7 });
        const liveSession = await rt.sessionClient.setLiveMode(sess.session_id, {
          enabled: true,
          auto_step: true,
          poll_interval_ms: 10000,
        });
        const refreshed = await rt.sessionClient.refreshSources(sess.session_id);
        if (!cancelled) {
          setSession(refreshed ?? liveSession);
          setTimelineTurn((refreshed ?? liveSession).world.turn);
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

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    nodeMarkersRef.current.forEach((marker) => marker.remove());
    nodeMarkersRef.current = [];

    for (const agentId of MAP_NODE_IDS) {
      const meta = AGENT_MAP_NODES[agentId];
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
  }, [intensityByAgent, selectedAgent]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    assetMarkersRef.current.forEach((marker) => marker.remove());
    assetMarkersRef.current = [];

    for (const asset of HARD_CODED_ENTITY_ASSETS) {
      const entityMeta = AGENT_MAP_NODES[asset.entityId];
      if (!entityMeta) continue;

      const layerStyle = ASSET_LAYER_STYLE[asset.layer];
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
      markerEl.style.border = `1px solid ${entityMeta.secondaryColor}`;
      markerEl.title = `${entityMeta.flag} ${entityMeta.label}: ${asset.label} (${asset.layer}) | lat ${asset.latitude.toFixed(2)}, lon ${asset.longitude.toFixed(2)}`;
      markerEl.setAttribute("aria-label", `${entityMeta.label} asset ${asset.label}`);
      markerEl.onclick = () => {
        setSelectedAgent(null);
        popupRef.current
          ?.setLngLat(asset.lngLat)
          .setHTML(
            `<div style="background:${entityMeta.color};color:#071014;padding:4px 8px;border-radius:999px;border:1px solid ${entityMeta.color};font:600 10px ui-monospace, SFMono-Regular, Menlo, monospace;letter-spacing:0.08em;text-transform:uppercase;box-shadow:0 4px 16px rgba(0,0,0,0.25);white-space:nowrap;">${escapeHtml(asset.label)}</div>`,
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
  }, []);






  return (
    <div className="relative h-screen w-screen overflow-hidden bg-background">
      <div ref={mapContainerRef} className="absolute inset-0" style={{ width: "100vw", height: "100vh" }} />

      {visible.topBar && <div className="pointer-events-none absolute top-6 left-1/2 z-20 -translate-x-1/2">
        <div
          className="pointer-events-auto flex select-none items-center gap-4 border border-border/40 bg-card/60 px-5 py-2.5 font-sans backdrop-blur-xl"
          style={{
            boxShadow:
              "0 0 8px rgba(0,0,0,0.03), 0 2px 6px rgba(0,0,0,0.08), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 12px rgba(0,0,0,0.15)",
          }}
        >
          <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
          <span className="text-xs font-semibold tracking-[0.2em] text-foreground/80 uppercase">Trenches</span>
          <span className="text-[10px] font-mono text-muted-foreground">{session ? `T${timelineTurn}` : "IDLE"}</span>
          {session && (
            <>
              <div className="mx-1 h-4 w-px bg-border/40" />
              <div className="flex items-center gap-4 font-mono text-xs">
                <StatusPill label="TENSION" value={session.world.tension_level.toFixed(0)} warn={session.world.tension_level > 60} />
                <StatusPill label="MARKET" value={session.world.market_stress.toFixed(0)} warn={session.world.market_stress > 60} />
                <StatusPill label="OIL" value={session.world.oil_pressure.toFixed(0)} warn={session.world.oil_pressure > 60} />
                <StatusPill label="EVENTS" value={String(session.world.active_events.length)} />
              </div>
            </>
          )}
        </div>
      </div>}



      {selectedAgent && visible.agentContext && (
        <div className="absolute bottom-52 left-1/2 z-20 w-[420px] -translate-x-1/2 border border-border/30 bg-card/70 p-3 backdrop-blur-xl">
          <div className="mb-2 flex items-center justify-between">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-foreground/80">{AGENT_MAP_NODES[selectedAgent]?.flag} {AGENT_MAP_NODES[selectedAgent]?.label}</div>
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
        />
        {visible.mapControls && <div className="pointer-events-auto">
          <div
            className="flex select-none items-center gap-1 border border-border/30 bg-card/30 px-3 py-2 backdrop-blur-xl"
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
                togglePanelsRef.current.forEach((fn) => fn(next));
              }}
              className="flex h-7 w-7 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
              title={panelsCollapsed ? "Expand panels" : "Collapse panels"}
            >
              {panelsCollapsed ? <Columns2 className="h-3.5 w-3.5" /> : <Maximize className="h-3.5 w-3.5" />}
            </button>
          </div>
        </div>}

        {visible.timeline && <EventTimeline session={session} onTurnChange={setTimelineTurn} interactionFocus={interactionFocus} onInteractionFocus={setInteractionFocus} onRegisterToggle={(fn) => { togglePanelsRef.current[2] = fn; }} embedded onCollapsedChange={setTimelineCollapsed} />}
      </div>

      {visible.newsFeed && <NewsFeed items={newsItems} interactionFocus={interactionFocus} onInteractionFocus={setInteractionFocus} onRegisterToggle={(fn) => { togglePanelsRef.current[0] = fn; }} bottomOffset={sideBarBottom} />}
      {visible.activityLog && <ActivityLog items={activityItems} focusTurn={timelineTurn} interactionFocus={interactionFocus} onInteractionFocus={setInteractionFocus} onRegisterToggle={(fn) => { togglePanelsRef.current[1] = fn; }} bottomOffset={sideBarBottom} />}

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

function StatusPill({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-muted-foreground">{label}</span>
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

export type NewsItem = {
  id: string;
  source: string;
  summary: string;
  severity: number;
  timestamp: string;
  turn: number;
  agent: string | null;
  type: "event" | "intel";
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

function deriveNewsItems(session: SessionState | null): NewsItem[] {
  if (!session) return [];

  const items: NewsItem[] = [];
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
      });
    }
  }

  return items.slice(0, 30);
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


