"use client";

import { useEffect, useRef, useState } from "react";
import { ZoomIn, ZoomOut, RotateCcw, Columns2, Maximize, MessageSquare } from "lucide-react";
import mapboxgl from "mapbox-gl";
import { NewsFeed } from "@/src/components/NewsFeed";
import { ActivityLog } from "@/src/components/ActivityLog";
import { ChatPanel } from "@/src/components/ChatPanel";
import { cn } from "@/src/lib/utils";
import type { SessionState } from "@/src/lib/types";
import { bootstrapPlatform } from "@/src/app/bootstrap";

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

export default function GlobePage() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const [session, setSession] = useState<SessionState | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [panelsCollapsed, setPanelsCollapsed] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const togglePanelsRef = useRef<Array<(collapsed: boolean) => void>>([]);

  // Initialize Mapbox globe — run first, independently
  useEffect(() => {
    const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
    if (!token) {
      setMapError("NEXT_PUBLIC_MAPBOX_TOKEN not set");
      return;
    }

    const container = mapContainerRef.current;
    if (!container) {
      setMapError("Map container not available");
      return;
    }

    // Prevent double init in strict mode
    if (mapRef.current) return;

    try {
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

      map.on("style.load", () => {
        map.setFog({
          color: "rgba(20, 20, 20, 0.95)",
          "high-color": "rgba(30, 30, 30, 0.4)",
          "space-color": "rgba(10, 10, 10, 1)",
          "star-intensity": 0.06,
          range: [-1, 2],
        });

        const paintUpdates: Array<{
          id: string;
          prop: string;
          value: string | number;
        }> = [
          { id: "background", prop: "background-color", value: "#111111" },
          { id: "land", prop: "fill-color", value: "#1e1e1e" },
          { id: "water", prop: "fill-color", value: "#111111" },
          { id: "waterway", prop: "line-color", value: "#2a2a2a" },
          {
            id: "country-label",
            prop: "text-color",
            value: "rgba(200, 200, 200, 0.3)",
          },
          {
            id: "marine-label",
            prop: "text-color",
            value: "rgba(120, 120, 120, 0.25)",
          },
        ];

        for (const layerId of INTEL_HIDDEN_LAYERS) {
          if (map.getLayer(layerId)) {
            map.setLayoutProperty(layerId, "visibility", "none");
          }
        }

        for (const update of paintUpdates) {
          if (!map.getLayer(update.id)) continue;
          try {
            map.setPaintProperty(
              update.id,
              update.prop as never,
              update.value as never
            );
          } catch {
            continue;
          }
        }

        // Paint admin boundaries: countries red, states white
        const allLayers = map.getStyle().layers ?? [];
        for (const layer of allLayers) {
          if (!layer.id.includes("admin") && !layer.id.includes("boundar")) continue;
          try {
            if (layer.type === "line") {
              if (layer.id.includes("1")) {
                // State/province borders → white
                map.setPaintProperty(layer.id, "line-color", "rgba(255, 255, 255, 0.35)");
                map.setPaintProperty(layer.id, "line-width", 0.5);
              } else {
                // Country borders → red
                map.setPaintProperty(layer.id, "line-color", "#e53935");
                map.setPaintProperty(layer.id, "line-opacity", 0.6);
                map.setPaintProperty(layer.id, "line-width", 0.8);
              }
            } else if (layer.type === "fill") {
              map.setPaintProperty(layer.id, "fill-opacity", 0);
            }
          } catch {
            continue;
          }
        }
      });

      map.on("error", (e) => {
        console.error("[Mapbox error]", e.error?.message || e);
      });
    } catch (err) {
      console.error("[Map init error]", err);
      setMapError(err instanceof Error ? err.message : "Map failed to load");
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // Bootstrap session — separate from map
  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        const rt = await bootstrapPlatform();
        if (cancelled) return;
        if (rt.backendStatus === "healthy") {
          const sess = await rt.sessionClient.createSession({ seed: 7 });
          if (!cancelled) setSession(sess);
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

  const newsItems = deriveNewsItems(session);
  const activityItems = deriveActivityItems(session);

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-background">
      {/* Fullscreen Mapbox Globe */}
      <div
        ref={mapContainerRef}
        className="absolute inset-0"
        style={{ width: "100vw", height: "100vh" }}
      ></div>

      {/* Top center title */}
      <div className="pointer-events-none absolute top-6 left-1/2 z-20 -translate-x-1/2">
        <div
          className="pointer-events-auto flex select-none items-center gap-4 border border-border/40 bg-card/60 px-5 py-2.5 font-sans backdrop-blur-xl"
          style={{
            boxShadow:
              "0 0 8px rgba(0,0,0,0.03), 0 2px 6px rgba(0,0,0,0.08), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 12px rgba(0,0,0,0.15)",
          }}
        >
          <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
          <span className="text-xs font-semibold tracking-[0.2em] text-foreground/80 uppercase">
            Trenches
          </span>
          <span className="text-[10px] font-mono text-muted-foreground">
            {session ? `T${session.world.turn}` : "IDLE"}
          </span>
          {session && (
            <>
              <div className="mx-1 h-4 w-px bg-border/40" />
              <div className="flex items-center gap-4 font-mono text-xs">
                <StatusPill label="TENSION" value={session.world.tension_level.toFixed(0)} warn={session.world.tension_level > 60} />
                <StatusPill label="MARKET" value={session.world.market_stress.toFixed(0)} warn={session.world.market_stress > 60} />
                <StatusPill label="OIL" value={session.world.oil_pressure.toFixed(0)} warn={session.world.oil_pressure > 60} />
                <StatusPill label="EVENTS" value={String(session.world.active_events.length)} />
                <StatusPill label="LIVE" value={session.live.enabled ? "ON" : "OFF"} />
              </div>
            </>
          )}
        </div>
      </div>



      {/* Chat Panel */}
      <ChatPanel
        open={chatOpen}
        onClose={() => setChatOpen(false)}
        sessionId={session?.session_id ?? null}
      />

      {/* Bottom center map controls */}
      <div className="pointer-events-none absolute bottom-6 left-1/2 z-20 -translate-x-1/2">
        <div
          className="pointer-events-auto flex select-none items-center gap-1 border border-border/30 bg-card/30 px-3 py-2 backdrop-blur-xl"
          style={{
            boxShadow:
              "0 0 8px rgba(0,0,0,0.03), 0 2px 6px rgba(0,0,0,0.08), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 12px rgba(0,0,0,0.15)",
          }}
        >
          <button
            onClick={() => mapRef.current?.zoomIn({ duration: 300 })}
            className="flex h-7 w-7 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
            title="Zoom in"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => mapRef.current?.zoomOut({ duration: 300 })}
            className="flex h-7 w-7 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
            title="Zoom out"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <div className="mx-1 h-4 w-px bg-border/40" />
          <button
            onClick={() => setChatOpen((prev) => !prev)}
            className={cn(
              "flex h-7 w-7 cursor-pointer items-center justify-center transition-colors",
              chatOpen
                ? "text-primary"
                : "text-muted-foreground hover:text-foreground"
            )}
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
            {panelsCollapsed ? (
              <Columns2 className="h-3.5 w-3.5" />
            ) : (
              <Maximize className="h-3.5 w-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Left floating panel — News Feed */}
      <NewsFeed items={newsItems} onRegisterToggle={(fn) => { togglePanelsRef.current[0] = fn; }} />

      {/* Right floating panel — Activity Log */}
      <ActivityLog items={activityItems} onRegisterToggle={(fn) => { togglePanelsRef.current[1] = fn; }} />

      {/* Fallback when no mapbox token or error */}
      {mapError && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-background">
          <div className="border border-border/40 bg-card/80 p-8 text-center backdrop-blur-xl">
            <p className="text-sm font-mono text-muted-foreground uppercase tracking-wider">
              Map error:{" "}
              <code className="text-primary">{mapError}</code>
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
}: {
  label: string;
  value: string;
  warn?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-muted-foreground">{label}</span>
      <span className={warn ? "text-primary font-bold" : "text-foreground"}>
        {value}
      </span>
    </div>
  );
}

// --- Data derivation helpers ---

export type NewsItem = {
  id: string;
  source: string;
  summary: string;
  severity: number;
  timestamp: string;
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
      type: "event",
    });
  }

  const agents = Object.keys(session.observations);
  for (const agentId of agents) {
    const obs = session.observations[agentId];
    if (!obs) continue;
    for (const brief of obs.public_brief) {
      items.push({
        id: `${agentId}-pub-${brief.source}-${brief.summary.slice(0, 20)}`,
        source: brief.source,
        summary: brief.summary,
        severity: brief.confidence,
        timestamp: session.updated_at,
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
    const agents = Object.keys(trace.actions);
    for (const agentId of agents) {
      const action = trace.actions[agentId];
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

  return items.slice(0, 50);
}
