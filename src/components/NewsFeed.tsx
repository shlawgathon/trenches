"use client";

import { useRef, useState, useEffect, useMemo, useCallback } from "react";
import { Activity, AlertTriangle, Radio, Zap, ChevronLeft, ChevronRight, Filter, X, ExternalLink } from "lucide-react";
import gsap from "gsap";
import { cn } from "@/src/lib/utils";
import { GlowingEffect } from "@/src/components/ui/glowing-effect";
import type { NewsItem } from "./GlobePage";

const COLLAPSED_WIDTH = 48;
const EXPANDED_WIDTH = 340;

const AGENT_COLORS: Record<string, string> = {
  us: "#e53935",
  israel: "#64b5f6",
  iran: "#689f38",
  hezbollah: "#ffa000",
  gulf: "#a1887f",
  oversight: "#b0b0b0",
};

const AGENT_LABELS: Record<string, string> = {
  us: "US",
  israel: "Israel",
  iran: "Iran",
  hezbollah: "Hezbollah",
  gulf: "Gulf",
  oversight: "Oversight",
};

const AGENT_FILTER_LABELS: Record<string, string> = {
  us: "🇺🇸",
  israel: "🇮🇱",
  iran: "🇮🇷",
  hezbollah: "Hez",
  gulf: "Gulf",
};

const SEVERITY_STYLES: Record<string, string> = {
  high: "border-primary/60 text-primary",
  medium: "border-chart-4/60 text-chart-4",
  low: "border-muted-foreground/40 text-muted-foreground",
};

function getSeverityLevel(severity: number): string {
  if (severity >= 0.7) return "high";
  if (severity >= 0.4) return "medium";
  return "low";
}

function SeverityIcon({ level }: { level: string }) {
  const iconClass = "h-3 w-3";
  switch (level) {
    case "high":
      return <AlertTriangle className={cn(iconClass, "text-primary")} />;
    case "medium":
      return <Zap className={cn(iconClass, "text-chart-4")} />;
    default:
      return <Radio className={cn(iconClass, "text-muted-foreground")} />;
  }
}

function formatRelativeTime(date: Date): string {
  const now = Date.now();
  const diffS = Math.floor((now - date.getTime()) / 1000);
  if (diffS < 10) return "just now";
  if (diffS < 60) return `${diffS}s ago`;
  const diffM = Math.floor(diffS / 60);
  if (diffM < 60) return `${diffM}m ago`;
  const diffH = Math.floor(diffM / 60);
  return `${diffH}h ago`;
}

type InteractionFocus = {
  turn: number;
  agent: string | null;
};

export function NewsFeed({
  items,
  onRegisterToggle,
  onCollapsedChange,
  interactionFocus,
  onInteractionFocus,
  bottomOffset = 80,
}: {
  items: NewsItem[];
  onRegisterToggle?: (fn: (collapsed: boolean) => void) => void;
  onCollapsedChange?: (collapsed: boolean) => void;
  interactionFocus?: InteractionFocus | null;
  onInteractionFocus?: (focus: InteractionFocus | null) => void;
  bottomOffset?: number;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [filterAgent, setFilterAgent] = useState<string | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const prevItemIdsRef = useRef<Set<string>>(new Set());
  const onCollapsedChangeRef = useRef(onCollapsedChange);
  onCollapsedChangeRef.current = onCollapsedChange;

  const applyCollapse = useRef((next: boolean) => {
    setCollapsed(next);
    onCollapsedChangeRef.current?.(next);
    if (panelRef.current) {
      gsap.to(panelRef.current, {
        width: next ? COLLAPSED_WIDTH : EXPANDED_WIDTH,
        duration: 0.4,
        ease: "power2.inOut",
      });
    }
  }).current;

  useEffect(() => {
    onRegisterToggle?.(applyCollapse);
  }, []);

  const toggle = () => applyCollapse(!collapsed);

  const initialBottomRef = useRef(bottomOffset);
  const hasMountedRef = useRef(false);

  useEffect(() => {
    if (!panelRef.current) return;
    if (!hasMountedRef.current) {
      hasMountedRef.current = true;
      return;
    }
    gsap.to(panelRef.current, {
      bottom: bottomOffset,
      duration: 0.4,
      ease: "power2.inOut",
    });
  }, [bottomOffset]);

  // Unique agents in current items (for filter buttons)
  const agents = useMemo(() => {
    const set = new Set<string>();
    for (const item of items) {
      if (item.agent) set.add(item.agent);
    }
    return [...set].sort();
  }, [items]);

  // Filtered items
  const filteredItems = filterAgent
    ? items.filter((item) => item.agent === filterAgent)
    : items;

  // Animate new items — first slides from top, rest push down naturally
  useEffect(() => {
    if (!listRef.current || filteredItems.length === 0) return;
    const prevIds = prevItemIdsRef.current;
    const newIds = filteredItems.map((i) => i.id).filter((id) => !prevIds.has(id));
    if (newIds.length > 0 && newIds[0]) {
      // Only animate the first (newest / top-most) item — rest push down via DOM
      const firstEl = listRef.current?.querySelector(`[data-item-id="${CSS.escape(newIds[0])}"]`);
      if (firstEl) {
        gsap.fromTo(
          firstEl,
          { opacity: 0, y: -30, height: 0, paddingTop: 0, paddingBottom: 0 },
          { opacity: 1, y: 0, height: "auto", paddingTop: "", paddingBottom: "", duration: 0.4, ease: "power2.out" }
        );
      }
      listRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    }
    prevItemIdsRef.current = new Set(filteredItems.map((i) => i.id));
  }, [filteredItems]);

  return (
    <div ref={panelRef} className="absolute top-4 left-4 z-20 select-none" style={{ width: EXPANDED_WIDTH, bottom: initialBottomRef.current }}>
      <div className="relative h-full rounded-md border-[0.75px] border-border/30 p-0">
        <GlowingEffect spread={40} glow={true} disabled={false} proximity={64} inactiveZone={0.01} borderWidth={2} />
        <div
          className="relative flex h-full overflow-hidden rounded-[inherit] bg-card/25 backdrop-blur-lg"
          style={{
            boxShadow:
              "0 0 8px rgba(0,0,0,0.03), 0 2px 6px rgba(0,0,0,0.08), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 12px rgba(0,0,0,0.15)",
          }}
        >
          <div className={cn("flex min-w-0 flex-1 flex-col transition-opacity duration-300", collapsed ? "opacity-0 pointer-events-none" : "opacity-100")}>
            {/* Header */}
            <div className="flex items-center gap-2 border-b border-border/30 px-4 py-3">
              <Activity className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span className="whitespace-nowrap text-[10px] font-semibold tracking-[0.2em] text-foreground/80 uppercase font-sans">Live Intel Feed</span>
              <span className="ml-auto text-[9px] font-mono text-muted-foreground">{filteredItems.length}</span>
            </div>

            {/* Agent filter bar */}
            <div className="flex items-center gap-1 border-b border-border/20 px-3 py-1.5 overflow-x-auto">
                {(["us", "israel", "iran", "hezbollah", "gulf"] as const).map((a) => {
                  const color = AGENT_COLORS[a] ?? "#888";
                  const active = filterAgent === a;
                  return (
                    <button
                      key={a}
                      onClick={() => setFilterAgent(active ? null : a)}
                      className={cn(
                        "flex items-center gap-1 px-1.5 py-0.5 text-[8px] font-mono uppercase tracking-wider transition-colors cursor-pointer rounded-sm",
                        active
                          ? "text-foreground"
                          : "text-muted-foreground hover:text-foreground"
                      )}
                      style={active ? { backgroundColor: `${color}20`, borderBottom: `1px solid ${color}` } : undefined}
                    >
                      <div className="h-1.5 w-1.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                      {AGENT_FILTER_LABELS[a] ?? a}
                    </button>
                  );
                })}
                {filterAgent && (
                  <button
                    onClick={() => setFilterAgent(null)}
                    className="ml-auto cursor-pointer text-muted-foreground hover:text-foreground"
                    title="Clear filter"
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>

            {/* Items */}
            <div ref={listRef} className="flex-1 overflow-y-auto scrollbar-thin">
              {filteredItems.length === 0 ? (
                <div className="flex h-full items-center justify-center p-6">
                  <p className="text-center text-xs font-mono text-muted-foreground">No events yet.<br />Waiting for intel stream...</p>
                </div>
              ) : (
                <div className="divide-y divide-border/20">
                  {filteredItems.map((item) => {
                    const level = getSeverityLevel(item.severity);
                    const agentColor = item.agent ? AGENT_COLORS[item.agent] ?? "#888" : "#888";
                    const agentLabel = item.agent ? (AGENT_LABELS[item.agent] ?? item.agent) : "Unknown";
                    const highlighted =
                      interactionFocus &&
                      item.turn === interactionFocus.turn &&
                      (!interactionFocus.agent || item.agent === interactionFocus.agent);

                    // Format timestamp
                    const ts = item.timestamp ? new Date(item.timestamp) : null;
                    const timeLabel = ts ? formatRelativeTime(ts) : "";

                    return (
                      <div
                        key={item.id}
                        data-item-id={item.id}
                        className={cn(
                          "group px-4 py-3 transition-colors hover:bg-muted/20 cursor-pointer",
                          highlighted && "bg-primary/10 ring-1 ring-primary/40"
                        )}
                        onMouseEnter={() => onInteractionFocus?.({ turn: item.turn, agent: item.agent ?? null })}
                        onMouseLeave={() => onInteractionFocus?.(null)}
                        onClick={() => onInteractionFocus?.({ turn: item.turn, agent: item.agent ?? null })}
                      >
                        <div className="flex items-start gap-2.5">
                          <div className="mt-0.5 shrink-0"><SeverityIcon level={level} /></div>
                          <div className="min-w-0 flex-1">
                            <div className="mb-1 flex items-center gap-2">
                              <span className={cn("inline-flex items-center border px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider", SEVERITY_STYLES[level])}>{item.type === "event" ? "EVENT" : "INTEL"}</span>
                              {/* Agent label */}
                              <span className="inline-flex items-center gap-1 text-[9px] font-mono">
                                <span className="h-1.5 w-1.5 rounded-full inline-block shrink-0" style={{ backgroundColor: agentColor }} />
                                <span style={{ color: agentColor }}>{agentLabel}</span>
                              </span>
                              <span className="ml-auto text-[9px] font-mono text-muted-foreground">T{item.turn}</span>
                            </div>
                            {item.url ? (
                              <a href={item.url} target="_blank" rel="noopener noreferrer" className="group/link flex items-start gap-1 text-xs leading-relaxed text-foreground/90 font-sans hover:text-primary transition-colors">
                                <span className="flex-1">{item.summary}</span>
                                <ExternalLink className="h-2.5 w-2.5 mt-0.5 shrink-0 text-muted-foreground/40 group-hover/link:text-primary transition-colors" />
                              </a>
                            ) : (
                              <p className="text-xs leading-relaxed text-foreground/90 font-sans">{item.summary}</p>
                            )}
                            <div className="mt-1 flex items-center gap-2 text-[8px] font-mono text-muted-foreground/60">
                              <span className="truncate">{item.source}</span>
                              {item.translateUrl && (
                                <a href={item.translateUrl} target="_blank" rel="noopener noreferrer" className="shrink-0 text-primary/50 hover:text-primary transition-colors">[translate]</a>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          <button
            onClick={toggle}
            className="flex w-[48px] shrink-0 cursor-pointer flex-col items-center justify-center gap-2 border-l border-border/20 text-muted-foreground transition-colors hover:bg-muted/10 hover:text-foreground"
            title={collapsed ? "Expand panel" : "Collapse panel"}
          >
            {collapsed ? (
              <>
                <ChevronRight className="h-4 w-4" />
                <span className="text-[9px] font-mono tracking-wider uppercase [writing-mode:vertical-lr]">Intel</span>
              </>
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

