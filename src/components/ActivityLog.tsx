"use client";

import { useRef, useState, useEffect } from "react";
import {
  Users,
  ChevronLeft,
  ChevronRight,
  Shield,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import gsap from "gsap";
import { cn } from "@/src/lib/utils";
import { GlowingEffect } from "@/src/components/ui/glowing-effect";
import type { ActivityItem } from "./GlobePage";

const COLLAPSED_WIDTH = 48;
const EXPANDED_WIDTH = 360;

const AGENT_COLORS: Record<string, string> = {
  us: "#e53935",
  israel: "#64b5f6",
  iran: "#689f38",
  hezbollah: "#ffa000",
  gulf: "#a1887f",
  oversight: "#b0b0b0",
};

const ACTION_LABELS: Record<string, string> = {
  negotiate: "NEGOTIATE",
  defend: "DEFEND",
  intel_query: "INTEL",
  hold: "HOLD",
  strike: "STRIKE",
  sanction: "SANCTION",
  mobilize: "MOBILIZE",
  deceive: "DECEIVE",
  oversight_review: "PREDICT",
};

type InteractionFocus = {
  turn: number;
  agent: string | null;
};

function AgentDot({ agent }: { agent: string }) {
  const color = AGENT_COLORS[agent] ?? "#888";
  return <div className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: color }} />;
}

function TensionIndicator({ delta }: { delta: number }) {
  if (Math.abs(delta) < 0.1) return null;
  if (delta > 0) {
    return <TrendingUp className="h-3 w-3 text-primary" />;
  }
  return <TrendingDown className="h-3 w-3 text-secondary" />;
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

export function ActivityLog({
  items,
  focusTurn,
  onRegisterToggle,
  onCollapsedChange,
  interactionFocus,
  onInteractionFocus,
  bottomOffset = 80,
}: {
  items: ActivityItem[];
  focusTurn?: number;
  onRegisterToggle?: (fn: (collapsed: boolean) => void) => void;
  onCollapsedChange?: (collapsed: boolean) => void;
  interactionFocus?: InteractionFocus | null;
  onInteractionFocus?: (focus: InteractionFocus | null) => void;
  bottomOffset?: number;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const visibleItems = focusTurn === undefined ? items : items.filter((item) => item.turn <= focusTurn);
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

  // Animate new items — first slides from top, rest push down naturally
  useEffect(() => {
    if (!listRef.current || visibleItems.length === 0) return;
    const prevIds = prevItemIdsRef.current;
    const newIds = visibleItems.map((i) => i.id).filter((id) => !prevIds.has(id));
    if (newIds.length > 0 && newIds[0]) {
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
    prevItemIdsRef.current = new Set(visibleItems.map((i) => i.id));
  }, [visibleItems]);

  return (
    <div ref={panelRef} className="absolute top-4 right-4 z-20 select-none" style={{ width: EXPANDED_WIDTH, bottom: initialBottomRef.current }}>
      <div className="relative h-full rounded-md border-[0.75px] border-border/30 p-0">
        <GlowingEffect spread={40} glow={true} disabled={false} proximity={64} inactiveZone={0.01} borderWidth={2} />
        <div
          className="relative flex h-full overflow-hidden rounded-[inherit] bg-card/25 backdrop-blur-lg"
          style={{
            boxShadow:
              "0 0 8px rgba(0,0,0,0.03), 0 2px 6px rgba(0,0,0,0.08), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 12px rgba(0,0,0,0.15)",
          }}
        >
          <button
            onClick={toggle}
            className="flex w-[48px] shrink-0 cursor-pointer flex-col items-center justify-center gap-2 border-r border-border/20 text-muted-foreground transition-colors hover:bg-muted/10 hover:text-foreground"
            title={collapsed ? "Expand panel" : "Collapse panel"}
          >
            {collapsed ? (
              <>
                <ChevronLeft className="h-4 w-4" />
                <span className="text-[9px] font-mono tracking-wider uppercase [writing-mode:vertical-lr]">Entity Activity</span>
              </>
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>

          <div className={cn("flex min-w-0 flex-1 flex-col transition-opacity duration-300", collapsed ? "opacity-0 pointer-events-none" : "opacity-100")}>
            <div className="flex items-center gap-2 border-b border-border/30 px-4 py-3">
              <Users className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span className="whitespace-nowrap text-[10px] font-semibold tracking-[0.2em] text-foreground/80 uppercase font-sans">Entity Activity</span>
              <span className="ml-auto text-[9px] font-mono text-muted-foreground">{visibleItems.length}</span>
            </div>

            <div ref={listRef} className="flex-1 overflow-y-auto scrollbar-thin">
              {visibleItems.length === 0 ? (
                <div className="flex h-full items-center justify-center p-6">
                  <p className="text-center text-xs font-mono text-muted-foreground">No entity actions recorded.<br />Awaiting simulation step.</p>
                </div>
              ) : (
                <div className="divide-y divide-border/20">
                  {visibleItems.map((item) => {
                    const highlighted =
                      interactionFocus &&
                      item.turn === interactionFocus.turn &&
                      (!interactionFocus.agent || item.agent === interactionFocus.agent);
                    const ts = item.timestamp ? new Date(item.timestamp) : null;
                    const timeLabel = ts ? formatRelativeTime(ts) : "";

                    return (
                      <div
                        key={item.id}
                        data-item-id={item.id}
                        className={cn("group px-4 py-3 transition-colors hover:bg-muted/20 cursor-pointer", highlighted && "bg-primary/10 ring-1 ring-primary/40")}
                        onMouseEnter={() => onInteractionFocus?.({ turn: item.turn, agent: item.agent })}
                        onMouseLeave={() => onInteractionFocus?.(null)}
                        onClick={() => onInteractionFocus?.({ turn: item.turn, agent: item.agent })}
                      >
                        <div className="flex items-start gap-2.5">
                          <div className="mt-1 shrink-0"><AgentDot agent={item.agent} /></div>
                          <div className="min-w-0 flex-1">
                            <div className="mb-1 flex items-center gap-2">
                              <span className="text-[10px] font-semibold uppercase tracking-wider text-foreground/80 font-sans">{item.agent}</span>
                              <ChevronRight className="h-2.5 w-2.5 text-muted-foreground" />
                              <span className="inline-flex items-center border border-border/50 px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider text-muted-foreground">
                                {ACTION_LABELS[item.actionType] ?? item.actionType.toUpperCase()}
                              </span>
                              {item.target && (
                                <>
                                  <ChevronRight className="h-2.5 w-2.5 text-muted-foreground/50" />
                                  <span className="text-[9px] font-mono text-accent">{item.target}</span>
                                </>
                              )}
                            </div>

                            <p className="mb-1.5 text-xs leading-relaxed text-foreground/80 font-sans">{item.summary}</p>

                            <div className="flex items-center gap-3 text-[9px] font-mono text-muted-foreground">
                              <span>T{item.turn}</span>
                              {timeLabel && <span className="text-muted-foreground/60">{timeLabel}</span>}
                              {item.rewardTotal !== null && (
                                <span className={cn(item.rewardTotal >= 0 ? "text-secondary" : "text-primary")}>
                                  {item.rewardTotal >= 0 ? "+" : ""}
                                  {item.rewardTotal.toFixed(2)}
                                </span>
                              )}
                              <TensionIndicator delta={item.tensionDelta} />
                              {item.oversightTriggered && (
                                <div className="flex items-center gap-1 text-primary">
                                  <Shield className="h-2.5 w-2.5" />
                                  <span>OVERSIGHT</span>
                                </div>
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
        </div>
      </div>
    </div>
  );
}
