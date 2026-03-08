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
  oversight_review: "REVIEW",
};

function AgentDot({ agent }: { agent: string }) {
  const color = AGENT_COLORS[agent] ?? "#888";
  return (
    <div
      className="h-2 w-2 shrink-0 rounded-full"
      style={{ backgroundColor: color }}
    />
  );
}

function TensionIndicator({ delta }: { delta: number }) {
  if (Math.abs(delta) < 0.1) return null;
  if (delta > 0) {
    return <TrendingUp className="h-3 w-3 text-primary" />;
  }
  return <TrendingDown className="h-3 w-3 text-secondary" />;
}

export function ActivityLog({ items, onRegisterToggle }: { items: ActivityItem[]; onRegisterToggle?: (fn: (collapsed: boolean) => void) => void }) {
  const [collapsed, setCollapsed] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  const applyCollapse = (next: boolean) => {
    setCollapsed(next);
    if (panelRef.current) {
      gsap.to(panelRef.current, {
        width: next ? COLLAPSED_WIDTH : EXPANDED_WIDTH,
        duration: 0.4,
        ease: "power2.inOut",
      });
    }
  };

  useEffect(() => {
    onRegisterToggle?.(applyCollapse);
  }, []);

  const toggle = () => applyCollapse(!collapsed);

  return (
    <div
      ref={panelRef}
      className="absolute top-6 bottom-6 right-4 z-20"
      style={{ width: EXPANDED_WIDTH }}
    >
      <div className="relative h-full rounded-md border-[0.75px] border-border/30 p-0">
        <GlowingEffect
          spread={40}
          glow={true}
          disabled={false}
          proximity={64}
          inactiveZone={0.01}
          borderWidth={2}
        />
        <div
          className="relative flex h-full overflow-hidden rounded-[inherit] bg-card/50 backdrop-blur-2xl"
          style={{
            boxShadow:
              "0 0 8px rgba(0,0,0,0.03), 0 2px 6px rgba(0,0,0,0.08), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 12px rgba(0,0,0,0.15)",
          }}
        >
          {/* Edge toggle bar — always visible, full height, LEFT side for right panel */}
          <button
            onClick={toggle}
            className="flex w-[48px] shrink-0 cursor-pointer flex-col items-center justify-center gap-2 border-r border-border/20 text-muted-foreground transition-colors hover:bg-muted/10 hover:text-foreground"
            title={collapsed ? "Expand panel" : "Collapse panel"}
          >
            {collapsed ? (
              <>
                <ChevronLeft className="h-4 w-4" />
                <span className="text-[9px] font-mono tracking-wider uppercase [writing-mode:vertical-lr]">
                  Activity
                </span>
              </>
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>

          {/* Content area — hidden when collapsed via overflow */}
          <div className={cn(
            "flex min-w-0 flex-1 flex-col transition-opacity duration-300",
            collapsed ? "opacity-0 pointer-events-none" : "opacity-100"
          )}>
            {/* Header */}
            <div className="flex items-center gap-2 border-b border-border/30 px-4 py-3">
              <Users className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span className="whitespace-nowrap text-[10px] font-semibold tracking-[0.2em] text-foreground/80 uppercase font-sans">
                Agent Activity
              </span>
              <span className="ml-auto text-[9px] font-mono text-muted-foreground">
                {items.length}
              </span>
            </div>

            {/* Scrollable list */}
            <div className="flex-1 overflow-y-auto scrollbar-thin">
              {items.length === 0 ? (
                <div className="flex h-full items-center justify-center p-6">
                  <p className="text-center text-xs font-mono text-muted-foreground">
                    No agent actions recorded.
                    <br />
                    Awaiting simulation step.
                  </p>
                </div>
              ) : (
                <div className="divide-y divide-border/20">
                  {items.map((item) => (
                    <div
                      key={item.id}
                      className="group px-4 py-3 transition-colors hover:bg-muted/20"
                    >
                      <div className="flex items-start gap-2.5">
                        <div className="mt-1 shrink-0">
                          <AgentDot agent={item.agent} />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="mb-1 flex items-center gap-2">
                            <span className="text-[10px] font-semibold uppercase tracking-wider text-foreground/80 font-sans">
                              {item.agent}
                            </span>
                            <ChevronRight className="h-2.5 w-2.5 text-muted-foreground" />
                            <span className="inline-flex items-center border border-border/50 px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider text-muted-foreground">
                              {ACTION_LABELS[item.actionType] ??
                                item.actionType.toUpperCase()}
                            </span>
                            {item.target && (
                              <>
                                <ChevronRight className="h-2.5 w-2.5 text-muted-foreground/50" />
                                <span className="text-[9px] font-mono text-accent">
                                  {item.target}
                                </span>
                              </>
                            )}
                          </div>

                          <p className="mb-1.5 text-xs leading-relaxed text-foreground/80 font-sans">
                            {item.summary}
                          </p>

                          <div className="flex items-center gap-3 text-[9px] font-mono text-muted-foreground">
                            <span>T{item.turn}</span>
                            {item.rewardTotal !== null && (
                              <span
                                className={cn(
                                  item.rewardTotal >= 0
                                    ? "text-secondary"
                                    : "text-primary"
                                )}
                              >
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
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
