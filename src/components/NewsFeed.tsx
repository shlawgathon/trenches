"use client";

import { useRef, useState, useEffect } from "react";
import { Activity, AlertTriangle, Radio, Zap, ChevronLeft, ChevronRight } from "lucide-react";
import gsap from "gsap";
import { cn } from "@/src/lib/utils";
import { GlowingEffect } from "@/src/components/ui/glowing-effect";
import type { NewsItem } from "./GlobePage";

const COLLAPSED_WIDTH = 48;
const EXPANDED_WIDTH = 340;

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

export function NewsFeed({ items, onRegisterToggle }: { items: NewsItem[]; onRegisterToggle?: (fn: (collapsed: boolean) => void) => void }) {
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
      className="absolute top-6 bottom-6 left-4 z-20"
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
          {/* Content area — hidden when collapsed via overflow */}
          <div className={cn(
            "flex min-w-0 flex-1 flex-col transition-opacity duration-300",
            collapsed ? "opacity-0 pointer-events-none" : "opacity-100"
          )}>
            {/* Header */}
            <div className="flex items-center gap-2 border-b border-border/30 px-4 py-3">
              <Activity className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span className="whitespace-nowrap text-[10px] font-semibold tracking-[0.2em] text-foreground/80 uppercase font-sans">
                Incoming Intel
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
                    No intelligence feed active.
                    <br />
                    Awaiting backend session.
                  </p>
                </div>
              ) : (
                <div className="divide-y divide-border/20">
                  {items.map((item) => {
                    const level = getSeverityLevel(item.severity);
                    return (
                      <div
                        key={item.id}
                        className="group px-4 py-3 transition-colors hover:bg-muted/20"
                      >
                        <div className="flex items-start gap-2.5">
                          <div className="mt-0.5 shrink-0">
                            <SeverityIcon level={level} />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="mb-1 flex items-center gap-2">
                              <span
                                className={cn(
                                  "inline-flex items-center border px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider",
                                  SEVERITY_STYLES[level]
                                )}
                              >
                                {item.type === "event" ? "EVENT" : "INTEL"}
                              </span>
                              <span className="truncate text-[9px] font-mono text-muted-foreground">
                                {item.source}
                              </span>
                            </div>
                            <p className="text-xs leading-relaxed text-foreground/90 font-sans">
                              {item.summary}
                            </p>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Edge toggle bar — always visible, full height */}
          <button
            onClick={toggle}
            className="flex w-[48px] shrink-0 cursor-pointer flex-col items-center justify-center gap-2 border-l border-border/20 text-muted-foreground transition-colors hover:bg-muted/10 hover:text-foreground"
            title={collapsed ? "Expand panel" : "Collapse panel"}
          >
            {collapsed ? (
              <>
                <ChevronRight className="h-4 w-4" />
                <span className="text-[9px] font-mono tracking-wider uppercase [writing-mode:vertical-lr]">
                  Intel
                </span>
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
