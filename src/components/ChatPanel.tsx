"use client";

import { useRef, useState, useEffect, useMemo } from "react";
import {
  Send,
  X,
  Loader2,
  Zap,
  Trash2,
  RotateCcw,
  Edit3,
  Check,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
} from "lucide-react";
import gsap from "gsap";
import { cn } from "@/src/lib/utils";
import { GlowingEffect } from "@/src/components/ui/glowing-effect";
import type { SessionState, ExternalSignal } from "@/src/lib/types";

/* ── Types ── */

type Message = {
  id: string;
  role: "user" | "assistant" | "system" | "injection";
  content: string;
  timestamp: number;
  syntheticEventId?: string;
};

export type SyntheticEvent = {
  id: string;
  signal: ExternalSignal;
  injectedAtTurn: number;
  timestamp: number;
  removed?: boolean;
};

interface ChatPanelProps {
  open: boolean;
  onClose: () => void;
  sessionId?: string | null;
  session?: SessionState | null;
  syntheticEvents?: SyntheticEvent[];
  onInjectEvent?: (signal: ExternalSignal) => Promise<void>;
  onRemoveEvent?: (eventId: string) => void;
  onRewindToEvent?: (eventId: string) => void;
  onHeaderMouseDown?: () => void;
  offset?: { x: number; y: number };
}

/* ── Helpers ── */

function buildContextBrief(session: SessionState | null | undefined): string {
  if (!session) return "";
  const w = session.world;
  const turn = w.turn;
  const tension = w.tension_level?.toFixed(1) ?? "?";
  const market = w.market_stress?.toFixed(1) ?? "?";
  const oil = w.oil_pressure?.toFixed(1) ?? "?";

  const lines: string[] = [
    `[T${turn}] TENSION=${tension} MARKET=${market} OIL=${oil}`,
  ];

  // Active events
  if (w.active_events?.length) {
    lines.push(`ACTIVE EVENTS (${w.active_events.length}):`);
    for (const evt of w.active_events.slice(0, 5)) {
      lines.push(`  • [sev ${evt.severity.toFixed(1)}] ${evt.summary} (${evt.source})`);
    }
  }

  // Recent actions from traces
  if (session.recent_traces?.length) {
    const latest = session.recent_traces[session.recent_traces.length - 1];
    if (latest) {
      const actionParts: string[] = [];
      for (const [agentId, action] of Object.entries(latest.actions)) {
        if (!action) continue;
        actionParts.push(`${agentId}→${action.type}${action.target ? `→${action.target}` : ""}: ${action.summary}`);
      }
      if (actionParts.length) {
        lines.push(`LAST ACTIONS (T${latest.turn}):`);
        for (const part of actionParts) {
          lines.push(`  • ${part}`);
        }
      }
    }
  }

  // Agent observations summary
  const obsKeys = Object.keys(session.observations ?? {});
  if (obsKeys.length) {
    const briefs: string[] = [];
    for (const agentId of obsKeys) {
      const obs = session.observations[agentId];
      if (obs?.public_brief?.length) {
        briefs.push(`${agentId}: ${obs.public_brief[0]?.summary ?? "—"}`);
      }
    }
    if (briefs.length) {
      lines.push(`INTEL BRIEFS:`);
      for (const b of briefs.slice(0, 4)) {
        lines.push(`  • ${b}`);
      }
    }
  }

  // Risk scores
  if (w.risk_scores && Object.keys(w.risk_scores).length) {
    const riskParts = Object.entries(w.risk_scores)
      .map(([a, s]) => `${a}=${(s as number).toFixed(2)}`)
      .join(" ");
    lines.push(`RISK: ${riskParts}`);
  }

  return lines.join("\n");
}

function parseInjectCommand(text: string): { headline: string; severity: number; region?: string } | null {
  const match = text.match(/^\/inject\s+(.+)/i);
  if (!match) return null;
  const headline = match[1]!.trim();
  // Extract optional severity tag like [0.8] or [high]
  const sevMatch = headline.match(/\[(\d*\.?\d+)\]\s*/);
  let severity = 0.7;
  let cleanHeadline = headline;
  if (sevMatch) {
    severity = Math.min(1, Math.max(0, parseFloat(sevMatch[1]!)));
    cleanHeadline = headline.replace(sevMatch[0], "").trim();
  } else if (/\[high\]/i.test(headline)) {
    severity = 0.9;
    cleanHeadline = headline.replace(/\[high\]/i, "").trim();
  } else if (/\[low\]/i.test(headline)) {
    severity = 0.3;
    cleanHeadline = headline.replace(/\[low\]/i, "").trim();
  }
  // Extract optional region tag like {middle_east}
  const regionMatch = cleanHeadline.match(/\{([^}]+)\}/);
  let region: string | undefined;
  if (regionMatch) {
    region = regionMatch[1]!.trim();
    cleanHeadline = cleanHeadline.replace(regionMatch[0], "").trim();
  }
  return { headline: cleanHeadline, severity, region };
}

/* ── Component ── */

export function ChatPanel({
  open,
  onClose,
  sessionId,
  session,
  syntheticEvents = [],
  onInjectEvent,
  onRemoveEvent,
  onRewindToEvent,
  onHeaderMouseDown,
}: ChatPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "sys-0",
      role: "system",
      content:
        "OVERSIGHT CHANNEL — Direct line to the simulation oversight.\n" +
        "• Chat normally to query world state\n" +
        "• /inject <headline> to add a synthetic event\n" +
        "  Options: [0.8] severity · {region} · [high] [low]",
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [eventsExpanded, setEventsExpanded] = useState(true);
  const [editingEventId, setEditingEventId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");

  // GSAP open/close animation
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (open) setVisible(true);
  }, [open]);

  useEffect(() => {
    if (!panelRef.current) return;
    if (open && visible) {
      gsap.fromTo(
        panelRef.current,
        { opacity: 0, scale: 0.95, y: 10, backdropFilter: "blur(0px)" },
        { opacity: 1, scale: 1, y: 0, backdropFilter: "blur(16px)", duration: 0.35, ease: "power3.out" }
      );
      setTimeout(() => inputRef.current?.focus(), 350);
    } else if (!open && visible) {
      gsap.to(panelRef.current, {
        opacity: 0, scale: 0.95, y: 10, backdropFilter: "blur(0px)",
        duration: 0.25, ease: "power2.in",
        onComplete: () => setVisible(false),
      });
    }
  }, [open, visible]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Live context brief for display
  const contextBrief = useMemo(() => buildContextBrief(session), [session]);

  // Active (not removed) synthetic events
  const activeEvents = syntheticEvents.filter((e) => !e.removed);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const parsed = parseInjectCommand(text);

    if (parsed && onInjectEvent) {
      // ── Injection mode ──
      const userMsg: Message = {
        id: `inject-cmd-${Date.now()}`,
        role: "user",
        content: text,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setLoading(true);

      try {
        const signal: ExternalSignal = {
          source: "oversight_injection",
          headline: parsed.headline,
          severity: parsed.severity,
          region: parsed.region ?? null,
          tags: ["synthetic", "oversight"],
        };
        await onInjectEvent(signal);

        const confirmMsg: Message = {
          id: `inject-ok-${Date.now()}`,
          role: "injection",
          content: `⚡ EVENT INJECTED [sev ${parsed.severity.toFixed(1)}]: ${parsed.headline}`,
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, confirmMsg]);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: `inject-err-${Date.now()}`,
            role: "assistant",
            content: `Injection failed: ${err instanceof Error ? err.message : "unknown error"}`,
            timestamp: Date.now(),
          },
        ]);
      } finally {
        setLoading(false);
      }
      return;
    }

    // ── Normal chat mode with full context ──
    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const assistantMsg: Message = {
        id: `asst-${Date.now()}`,
        role: "assistant",
        content: generateContextualResponse(text, contextBrief, activeEvents),
        timestamp: Date.now(),
      };
      await new Promise((r) => setTimeout(r, 300 + Math.random() * 200));
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "assistant",
          content: "Connection error. Unable to process.",
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  };

  if (!visible) return null;

  const isInjectMode = input.trimStart().startsWith("/inject");

  return (
    <div ref={panelRef} className="pointer-events-auto relative z-30 w-[540px]">
      <div className="relative h-full rounded-md border-[0.75px] border-border/30 p-0">
        <GlowingEffect spread={40} glow={true} disabled={false} proximity={64} inactiveZone={0.01} borderWidth={2} />
      <div
        className="pointer-events-auto relative flex h-[420px] flex-col overflow-hidden rounded-[inherit] bg-card/40 backdrop-blur-lg"
        style={{
          boxShadow:
            "0 0 8px rgba(0,0,0,0.03), 0 4px 12px rgba(0,0,0,0.15), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 20px rgba(0,0,0,0.2)",
        }}
      >
        {/* Header */}
        <div
          className="flex cursor-move items-center justify-between border-b border-border/30 px-4 py-2.5"
          onMouseDown={onHeaderMouseDown}
        >
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
            <span className="text-[10px] font-semibold tracking-[0.2em] text-foreground/80 uppercase font-sans">
              Oversight
            </span>
            {session && (
              <span className="text-[9px] font-mono text-muted-foreground">
                T{session.world.turn} · {session.world.tension_level?.toFixed(0) ?? "?"}% tension
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="flex h-6 w-6 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Synthetic Events Log */}
        {activeEvents.length > 0 && (
          <div className="border-b border-border/30">
            <button
              onClick={() => setEventsExpanded((p) => !p)}
              className="flex w-full cursor-pointer items-center gap-2 px-4 py-1.5 text-[9px] font-mono uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
            >
              <Zap className="h-3 w-3 text-chart-4" />
              <span>Synthetic Events ({activeEvents.length})</span>
              <span className="ml-auto">
                {eventsExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              </span>
            </button>
            {eventsExpanded && (
              <div className="max-h-[120px] overflow-y-auto px-3 pb-2 scrollbar-thin">
                {activeEvents.map((evt) => (
                  <div
                    key={evt.id}
                    className="group mb-1 flex items-start gap-2 rounded-sm border border-chart-4/20 bg-chart-4/5 px-2.5 py-1.5"
                  >
                    <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-chart-4" />
                    <div className="min-w-0 flex-1">
                      {editingEventId === evt.id ? (
                        <div className="flex items-center gap-1">
                          <input
                            value={editText}
                            onChange={(e) => setEditText(e.target.value)}
                            className="flex-1 bg-transparent text-[10px] text-foreground outline-none border-b border-chart-4/40"
                            autoFocus
                          />
                          <button
                            onClick={() => {
                              setEditingEventId(null);
                              // Re-inject with edited text
                              if (editText.trim() && onInjectEvent) {
                                void onInjectEvent({
                                  source: "oversight_injection",
                                  headline: editText.trim(),
                                  severity: evt.signal.severity,
                                  tags: ["synthetic", "oversight", "edited"],
                                });
                              }
                            }}
                            className="cursor-pointer text-secondary hover:text-foreground"
                          >
                            <Check className="h-3 w-3" />
                          </button>
                        </div>
                      ) : (
                        <p className="text-[10px] leading-relaxed text-foreground/80">{evt.signal.headline}</p>
                      )}
                      <div className="mt-0.5 flex items-center gap-2 text-[8px] font-mono text-muted-foreground">
                        <span>T{evt.injectedAtTurn}</span>
                        <span>sev {(evt.signal.severity ?? 0.5).toFixed(1)}</span>
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => {
                          setEditingEventId(evt.id);
                          setEditText(evt.signal.headline);
                        }}
                        className="cursor-pointer p-0.5 text-muted-foreground hover:text-foreground"
                        title="Edit & re-inject"
                      >
                        <Edit3 className="h-3 w-3" />
                      </button>
                      <button
                        onClick={() => onRewindToEvent?.(evt.id)}
                        className="cursor-pointer p-0.5 text-muted-foreground hover:text-chart-4"
                        title="Rewind to before this event"
                      >
                        <RotateCcw className="h-3 w-3" />
                      </button>
                      <button
                        onClick={() => onRemoveEvent?.(evt.id)}
                        className="cursor-pointer p-0.5 text-muted-foreground hover:text-primary"
                        title="Remove event"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-3 scrollbar-thin">
          <div className="flex flex-col gap-3">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={cn(
                  "max-w-[85%] text-xs leading-relaxed",
                  msg.role === "user"
                    ? "ml-auto rounded-md bg-primary/15 px-3 py-2 text-foreground"
                    : msg.role === "system"
                    ? "text-muted-foreground font-mono text-[10px] border-l-2 border-primary/30 pl-3 py-1 whitespace-pre-line"
                    : msg.role === "injection"
                    ? "rounded-md border border-chart-4/40 bg-chart-4/10 px-3 py-2 text-chart-4 font-mono text-[10px]"
                    : "rounded-md border border-border/30 bg-muted/20 px-3 py-2 text-foreground/90 font-sans"
                )}
              >
                {msg.content}
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                <span className="text-[10px] font-mono">
                  {isInjectMode ? "Injecting event…" : "Analyzing…"}
                </span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-border/30 px-3 py-2.5">
          {isInjectMode && (
            <div className="mb-1.5 flex items-center gap-2 text-[9px] font-mono text-chart-4">
              <Zap className="h-3 w-3" />
              <span>INJECT MODE — event will be processed by all agents</span>
            </div>
          )}
          <div className="flex items-center gap-2">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isInjectMode ? "Describe the synthetic event…" : "Query oversight or /inject <event>…"}
              className={cn(
                "flex-1 bg-transparent text-xs text-foreground font-sans placeholder:text-muted-foreground/50 outline-none",
                isInjectMode && "text-chart-4"
              )}
              disabled={loading}
            />
            <button
              onClick={() => void sendMessage()}
              disabled={loading || !input.trim()}
              className={cn(
                "flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-sm transition-colors",
                input.trim()
                  ? isInjectMode
                    ? "bg-chart-4/20 text-chart-4 hover:bg-chart-4/30"
                    : "bg-primary/20 text-primary hover:bg-primary/30"
                  : "text-muted-foreground/30"
              )}
            >
              {isInjectMode ? <Zap className="h-3.5 w-3.5" /> : <Send className="h-3.5 w-3.5" />}
            </button>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}

/* ── Response generator with full context ── */

function generateContextualResponse(
  question: string,
  contextBrief: string,
  syntheticEvents: SyntheticEvent[]
): string {
  const q = question.toLowerCase();

  if (!contextBrief) {
    return "No active session. Start the backend and create a session to get real-time oversight intelligence.";
  }

  // Build synthetic event context
  const syntheticCtx = syntheticEvents.length
    ? `\n\nACTIVE SYNTHETIC EVENTS (${syntheticEvents.length}):\n` +
      syntheticEvents
        .map((e) => `  • [T${e.injectedAtTurn}, sev ${(e.signal.severity ?? 0.5).toFixed(1)}] ${e.signal.headline}`)
        .join("\n")
    : "";

  const fullCtx = contextBrief + syntheticCtx;

  if (q.includes("status") || q.includes("sitrep") || q.includes("brief") || q.includes("state")) {
    return `OVERSIGHT SITUATION REPORT:\n\n${fullCtx}`;
  }

  if (q.includes("tension") || q.includes("stress") || q.includes("escalat")) {
    return `${fullCtx}\n\nTension tracks composite pressure from agent actions. Values above 60 are critical. Strikes and mobilizations spike tension; negotiations and holds reduce it.`;
  }

  if (q.includes("agent") || q.includes("who") || q.includes("player") || q.includes("actor")) {
    return `${fullCtx}\n\n6 geopolitical agents: US, Israel, Iran, Hezbollah, Gulf coalition, and Oversight. Each selects actions per turn based on observations and fog-of-war.`;
  }

  if (q.includes("oil") || q.includes("market") || q.includes("econom")) {
    return `${fullCtx}\n\nOil pressure and market stress reflect economic dimensions. Strikes and sanctions spike these; diplomatic actions stabilize them.`;
  }

  if (q.includes("event") || q.includes("news") || q.includes("intel") || q.includes("synthetic")) {
    return `${fullCtx}\n\nEvents are injected via real-world RSS harvesting, scenario scripts, or oversight synthetic injection (/inject). Each event triggers agent re-evaluation.`;
  }

  if (q.includes("inject") || q.includes("how")) {
    return `To inject a synthetic event:\n  /inject <headline>\n  /inject [0.8] Major oil disruption in Gulf\n  /inject {middle_east} [high] Diplomatic summit cancelled\n\nSeverity: [0.0-1.0] or [high]/[low]\nRegion: {region_name}\n\nInjected events trigger immediate agent reactions and advance the simulation.`;
  }

  return `${fullCtx}\n\nFor specific analysis, ask about: tension, agents, markets, events, or inject synthetic events with /inject.`;
}
