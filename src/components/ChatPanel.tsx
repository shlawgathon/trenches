"use client";

import { useRef, useState, useEffect } from "react";
import { Send, X, Loader2 } from "lucide-react";
import gsap from "gsap";
import { cn } from "@/src/lib/utils";
import { getRuntimeEnv } from "@/src/lib/env";

type Message = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
};

interface ChatPanelProps {
  open: boolean;
  onClose: () => void;
  sessionId?: string | null;
}

export function ChatPanel({ open, onClose, sessionId }: ChatPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "sys-0",
      role: "system",
      content: "TRENCHES AI — Ask about the simulation, agent behaviors, tensions, or world state.",
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // GSAP open/close animation
  useEffect(() => {
    if (!panelRef.current) return;

    if (open) {
      gsap.fromTo(
        panelRef.current,
        { y: 40, opacity: 0, scale: 0.95, backdropFilter: "blur(0px)", pointerEvents: "none" },
        {
          y: 0,
          opacity: 1,
          scale: 1,
          backdropFilter: "blur(16px)",
          pointerEvents: "auto",
          duration: 0.35,
          ease: "power3.out",
        }
      );
      setTimeout(() => inputRef.current?.focus(), 350);
    } else {
      gsap.to(panelRef.current, {
        y: 20,
        opacity: 0,
        scale: 0.97,
        backdropFilter: "blur(0px)",
        pointerEvents: "none",
        duration: 0.2,
        ease: "power2.in",
      });
    }
  }, [open]);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

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
      // Fetch current session state for context
      let context = "";
      if (sessionId) {
        const { apiBaseUrl } = getRuntimeEnv();
        try {
          const stateRes = await fetch(`${apiBaseUrl}/sessions/${sessionId}`);
          if (stateRes.ok) {
            const state = await stateRes.json();
            context = `Current simulation state (Turn ${state.world?.turn ?? "?"}): ` +
              `Tension=${state.world?.tension_level?.toFixed(1) ?? "?"}, ` +
              `Market Stress=${state.world?.market_stress?.toFixed(1) ?? "?"}, ` +
              `Oil Pressure=${state.world?.oil_pressure?.toFixed(1) ?? "?"}, ` +
              `Active Events=${state.world?.active_events?.length ?? 0}. ` +
              `Agents: ${Object.keys(state.observations ?? {}).join(", ")}. `;

            // Add recent reactions if available
            try {
              const reactRes = await fetch(`${apiBaseUrl}/sessions/${sessionId}/reactions`);
              if (reactRes.ok) {
                const reactions = await reactRes.json();
                if (reactions.length > 0) {
                  const recent = reactions.slice(-3);
                  context += "Recent agent reactions: " +
                    recent.map((r: { agent_id: string; summary: string }) =>
                      `${r.agent_id}: ${r.summary}`
                    ).join("; ") + ". ";
                }
              }
            } catch {
              // reactions endpoint optional
            }
          }
        } catch {
          context = "Backend not reachable — answering from general knowledge. ";
        }
      }

      // Build a local response based on context (no external LLM dependency)
      const assistantMsg: Message = {
        id: `asst-${Date.now()}`,
        role: "assistant",
        content: generateLocalResponse(text, context),
        timestamp: Date.now(),
      };

      // Simulate slight delay for UX
      await new Promise((r) => setTimeout(r, 400 + Math.random() * 300));
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "assistant",
          content: "Connection error. Unable to fetch simulation data.",
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

  return (
    <div
      ref={panelRef}
      className="pointer-events-none absolute bottom-20 left-1/2 z-30 w-[540px] -translate-x-1/2 opacity-0"
    >
      <div
        className="pointer-events-auto flex h-[320px] flex-col overflow-hidden rounded-md border border-border/40 bg-card/40 backdrop-blur-lg"
        style={{
          boxShadow:
            "0 0 8px rgba(0,0,0,0.03), 0 4px 12px rgba(0,0,0,0.15), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 20px rgba(0,0,0,0.2)",
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/30 px-4 py-2.5">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
            <span className="text-[10px] font-semibold tracking-[0.2em] text-foreground/80 uppercase font-sans">
              AI Intel
            </span>
          </div>
          <button
            onClick={onClose}
            className="flex h-6 w-6 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>

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
                    ? "text-muted-foreground font-mono text-[10px] border-l-2 border-primary/30 pl-3 py-1"
                    : "rounded-md border border-border/30 bg-muted/20 px-3 py-2 text-foreground/90 font-sans"
                )}
              >
                {msg.content}
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                <span className="text-[10px] font-mono">Analyzing...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-border/30 px-3 py-2.5">
          <div className="flex items-center gap-2">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about the simulation..."
              className="flex-1 bg-transparent text-xs text-foreground font-sans placeholder:text-muted-foreground/50 outline-none"
              disabled={loading}
            />
            <button
              onClick={() => void sendMessage()}
              disabled={loading || !input.trim()}
              className={cn(
                "flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-sm transition-colors",
                input.trim()
                  ? "bg-primary/20 text-primary hover:bg-primary/30"
                  : "text-muted-foreground/30"
              )}
            >
              <Send className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Local response generator using session context
function generateLocalResponse(question: string, context: string): string {
  const q = question.toLowerCase();

  if (!context) {
    return "No active session connected. Start the backend and create a session to get real-time simulation intelligence.";
  }

  if (q.includes("tension") || q.includes("stress") || q.includes("escalat")) {
    return `${context}\n\nThe simulation tracks tension as a composite metric influenced by agent actions (strikes, sanctions, mobilizations increase it; negotiations and holds decrease it). Values above 60 are considered critical.`;
  }

  if (q.includes("agent") || q.includes("who") || q.includes("player")) {
    return `${context}\n\nThe simulation runs 6 geopolitical agents: US, Israel, Iran, Hezbollah, Gulf coalition, and an Oversight entity. Each agent selects actions per turn based on their observations and fog-of-war constraints.`;
  }

  if (q.includes("oil") || q.includes("market") || q.includes("econom")) {
    return `${context}\n\nOil pressure and market stress reflect economic dimensions of the crisis. Strikes and sanctions tend to spike these values, while diplomatic actions stabilize them.`;
  }

  if (q.includes("event") || q.includes("news") || q.includes("intel")) {
    return `${context}\n\nEvents are injected via real-world source harvesting or scenario scripts. Each event has a severity (0-1) and source attribution. Agents receive filtered intel based on their fog-of-war visibility.`;
  }

  if (q.includes("reward") || q.includes("score") || q.includes("perform")) {
    return `${context}\n\nRewards are differentiated per agent based on their objectives: stability-oriented agents gain from reduced tension, while adversarial agents may benefit from escalation. The oversight entity penalizes rule violations.`;
  }

  return `${context}\n\nFor specific analysis, try asking about: tension levels, agent behaviors, market impacts, active events, or reward patterns.`;
}
