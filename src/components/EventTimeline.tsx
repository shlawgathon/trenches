"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
    Play,
    Pause,
    SkipBack,
    SkipForward,
    Rewind,
    FastForward,
    GitBranch,
    ChevronUp,
    ChevronDown,
    Filter,
    X,
    Trophy,
    TerminalSquare,
    Activity,
} from "lucide-react";
import gsap from "gsap";
import { cn } from "@/src/lib/utils";
import { GlowingEffect } from "@/src/components/ui/glowing-effect";
import type { SessionState } from "@/src/lib/types";
import {
    deriveTimelineEvents,
    deriveTurnSnapshots,
    linkPredictionsToOutcomes,
} from "@/src/lib/timeline-types";
import type {
    TimelineEvent,
    PlaybackSpeed,
    TimelineEventType,
} from "@/src/lib/timeline-types";

/* ── Constants ── */

const AGENT_COLORS: Record<string, string> = {
    us: "#e53935",
    israel: "#64b5f6",
    iran: "#689f38",
    hezbollah: "#ffa000",
    gulf: "#a1887f",
    oversight: "#b0b0b0",
    global: "#888888",
};

const TYPE_COLORS: Record<TimelineEventType, string> = {
    prediction: "#e53935",
    actual: "#689f38",
    injection: "#ffa000",
};

const SPEEDS: PlaybackSpeed[] = [0.5, 1, 2, 4];

const COLLAPSED_HEIGHT = 48;
const EXPANDED_HEIGHT = 180;
const TIMELINE_PLOT_LEFT = 236;
const TIMELINE_PLOT_RIGHT = 24;

type TimelineViewMode = "timeline" | "console";
type ConsoleEntryKind = "action" | "reaction" | "source" | "event";

type ConsoleEntry = {
    id: string;
    kind: ConsoleEntryKind;
    turn: number;
    agent: string | null;
    title: string;
    summary: string;
    detail: string | null;
    timestamp: string;
    accent: string;
};

/* ── Sub-components ── */

function EventMarker({
    event,
    position,
    isMatched,
    active,
    onHover,
    onLeave,
    onClick,
}: {
    event: TimelineEvent;
    position: number;
    isMatched: boolean;
    active: boolean;
    onHover: (e: TimelineEvent, rect: DOMRect) => void;
    onLeave: () => void;
    onClick: (event: TimelineEvent) => void;
}) {
    const ref = useRef<HTMLDivElement>(null);
    const color = TYPE_COLORS[event.type];
    const isInjection = event.type === "injection";
    const isFaded = event.type === "prediction" && !event.matchedPredictionId;

    return (
        <div
            ref={ref}
            className={cn("absolute top-1/2 -translate-y-1/2 cursor-pointer transition-transform duration-150 hover:scale-150", active && "scale-150") }
            style={{ left: `${position * 100}%` }}
            onMouseEnter={() => {
                if (ref.current) onHover(event, ref.current.getBoundingClientRect());
            }}
            onMouseLeave={onLeave}
            onClick={() => onClick(event)}
        >
            {isInjection ? (
                <div
                    className="h-3.5 w-3.5 rotate-45 border-2"
                    style={{
                        backgroundColor: `${color}aa`,
                        borderColor: color,
                        opacity: isFaded ? 0.4 : 1,
                        boxShadow: active ? `0 0 10px ${color}` : `0 0 3px ${color}40`,
                    }}
                />
            ) : (
                <div
                    className={cn("h-3 w-3 rounded-full", isMatched && "ring-2 ring-offset-1 ring-offset-transparent")}
                    style={{
                        backgroundColor: color,
                        opacity: isFaded ? 0.4 : 1,
                        boxShadow: active ? `0 0 12px ${color}` : `0 0 4px ${color}60`,
                    }}
                />
            )}
        </div>
    );
}

function EventTooltip({
    event,
    position,
}: {
    event: TimelineEvent | null;
    position: { x: number; y: number } | null;
}) {
    if (!event || !position) return null;

    const agentColor = AGENT_COLORS[event.agent] ?? "#888";
    const typeLabel = event.type.toUpperCase();

    return (
        <div
            className="pointer-events-none fixed z-50 max-w-[260px] border border-border/40 bg-card/95 px-3 py-2 font-sans backdrop-blur-xl"
            style={{
                left: Math.min(position.x, window.innerWidth - 280),
                top: position.y - 90,
                boxShadow:
                    "0 0 8px rgba(0,0,0,0.15), 0 4px 12px rgba(0,0,0,0.25)",
            }}
        >
            <div className="mb-1 flex items-center gap-2">
                <div
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ backgroundColor: agentColor }}
                />
                <span className="text-[9px] font-semibold uppercase tracking-wider text-foreground/80">
                    {event.agent}
                </span>
                <span
                    className="ml-auto inline-flex items-center border px-1 py-0.5 text-[8px] font-mono uppercase tracking-wider"
                    style={{
                        borderColor: `${TYPE_COLORS[event.type]}60`,
                        color: TYPE_COLORS[event.type],
                    }}
                >
                    {typeLabel}
                </span>
            </div>
            <p className="text-[11px] leading-relaxed text-foreground/90">
                {event.summary}
            </p>
            <div className="mt-1.5 flex items-center gap-3 text-[9px] font-mono text-muted-foreground">
                <span>T{event.turn}</span>
                <span>SEV {(event.severity * 10).toFixed(1)}</span>
                {event.matchedPredictionId && (
                    <span className="text-secondary">✓ matched → +reward</span>
                )}
                {event.type === "prediction" && !event.matchedPredictionId && (
                    <span className="text-muted-foreground/50">⊘ unmatched</span>
                )}
                {event.type === "injection" && (
                    <span className="text-destructive">⑂ manual · no reward</span>
                )}
                {event.branchId && (
                    <span className="text-destructive">⑂ branch</span>
                )}
            </div>
        </div>
    );
}

function SpeedButton({
    speed,
    active,
    onClick,
}: {
    speed: PlaybackSpeed;
    active: boolean;
    onClick: () => void;
}) {
    return (
        <button
            onClick={onClick}
            className={cn(
                "px-1.5 py-0.5 text-[9px] font-mono transition-colors cursor-pointer",
                active
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground hover:text-foreground"
            )}
        >
            {speed}×
        </button>
    );
}

/* ── Main Component ── */

export type TimelineInteractionFocus = {
    turn: number;
    agent: string | null;
};

export type EventTimelineProps = {
    session: SessionState | null;
    onTurnChange?: (turn: number) => void;
    onRegisterToggle?: (fn: (collapsed: boolean) => void) => void;
    interactionFocus?: TimelineInteractionFocus | null;
    onInteractionFocus?: (focus: TimelineInteractionFocus | null) => void;
    embedded?: boolean;
    onCollapsedChange?: (collapsed: boolean) => void;
};

export function EventTimeline({
    session,
    onTurnChange,
    onRegisterToggle,
    interactionFocus,
    onInteractionFocus,
    embedded,
    onCollapsedChange,
}: EventTimelineProps) {
    const panelRef = useRef<HTMLDivElement>(null);
    const trackRef = useRef<HTMLDivElement>(null);
    const playIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const [collapsed, setCollapsed] = useState(false);
    const [currentTurn, setCurrentTurn] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [speed, setSpeed] = useState<PlaybackSpeed>(1);
    const [filterAgent, setFilterAgent] = useState<string | null>(null);
    const [filterType, setFilterType] = useState<TimelineEventType | null>(null);
    const [filterTurnFrom, setFilterTurnFrom] = useState<number | null>(null);
    const [filterTurnTo, setFilterTurnTo] = useState<number | null>(null);
    const [showFilters, setShowFilters] = useState(false);
    const [hoveredEvent, setHoveredEvent] = useState<TimelineEvent | null>(null);
    const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [topAgent, setTopAgent] = useState<string>("oversight");
    const [bottomAgent, setBottomAgent] = useState<string>("oversight");
    const [viewMode, setViewMode] = useState<TimelineViewMode>("timeline");

    // Derive data
    const rawEvents = deriveTimelineEvents(session);
    const events = linkPredictionsToOutcomes(rawEvents);
    const snapshots = deriveTurnSnapshots(session);
    const maxTurn = session?.world.turn ?? 0;
    const consoleEntries = deriveConsoleEntries(session);

    // Filter events
    const filteredEvents = events.filter((e) => {
        if (filterAgent && e.agent !== filterAgent) return false;
        if (filterType && e.type !== filterType) return false;
        if (filterTurnFrom !== null && e.turn < filterTurnFrom) return false;
        if (filterTurnTo !== null && e.turn > filterTurnTo) return false;
        return true;
    });

    // Unique agents for filter dropdown
    const agents = [...new Set(events.map((e) => e.agent))];

    // Has branches?
    const hasBranches = events.some((e) => e.type === "injection");

    // Reward stats
    const matchedPredictions = events.filter((e) => e.type === "prediction" && e.matchedPredictionId).length;
    const totalPredictions = events.filter((e) => e.type === "prediction").length;
    const hasActiveFilters = !!(filterAgent || filterType || filterTurnFrom !== null || filterTurnTo !== null);

    // Sync current turn with session
    useEffect(() => {
        if (session && !isDragging && !playing) {
            setCurrentTurn(session.world.turn);
        }
    }, [session?.world.turn]);

    // Collapse/expand
    const applyCollapse = useCallback((next: boolean) => {
        setCollapsed(next);
        onCollapsedChange?.(next);
        if (panelRef.current) {
            gsap.to(panelRef.current, {
                height: next ? COLLAPSED_HEIGHT : EXPANDED_HEIGHT,
                duration: 0.4,
                ease: "power2.inOut",
            });
        }
    }, [onCollapsedChange]);

    useEffect(() => {
        onRegisterToggle?.(applyCollapse);
    }, [onRegisterToggle, applyCollapse]);

    // Playback loop
    useEffect(() => {
        if (playing && maxTurn > 0) {
            const intervalMs = 1000 / speed;
            playIntervalRef.current = setInterval(() => {
                setCurrentTurn((prev) => {
                    const next = prev + 1;
                    if (next > maxTurn) {
                        setPlaying(false);
                        return maxTurn;
                    }
                    return next;
                });
            }, intervalMs);
        }
        return () => {
            if (playIntervalRef.current) clearInterval(playIntervalRef.current);
        };
    }, [playing, speed, maxTurn]);

    // Sync parent when playback advances currentTurn
    useEffect(() => {
        onTurnChange?.(currentTurn);
    }, [currentTurn]);

    // Scrubber drag handlers
    const handleTrackClick = (e: React.MouseEvent) => {
        if (!trackRef.current || maxTurn === 0) return;
        const rect = trackRef.current.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const turn = Math.round(ratio * maxTurn);
        setCurrentTurn(turn);
        onTurnChange?.(turn);
    };

    const handleDragStart = () => setIsDragging(true);

    const handleDrag = useCallback(
        (e: MouseEvent) => {
            if (!trackRef.current || maxTurn === 0) return;
            const rect = trackRef.current.getBoundingClientRect();
            const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const turn = Math.round(ratio * maxTurn);
            setCurrentTurn(turn);
            onTurnChange?.(turn);
        },
        [maxTurn, onTurnChange]
    );

    const handleDragEnd = useCallback(() => {
        setIsDragging(false);
    }, []);

    useEffect(() => {
        if (isDragging) {
            window.addEventListener("mousemove", handleDrag);
            window.addEventListener("mouseup", handleDragEnd);
            return () => {
                window.removeEventListener("mousemove", handleDrag);
                window.removeEventListener("mouseup", handleDragEnd);
            };
        }
    }, [isDragging, handleDrag, handleDragEnd]);

    // Control helpers
    const stepBack = () => {
        const next = Math.max(0, currentTurn - 1);
        setCurrentTurn(next);
        onTurnChange?.(next);
    };
    const stepForward = () => {
        const next = Math.min(maxTurn, currentTurn + 1);
        setCurrentTurn(next);
        onTurnChange?.(next);
    };
    const rewind = () => {
        setCurrentTurn(0);
        onTurnChange?.(0);
    };
    const fastForward = () => {
        setCurrentTurn(maxTurn);
        onTurnChange?.(maxTurn);
    };

    // Tension chart for scrubber background
    const tensionPath = snapshots.length > 1
        ? snapshots
            .map((s, i) => {
                const x = (s.turn / Math.max(maxTurn, 1)) * 100;
                const y = 100 - Math.min(s.tensionAfter, 100);
                return `${i === 0 ? "M" : "L"} ${x} ${y}`;
            })
            .join(" ")
        : "";

    // Current snapshot info
    const currentSnapshot = snapshots.find((s) => s.turn === currentTurn);

    const progress = maxTurn > 0 ? (currentTurn / maxTurn) * 100 : 0;

    const eventMatchesInteraction = (event: TimelineEvent): boolean => {
        if (!interactionFocus) return false;
        return event.turn === interactionFocus.turn && (!interactionFocus.agent || event.agent === interactionFocus.agent);
    };

    return (
        <>
            <EventTooltip event={hoveredEvent} position={tooltipPos} />

            <div
                ref={panelRef}
                className={cn(
                    "z-40 select-none pointer-events-auto",
                    embedded ? "relative w-full" : "absolute right-20 bottom-4 left-20"
                )}
                style={{ height: EXPANDED_HEIGHT, overflow: "hidden" }}
            >
                {/* ── Panel wrapper: matches NewsFeed / ActivityLog structure ── */}
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
                        className="relative flex h-full flex-col overflow-hidden rounded-[inherit] bg-card/25 backdrop-blur-lg"
                        style={{
                            boxShadow:
                                "0 0 8px rgba(0,0,0,0.03), 0 2px 6px rgba(0,0,0,0.08), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 12px rgba(0,0,0,0.15)",
                        }}
                    >
                        {/* Header row */}
                        <div
                            className="flex shrink-0 cursor-pointer items-center gap-2 border-b border-border/30 px-4 py-3 hover:bg-muted/10"
                            onClick={() => applyCollapse(!collapsed)}
                        >
                            <button
                                onClick={(e) => { e.stopPropagation(); applyCollapse(!collapsed); }}
                                className="flex h-5 w-5 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
                                title={collapsed ? "Expand timeline" : "Collapse timeline"}
                            >
                                {collapsed ? (
                                    <ChevronUp className="h-3.5 w-3.5" />
                                ) : (
                                    <ChevronDown className="h-3.5 w-3.5" />
                                )}
                            </button>

                            <span className="whitespace-nowrap text-[10px] font-semibold tracking-[0.2em] text-foreground/80 uppercase font-sans">
                                Timeline
                            </span>

                            <span className="text-[9px] font-mono text-muted-foreground">
                                T{currentTurn}/{maxTurn}
                            </span>

                            {currentSnapshot && (
                                <>
                                    <div className="mx-1 h-4 w-px bg-border/40" />
                                    <span
                                        className={cn(
                                            "text-[9px] font-mono",
                                            currentSnapshot.escalation
                                                ? "text-primary"
                                                : "text-muted-foreground"
                                        )}
                                    >
                                        ⚡ {currentSnapshot.tensionAfter.toFixed(0)}
                                    </span>
                                    {currentSnapshot.hasOversight && (
                                        <span className="text-[9px] font-mono text-primary">
                                            🛡 OVERSIGHT
                                        </span>
                                    )}
                                </>
                            )}

                            {hasBranches && (
                                <div className="flex items-center gap-1 text-[9px] font-mono text-destructive">
                                    <GitBranch className="h-3 w-3" />
                                    <span>BRANCH</span>
                                </div>
                            )}

                            {totalPredictions > 0 && (
                                <>
                                    <div className="mx-1 h-4 w-px bg-border/40" />
                                    <div className="flex items-center gap-1 text-[9px] font-mono">
                                        <Trophy className="h-3 w-3 text-secondary" />
                                        <span className={matchedPredictions > 0 ? "text-secondary" : "text-muted-foreground"}>
                                            {matchedPredictions}/{totalPredictions}
                                        </span>
                                    </div>
                                </>
                            )}

                            <div className="ml-auto flex items-center gap-1">
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        setViewMode("timeline");
                                    }}
                                    className={cn(
                                        "flex items-center gap-1 border px-2 py-1 text-[8px] font-mono uppercase tracking-[0.18em] transition-colors",
                                        viewMode === "timeline"
                                            ? "border-primary/60 bg-primary/15 text-primary"
                                            : "border-border/30 text-muted-foreground hover:text-foreground"
                                    )}
                                >
                                    <Activity className="h-3 w-3" />
                                    Timeline
                                </button>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        setViewMode("console");
                                    }}
                                    className={cn(
                                        "flex items-center gap-1 border px-2 py-1 text-[8px] font-mono uppercase tracking-[0.18em] transition-colors",
                                        viewMode === "console"
                                            ? "border-primary/60 bg-primary/15 text-primary"
                                            : "border-border/30 text-muted-foreground hover:text-foreground"
                                    )}
                                >
                                    <TerminalSquare className="h-3 w-3" />
                                    Console
                                </button>
                            </div>

                        </div>
                        {/* Content area with opacity transition matching side panels */}
                        <div className={cn("flex min-w-0 flex-1 flex-col transition-opacity duration-300", collapsed ? "opacity-0 pointer-events-none" : "opacity-100")}>

                        {/* Scrubber area — hidden when collapsed via overflow */}
                        <div className={cn(
                            "flex min-w-0 flex-1 flex-col transition-opacity duration-300",
                            collapsed ? "opacity-0 pointer-events-none" : "opacity-100"
                        )}>
                            {!collapsed && (viewMode === "timeline" ? (
                                <div className="flex flex-1 flex-col px-4 py-1">
                                    {/* Tension backdrop SVG */}
                                    <div className="relative flex-1">
                                        {tensionPath && (
                                            <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                                                <defs>
                                                    <linearGradient id="tensionGrad" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="0%" stopColor="#e53935" stopOpacity="0.2" />
                                                        <stop offset="100%" stopColor="#e53935" stopOpacity="0" />
                                                    </linearGradient>
                                                </defs>
                                                <path d={`${tensionPath} L 100 100 L 0 100 Z`} fill="url(#tensionGrad)" />
                                                <path d={tensionPath} fill="none" stroke="#e53935" strokeWidth="0.8" strokeOpacity="0.5" vectorEffect="non-scaling-stroke" />
                                                <line x1={progress} y1="0" x2={progress} y2="100" stroke="#e0e0e0" strokeWidth="0.5" strokeOpacity="0.6" vectorEffect="non-scaling-stroke" strokeDasharray="2 2" />
                                            </svg>
                                        )}

                                        {/* Top row: agent selector + prediction dots */}
                                        <div className="absolute inset-x-0 top-[4%] flex items-center gap-1.5 px-2">
                                            <span className="text-[8px] font-bold font-mono uppercase tracking-widest text-red-400/70 shrink-0 w-8">PRED</span>
                                            {(["oversight", "us", "israel", "iran", "hezbollah", "gulf"] as const).map((id) => (
                                                <button
                                                    key={`top-${id}`}
                                                    onClick={() => setTopAgent(id)}
                                                    className={cn(
                                                        "h-5 px-1.5 text-[8px] font-mono uppercase leading-none cursor-pointer transition-all border rounded-sm",
                                                        topAgent === id
                                                            ? "border-current text-foreground bg-foreground/10"
                                                            : "border-transparent text-muted-foreground/50 hover:text-muted-foreground/80"
                                                    )}
                                                    style={topAgent === id ? { borderColor: AGENT_COLORS[id] ?? "#888", color: AGENT_COLORS[id] ?? "#888" } : undefined}
                                                >
                                                    {id === "hezbollah" ? "HEZ" : id === "oversight" ? "OVST" : id.toUpperCase()}
                                                </button>
                                            ))}
                                        </div>

                                        {/* Top row dots */}
                                        <div className="absolute top-[22%] h-0" style={{ left: TIMELINE_PLOT_LEFT, right: TIMELINE_PLOT_RIGHT }}>
                                            {filteredEvents
                                                .filter((e) => e.type === "prediction" && e.agent === topAgent)
                                                .map((event) => {
                                                    const pos = maxTurn > 0 ? event.turn / maxTurn : 0;
                                                    return (
                                                        <EventMarker
                                                            key={event.id}
                                                            event={event}
                                                            position={pos}
                                                            isMatched={!!event.matchedPredictionId}
                                                            active={eventMatchesInteraction(event)}
                                                            onHover={(ev, rect) => {
                                                                setHoveredEvent(ev);
                                                                setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top });
                                                                onInteractionFocus?.({ turn: ev.turn, agent: ev.agent === "global" ? null : ev.agent });
                                                            }}
                                                            onLeave={() => { setHoveredEvent(null); setTooltipPos(null); onInteractionFocus?.(null); }}
                                                            onClick={(ev) => { setCurrentTurn(ev.turn); onTurnChange?.(ev.turn); }}
                                                        />
                                                    );
                                                })}
                                        </div>

                                        {/* Lane divider */}
                                        <div className="absolute top-[47%] h-px bg-border/30" style={{ left: TIMELINE_PLOT_LEFT, right: TIMELINE_PLOT_RIGHT }} />

                                        {/* Bottom row: agent selector + actual dots */}
                                        <div className="absolute inset-x-0 top-[52%] flex items-center gap-1.5 px-2">
                                            <span className="text-[8px] font-bold font-mono uppercase tracking-widest text-emerald-400/70 shrink-0 w-8">REAL</span>
                                            {(["oversight", "us", "israel", "iran", "hezbollah", "gulf"] as const).map((id) => (
                                                <button
                                                    key={`bot-${id}`}
                                                    onClick={() => setBottomAgent(id)}
                                                    className={cn(
                                                        "h-5 px-1.5 text-[8px] font-mono uppercase leading-none cursor-pointer transition-all border rounded-sm",
                                                        bottomAgent === id
                                                            ? "border-current text-foreground bg-foreground/10"
                                                            : "border-transparent text-muted-foreground/50 hover:text-muted-foreground/80"
                                                    )}
                                                    style={bottomAgent === id ? { borderColor: AGENT_COLORS[id] ?? "#888", color: AGENT_COLORS[id] ?? "#888" } : undefined}
                                                >
                                                    {id === "hezbollah" ? "HEZ" : id === "oversight" ? "OVST" : id.toUpperCase()}
                                                </button>
                                            ))}
                                        </div>

                                        {/* Bottom row dots */}
                                        <div className="absolute top-[72%] h-0" style={{ left: TIMELINE_PLOT_LEFT, right: TIMELINE_PLOT_RIGHT }}>
                                            {filteredEvents
                                                .filter((e) => e.type !== "prediction" && e.agent === bottomAgent)
                                                .map((event) => {
                                                    const pos = maxTurn > 0 ? event.turn / maxTurn : 0;
                                                    return (
                                                        <EventMarker
                                                            key={event.id}
                                                            event={event}
                                                            position={pos}
                                                            isMatched={!!event.matchedPredictionId}
                                                            active={eventMatchesInteraction(event)}
                                                            onHover={(ev, rect) => {
                                                                setHoveredEvent(ev);
                                                                setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top });
                                                                onInteractionFocus?.({ turn: ev.turn, agent: ev.agent === "global" ? null : ev.agent });
                                                            }}
                                                            onLeave={() => { setHoveredEvent(null); setTooltipPos(null); onInteractionFocus?.(null); }}
                                                            onClick={(ev) => { setCurrentTurn(ev.turn); onTurnChange?.(ev.turn); }}
                                                        />
                                                    );
                                                })}
                                        </div>

                                        {/* Branch fork markers */}
                                        {filteredEvents
                                            .filter((e) => e.type === "injection")
                                            .map((e) => {
                                                const pos = maxTurn > 0 ? e.turn / maxTurn : 0;
                                                return (
                                                    <div
                                                        key={`fork-${e.id}`}
                                                        className="absolute bottom-0 -translate-x-1/2"
                                                        style={{
                                                            left: `calc(${TIMELINE_PLOT_LEFT}px + ((100% - ${TIMELINE_PLOT_LEFT + TIMELINE_PLOT_RIGHT}px) * ${pos}))`,
                                                        }}
                                                    >
                                                        <GitBranch className="h-3 w-3 text-destructive/70" />
                                                    </div>
                                                );
                                            })}
                                    </div>

                                    {/* Scrubber track */}
                                    <div className="mt-1 flex items-center gap-3">
                                        {/* Playback controls */}
                                        <div className="flex shrink-0 items-center gap-0.5">
                                            <button
                                                onClick={rewind}
                                                className="flex h-6 w-6 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
                                                title="Rewind"
                                            >
                                                <Rewind className="h-3 w-3" />
                                            </button>
                                            <button
                                                onClick={stepBack}
                                                className="flex h-6 w-6 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
                                                title="Step back"
                                            >
                                                <SkipBack className="h-3 w-3" />
                                            </button>
                                            <button
                                                onClick={() => setPlaying(!playing)}
                                                className={cn(
                                                    "flex h-7 w-7 cursor-pointer items-center justify-center transition-colors",
                                                    playing
                                                        ? "text-primary"
                                                        : "text-muted-foreground hover:text-foreground"
                                                )}
                                                title={playing ? "Pause" : "Play"}
                                            >
                                                {playing ? (
                                                    <Pause className="h-3.5 w-3.5" />
                                                ) : (
                                                    <Play className="h-3.5 w-3.5" />
                                                )}
                                            </button>
                                            <button
                                                onClick={stepForward}
                                                className="flex h-6 w-6 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
                                                title="Step forward"
                                            >
                                                <SkipForward className="h-3 w-3" />
                                            </button>
                                            <button
                                                onClick={fastForward}
                                                className="flex h-6 w-6 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
                                                title="Fast-forward"
                                            >
                                                <FastForward className="h-3 w-3" />
                                            </button>
                                        </div>

                                        {/* Track */}
                                        <div
                                            ref={trackRef}
                                            className="relative h-2 flex-1 cursor-pointer bg-border/20"
                                            onClick={handleTrackClick}
                                        >
                                            {/* Progress fill */}
                                            <div
                                                className="absolute inset-y-0 left-0 bg-primary/40 transition-[width] duration-75"
                                                style={{ width: `${progress}%` }}
                                            />

                                            {/* Turn tick marks */}
                                            {maxTurn <= 50 &&
                                                Array.from({ length: maxTurn + 1 }, (_, i) => (
                                                    <div
                                                        key={i}
                                                        className="absolute top-0 h-full w-px bg-border/10"
                                                        style={{ left: `${(i / Math.max(maxTurn, 1)) * 100}%` }}
                                                    />
                                                ))}

                                            {/* Escalation markers on track */}
                                            {snapshots
                                                .filter((s) => s.escalation)
                                                .map((s) => (
                                                    <div
                                                        key={`esc-${s.turn}`}
                                                        className="absolute top-0 h-full w-0.5 bg-primary/50"
                                                        style={{
                                                            left: `${(s.turn / Math.max(maxTurn, 1)) * 100}%`,
                                                        }}
                                                    />
                                                ))}

                                            {/* Playhead thumb */}
                                            <div
                                                className="absolute top-1/2 h-4 w-1.5 -translate-x-1/2 -translate-y-1/2 cursor-grab bg-foreground/90 transition-[left] duration-75 active:cursor-grabbing"
                                                style={{ left: `${progress}%` }}
                                                onMouseDown={(e) => {
                                                    e.preventDefault();
                                                    handleDragStart();
                                                }}
                                            />
                                        </div>

                                        {/* Speed controls */}
                                        <div className="flex shrink-0 items-center gap-0.5 border-l border-border/20 pl-2">
                                            {SPEEDS.map((s) => (
                                                <SpeedButton
                                                    key={s}
                                                    speed={s}
                                                    active={speed === s}
                                                    onClick={() => setSpeed(s)}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex h-full flex-1 flex-col px-4 py-2">
                                    <div className="mb-2 flex items-center justify-between text-[9px] font-mono uppercase tracking-[0.18em] text-muted-foreground">
                                        <span>Unified console feed</span>
                                        <span>{consoleEntries.length} entries</span>
                                    </div>
                                    <div className="scrollbar-thin flex-1 space-y-2 overflow-y-auto pr-1">
                                        {consoleEntries.length === 0 ? (
                                            <div className="flex h-full items-center justify-center text-[11px] font-mono text-muted-foreground">
                                                No console activity yet.
                                            </div>
                                        ) : (
                                            consoleEntries.map((entry) => (
                                                <div
                                                    key={entry.id}
                                                    className="cursor-pointer border border-border/30 bg-background/20 px-3 py-2 hover:bg-muted/10"
                                                    onMouseEnter={() => onInteractionFocus?.({ turn: entry.turn, agent: entry.agent })}
                                                    onMouseLeave={() => onInteractionFocus?.(null)}
                                                    onClick={() => {
                                                        setCurrentTurn(entry.turn);
                                                        onTurnChange?.(entry.turn);
                                                        onInteractionFocus?.({ turn: entry.turn, agent: entry.agent });
                                                    }}
                                                >
                                                    <div className="mb-1 flex items-center gap-2">
                                                        <span
                                                            className="inline-flex h-2 w-2 rounded-full"
                                                            style={{ backgroundColor: entry.accent }}
                                                        />
                                                        <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-foreground/85">
                                                            {entry.title}
                                                        </span>
                                                        <span className="text-[9px] font-mono text-muted-foreground">
                                                            T{entry.turn}
                                                        </span>
                                                        {entry.agent && (
                                                            <span className="text-[9px] font-mono text-muted-foreground">
                                                                {entry.agent}
                                                            </span>
                                                        )}
                                                        <span className="ml-auto text-[9px] font-mono text-muted-foreground/70">
                                                            {formatConsoleTime(entry.timestamp)}
                                                        </span>
                                                    </div>
                                                    <p className="text-[11px] leading-relaxed text-foreground/90">
                                                        {entry.summary}
                                                    </p>
                                                    {entry.detail && (
                                                        <p className="mt-1 text-[10px] font-mono leading-relaxed text-muted-foreground">
                                                            {entry.detail}
                                                        </p>
                                                    )}
                                                </div>
                                            ))
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                        </div>

                        {/* Collapsed label */}
                        {collapsed && (
                            <div className="flex flex-1 items-center gap-3 px-4">
                                <span className="text-[9px] font-mono uppercase tracking-wider text-muted-foreground">
                                    {viewMode === "timeline" ? "Timeline" : "Console"} T{currentTurn}/{maxTurn}
                                </span>
                                <div className="h-1 flex-1 bg-border/20">
                                    <div
                                        className="h-full bg-primary/40"
                                        style={{ width: `${progress}%` }}
                                    />
                                </div>
                                {playing && (
                                    <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </>
    );
}

function deriveConsoleEntries(session: SessionState | null): ConsoleEntry[] {
    if (!session) return [];

    const entries: ConsoleEntry[] = [];

    for (const action of session.action_log ?? []) {
        const providerError = typeof action.metadata?.provider_error === "string" ? action.metadata.provider_error : null;
        const detailParts = [
            action.target ? `target=${action.target}` : null,
            action.reward_total !== undefined ? `reward=${action.reward_total.toFixed(2)}` : null,
            providerError ? `provider_error=${providerError}` : null,
        ].filter(Boolean);

        entries.push({
            id: `action-${action.created_at}-${action.actor}-${action.turn}`,
            kind: "action",
            turn: action.turn,
            agent: action.actor,
            title: "Action",
            summary: `${action.actor} ${action.action_type}${action.target ? ` -> ${action.target}` : ""}`,
            detail: detailParts.length > 0 ? detailParts.join(" · ") : action.summary,
            timestamp: action.created_at,
            accent: AGENT_COLORS[action.actor] ?? "#888888",
        });
    }

    for (const reaction of session.reaction_log ?? []) {
        const signalSummary = reaction.signals.map((signal) => signal.headline).slice(0, 2).join(" | ");
        const actorSummary = reaction.actor_outcomes
            .map((outcome) => `${outcome.agent_id}:${outcome.action.type}`)
            .slice(0, 6)
            .join(", ");

        entries.push({
            id: `reaction-${reaction.event_id}`,
            kind: "reaction",
            turn: reaction.turn,
            agent: null,
            title: "Reaction",
            summary: `${reaction.source} drove ${reaction.actor_outcomes.length} actor responses`,
            detail: [signalSummary || null, actorSummary || null, reaction.oversight_triggered ? "oversight_triggered=true" : null].filter(Boolean).join(" · ") || null,
            timestamp: reaction.created_at,
            accent: reaction.oversight_triggered ? "#e53935" : "#64b5f6",
        });
    }

    const seenSourceIds = new Set<string>();
    for (const [agentId, observation] of Object.entries(session.observations)) {
        for (const packet of observation.source_packets ?? []) {
            const dedupeId = `${packet.source_id}-${packet.fetched_at ?? packet.status}`;
            if (seenSourceIds.has(dedupeId)) continue;
            seenSourceIds.add(dedupeId);

            entries.push({
                id: `source-${dedupeId}`,
                kind: "source",
                turn: session.world.turn,
                agent: agentId,
                title: "Source",
                summary: `${packet.source_name} [${packet.status}]`,
                detail: [packet.delivery, packet.kind, packet.summary || null, packet.error || null].filter(Boolean).join(" · ") || null,
                timestamp: packet.fetched_at ?? session.updated_at,
                accent: packet.status === "error" ? "#e53935" : packet.status === "pending" ? "#ffa000" : "#689f38",
            });
        }
    }

    for (const event of session.world.active_events ?? []) {
        entries.push({
            id: `event-${event.id}`,
            kind: "event",
            turn: session.world.turn,
            agent: event.affected_agents[0] ?? null,
            title: "Event",
            summary: event.summary,
            detail: `${event.source} · severity=${event.severity.toFixed(2)}`,
            timestamp: session.updated_at,
            accent: TYPE_COLORS.actual,
        });
    }

    return entries
        .sort((a, b) => {
            const timeA = Date.parse(a.timestamp);
            const timeB = Date.parse(b.timestamp);
            if (Number.isFinite(timeA) && Number.isFinite(timeB) && timeA !== timeB) {
                return timeB - timeA;
            }
            return b.turn - a.turn;
        })
        .slice(0, 200);
}

function formatConsoleTime(timestamp: string): string {
    const parsed = Date.parse(timestamp);
    if (!Number.isFinite(parsed)) return timestamp;
    return new Date(parsed).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
