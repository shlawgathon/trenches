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
const EXPANDED_HEIGHT = 220;
const TURN_WIDTH = 20; // px per turn

type TimelineViewMode = "timeline" | "console";
type ConsoleEntryKind = "action" | "reaction" | "source" | "event" | "provider" | "prediction";

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
            style={{ left: position }}
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
    playing?: boolean;
    onPlayingChange?: (playing: boolean) => void;
    onRegisterToggle?: (fn: (collapsed: boolean) => void) => void;
    interactionFocus?: TimelineInteractionFocus | null;
    onInteractionFocus?: (focus: TimelineInteractionFocus | null) => void;
    embedded?: boolean;
    onCollapsedChange?: (collapsed: boolean) => void;
};

export function EventTimeline({
    session,
    onTurnChange,
    playing: controlledPlaying,
    onPlayingChange,
    onRegisterToggle,
    interactionFocus,
    onInteractionFocus,
    embedded,
    onCollapsedChange,
}: EventTimelineProps) {
    const panelRef = useRef<HTMLDivElement>(null);
    const trackRef = useRef<HTMLDivElement>(null);
    const filterRowRef = useRef<HTMLDivElement>(null);
    const scrollRef = useRef<HTMLDivElement>(null);
    const trackScrollRef = useRef<HTMLDivElement>(null);
    const [compactFilters, setCompactFilters] = useState(false);
    const syncingScroll = useRef(false);

    const [collapsed, setCollapsed] = useState(false);
    const [currentTurn, setCurrentTurn] = useState(0);
    const [localPlaying, setLocalPlaying] = useState(false);
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
    const playing = controlledPlaying ?? localPlaying;

    const setTimelinePlaying = useCallback((next: boolean) => {
        if (controlledPlaying === undefined) {
            setLocalPlaying(next);
        }
        onPlayingChange?.(next);
    }, [controlledPlaying, onPlayingChange]);

    const pausePlayback = useCallback(() => {
        if (playing) {
            setTimelinePlaying(false);
        }
    }, [playing, setTimelinePlaying]);

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

    // Sync current turn with session — only auto-advance if user is "following" (at/near latest turn)
    const prevMaxTurnRef = useRef(maxTurn);
    useEffect(() => {
        const prevMax = prevMaxTurnRef.current;
        prevMaxTurnRef.current = maxTurn;
        if (!session || isDragging) return;
        // Only auto-advance if user was at (or past) the previous max turn
        if (currentTurn >= prevMax) {
            setCurrentTurn(maxTurn);
        }
    }, [session?.world.turn]);

    // Resize observer for compact filter mode
    useEffect(() => {
        const el = panelRef.current;
        if (!el) return;
        const ro = new ResizeObserver(([entry]) => {
            setCompactFilters(entry.contentRect.width < 500);
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

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

    // Sync parent when playback advances currentTurn
    useEffect(() => {
        onTurnChange?.(currentTurn);
    }, [currentTurn]);

    // Scrubber drag handlers
    const handleTrackClick = (e: React.MouseEvent) => {
        pausePlayback();
        if (!trackRef.current || maxTurn === 0) return;
        const rect = trackRef.current.getBoundingClientRect();
        const scrollLeft = trackScrollRef.current?.scrollLeft ?? 0;
        const clickX = e.clientX - rect.left + scrollLeft - TIMELINE_PAD;
        const turn = Math.max(0, Math.min(maxTurn, Math.round(clickX / TURN_WIDTH)));
        setCurrentTurn(turn);
        onTurnChange?.(turn);
    };

    const handleDragStart = () => setIsDragging(true);

    const handleDrag = useCallback(
        (e: MouseEvent) => {
            pausePlayback();
            if (!trackRef.current || maxTurn === 0) return;
            const rect = trackRef.current.getBoundingClientRect();
            const scrollLeft = trackScrollRef.current?.scrollLeft ?? 0;
            const clickX = e.clientX - rect.left + scrollLeft - TIMELINE_PAD;
            const turn = Math.max(0, Math.min(maxTurn, Math.round(clickX / TURN_WIDTH)));
            setCurrentTurn(turn);
            onTurnChange?.(turn);
        },
        [maxTurn, onTurnChange, pausePlayback]
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
        pausePlayback();
        const next = Math.max(0, currentTurn - 1);
        setCurrentTurn(next);
        onTurnChange?.(next);
    };
    const stepForward = () => {
        pausePlayback();
        const next = Math.min(maxTurn, currentTurn + 1);
        setCurrentTurn(next);
        onTurnChange?.(next);
    };
    const rewind = () => {
        pausePlayback();
        setCurrentTurn(0);
        onTurnChange?.(0);
    };
    const fastForward = () => {
        pausePlayback();
        setCurrentTurn(maxTurn);
        onTurnChange?.(maxTurn);
    };

    // Fixed-width timeline calculations
    const TIMELINE_PAD = 44; // px padding at left (clears PRED/REAL labels)
    const innerWidth = Math.max(maxTurn, 1) * TURN_WIDTH + TIMELINE_PAD * 2;
    const turnToX = (turn: number) => TIMELINE_PAD + turn * TURN_WIDTH;

    // Tension chart for scrubber background (pixel-based SVG)
    const tensionPath = snapshots.length > 1
        ? snapshots
            .map((s, i) => {
                const x = turnToX(s.turn);
                const y = 100 - Math.min(s.tensionAfter, 100);
                return `${i === 0 ? "M" : "L"} ${x} ${y}`;
            })
            .join(" ")
        : "";

    // Current snapshot info
    const currentSnapshot = snapshots.find((s) => s.turn === currentTurn);
    const progress = maxTurn > 0 ? (currentTurn / maxTurn) * 100 : 0;

    // Auto-scroll only when playhead goes off-screen (not continuous centering)
    useEffect(() => {
        const el = scrollRef.current;
        if (!el || collapsed) return;
        const headX = turnToX(currentTurn);
        const viewLeft = el.scrollLeft;
        const viewRight = viewLeft + el.clientWidth;
        // Only scroll if playhead is near the edge or off-screen
        if (headX < viewLeft + 40 || headX > viewRight - 60) {
            el.scrollTo({ left: Math.max(0, headX - el.clientWidth * 0.75), behavior: "smooth" });
        }
    }, [currentTurn, collapsed, maxTurn]);

    // Sync scroll between dot area and scrubber track
    const handleSyncScroll = (source: "dots" | "track") => (e: React.UIEvent<HTMLDivElement>) => {
        if (syncingScroll.current) return;
        syncingScroll.current = true;
        const scrollLeft = (e.target as HTMLDivElement).scrollLeft;
        const other = source === "dots" ? trackScrollRef.current : scrollRef.current;
        if (other) other.scrollLeft = scrollLeft;
        requestAnimationFrame(() => { syncingScroll.current = false; });
    };

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
                        className="relative flex h-full flex-col rounded-[inherit] bg-card/25 backdrop-blur-lg"
                        style={{
                            boxShadow:
                                "0 0 8px rgba(0,0,0,0.03), 0 2px 6px rgba(0,0,0,0.08), inset 0 0 6px 6px rgba(255,255,255,0.04), 0 0 12px rgba(0,0,0,0.15)",
                        }}
                    >
                        {/* Header row */}
                        <div className="shrink-0 border-b border-border/30">
                            {/* Top line: collapse toggle, title, stats, mode switch */}
                            <div
                                className="flex cursor-pointer items-center gap-2 px-4 py-2 hover:bg-muted/10"
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

                            {/* Filter row: PRED / REAL agent selectors + type filter */}
                            {!collapsed && viewMode === "timeline" && (
                                <div ref={filterRowRef} className="flex items-center gap-3 border-t border-border/20 px-4 py-1">
                                    {/* PRED lane selector */}
                                    <div className="flex items-center gap-1">
                                        {!compactFilters && <span className="text-[8px] font-bold font-mono uppercase tracking-widest text-red-400/70 shrink-0 w-7">PRED</span>}
                                        {(["oversight", "us", "israel", "iran", "hezbollah", "gulf"] as const).map((id) => (
                                            <button
                                                key={`top-${id}`}
                                                onClick={() => setTopAgent(id)}
                                                title={`Predictions: ${id}`}
                                                className={cn(
                                                    "cursor-pointer transition-all border rounded-sm",
                                                    compactFilters
                                                        ? "h-4 w-4 flex items-center justify-center"
                                                        : "h-5 px-1.5 text-[8px] font-mono uppercase leading-none",
                                                    topAgent === id
                                                        ? "border-current text-foreground bg-foreground/10"
                                                        : "border-transparent text-muted-foreground/50 hover:text-muted-foreground/80"
                                                )}
                                                style={topAgent === id ? { borderColor: AGENT_COLORS[id] ?? "#888", color: AGENT_COLORS[id] ?? "#888" } : undefined}
                                            >
                                                {compactFilters ? (
                                                    <span
                                                        className="flex h-4 w-4 items-center justify-center rounded-full text-[7px] font-bold text-black/80"
                                                        style={{ backgroundColor: AGENT_COLORS[id] ?? "#888", opacity: topAgent === id ? 1 : 0.35 }}
                                                    >
                                                        {id[0].toUpperCase()}
                                                    </span>
                                                ) : (
                                                    id === "hezbollah" ? "HEZ" : id === "oversight" ? "OVST" : id.toUpperCase()
                                                )}
                                            </button>
                                        ))}
                                    </div>

                                    <div className="h-3.5 w-px bg-border/30" />

                                    {/* REAL lane selector */}
                                    <div className="flex items-center gap-1">
                                        {!compactFilters && <span className="text-[8px] font-bold font-mono uppercase tracking-widest text-emerald-400/70 shrink-0 w-7">REAL</span>}
                                        {(["oversight", "us", "israel", "iran", "hezbollah", "gulf"] as const).map((id) => (
                                            <button
                                                key={`bot-${id}`}
                                                onClick={() => setBottomAgent(id)}
                                                title={`Actuals: ${id}`}
                                                className={cn(
                                                    "cursor-pointer transition-all border rounded-sm",
                                                    compactFilters
                                                        ? "h-4 w-4 flex items-center justify-center"
                                                        : "h-5 px-1.5 text-[8px] font-mono uppercase leading-none",
                                                    bottomAgent === id
                                                        ? "border-current text-foreground bg-foreground/10"
                                                        : "border-transparent text-muted-foreground/50 hover:text-muted-foreground/80"
                                                )}
                                                style={bottomAgent === id ? { borderColor: AGENT_COLORS[id] ?? "#888", color: AGENT_COLORS[id] ?? "#888" } : undefined}
                                            >
                                                {compactFilters ? (
                                                    <span
                                                        className="flex h-4 w-4 items-center justify-center rounded-full text-[7px] font-bold text-black/80"
                                                        style={{ backgroundColor: AGENT_COLORS[id] ?? "#888", opacity: bottomAgent === id ? 1 : 0.35 }}
                                                    >
                                                        {id[0].toUpperCase()}
                                                    </span>
                                                ) : (
                                                    id === "hezbollah" ? "HEZ" : id === "oversight" ? "OVST" : id.toUpperCase()
                                                )}
                                            </button>
                                        ))}
                                    </div>

                                    {/* Type filter + active filter indicator */}
                                    <div className="ml-auto flex items-center gap-1.5">
                                        {hasActiveFilters && (
                                            <button
                                                onClick={() => {
                                                    setFilterAgent(null);
                                                    setFilterType(null);
                                                    setFilterTurnFrom(null);
                                                    setFilterTurnTo(null);
                                                }}
                                                className="flex items-center gap-0.5 text-[8px] font-mono text-primary/80 hover:text-primary cursor-pointer"
                                            >
                                                <X className="h-2.5 w-2.5" /> Clear
                                            </button>
                                        )}
                                        <button
                                            onClick={() => setShowFilters(!showFilters)}
                                            className={cn(
                                                "flex items-center gap-1 border px-1.5 py-0.5 text-[8px] font-mono uppercase tracking-wider transition-colors cursor-pointer",
                                                showFilters || hasActiveFilters
                                                    ? "border-primary/40 text-primary bg-primary/10"
                                                    : "border-border/30 text-muted-foreground hover:text-foreground"
                                            )}
                                        >
                                            <Filter className="h-2.5 w-2.5" />
                                            {hasActiveFilters ? `${filteredEvents.length}/${events.length}` : "Filter"}
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* Expanded filter panel */}
                            {!collapsed && showFilters && viewMode === "timeline" && (
                                <div className="flex items-center gap-3 border-t border-border/20 px-4 py-1.5 bg-muted/5">
                                    <div className="flex items-center gap-1.5">
                                        <span className="text-[8px] font-mono uppercase text-muted-foreground">Agent</span>
                                        <select
                                            value={filterAgent ?? ""}
                                            onChange={(e) => setFilterAgent(e.target.value || null)}
                                            className="h-5 bg-transparent border border-border/30 px-1 text-[9px] font-mono text-foreground/80 cursor-pointer"
                                        >
                                            <option value="">All</option>
                                            {agents.map((a) => (
                                                <option key={a} value={a}>{a}</option>
                                            ))}
                                        </select>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <span className="text-[8px] font-mono uppercase text-muted-foreground">Type</span>
                                        <select
                                            value={filterType ?? ""}
                                            onChange={(e) => setFilterType((e.target.value || null) as TimelineEventType | null)}
                                            className="h-5 bg-transparent border border-border/30 px-1 text-[9px] font-mono text-foreground/80 cursor-pointer"
                                        >
                                            <option value="">All</option>
                                            <option value="prediction">Prediction</option>
                                            <option value="actual">Actual</option>
                                            <option value="injection">Injection</option>
                                        </select>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <span className="text-[8px] font-mono uppercase text-muted-foreground">Turns</span>
                                        <input
                                            type="number"
                                            min={0}
                                            max={maxTurn}
                                            placeholder="from"
                                            value={filterTurnFrom ?? ""}
                                            onChange={(e) => setFilterTurnFrom(e.target.value ? Number(e.target.value) : null)}
                                            className="h-5 w-12 bg-transparent border border-border/30 px-1 text-[9px] font-mono text-foreground/80"
                                        />
                                        <span className="text-[8px] text-muted-foreground">–</span>
                                        <input
                                            type="number"
                                            min={0}
                                            max={maxTurn}
                                            placeholder="to"
                                            value={filterTurnTo ?? ""}
                                            onChange={(e) => setFilterTurnTo(e.target.value ? Number(e.target.value) : null)}
                                            className="h-5 w-12 bg-transparent border border-border/30 px-1 text-[9px] font-mono text-foreground/80"
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                        {/* Content area */}
                        <div className={cn("flex min-w-0 flex-1 flex-col overflow-hidden transition-opacity duration-300", collapsed ? "opacity-0 pointer-events-none" : "opacity-100")}>
                            {!collapsed && (viewMode === "timeline" ? (
                                <div className="flex flex-1 flex-col py-1 overflow-hidden">
                                    {/* Scrollable dot + tension area */}
                                    <div
                                        ref={scrollRef}
                                        className="relative flex-1 overflow-x-auto overflow-y-hidden scrollbar-thin"
                                        onScroll={handleSyncScroll("dots")}
                                    >
                                        <div className="relative h-full" style={{ width: innerWidth, minWidth: "100%" }}>
                                            {/* Tension backdrop SVG */}
                                            {tensionPath && (
                                                <svg className="absolute inset-0 h-full" style={{ width: innerWidth }} viewBox={`0 0 ${innerWidth} 100`} preserveAspectRatio="none">
                                                    <defs>
                                                        <linearGradient id="tensionGrad" x1="0" y1="0" x2="0" y2="1">
                                                            <stop offset="0%" stopColor="#e53935" stopOpacity="0.2" />
                                                            <stop offset="100%" stopColor="#e53935" stopOpacity="0" />
                                                        </linearGradient>
                                                    </defs>
                                                    <path d={`${tensionPath} L ${innerWidth} 100 L 0 100 Z`} fill="url(#tensionGrad)" />
                                                    <path d={tensionPath} fill="none" stroke="#e53935" strokeWidth="0.8" strokeOpacity="0.5" vectorEffect="non-scaling-stroke" />
                                                    <line x1={turnToX(currentTurn)} y1="0" x2={turnToX(currentTurn)} y2="100" stroke="#e0e0e0" strokeWidth="0.5" strokeOpacity="0.6" vectorEffect="non-scaling-stroke" strokeDasharray="2 2" />
                                                </svg>
                                            )}

                                            {/* Lane labels (sticky on left) */}
                                            <span className="sticky left-3 top-[10%] z-10 inline-block text-[7px] font-bold font-mono uppercase tracking-widest text-red-400/40 select-none" style={{ position: "absolute", top: "10%" }}>PRED</span>
                                            <span className="sticky left-3 top-[58%] z-10 inline-block text-[7px] font-bold font-mono uppercase tracking-widest text-emerald-400/40 select-none" style={{ position: "absolute", top: "58%" }}>REAL</span>

                                            {/* Lane divider */}
                                            <div className="absolute top-[47%] h-px bg-border/30" style={{ width: innerWidth }} />

                                            {/* Top row dots (predictions) */}
                                            <div className="absolute top-[15%] h-0" style={{ width: innerWidth }}>
                                                {filteredEvents
                                                    .filter((e) => e.type === "prediction" && e.agent === topAgent)
                                                    .map((event) => (
                                                        <EventMarker
                                                            key={event.id}
                                                            event={event}
                                                            position={turnToX(event.turn)}
                                                            isMatched={!!event.matchedPredictionId}
                                                            active={eventMatchesInteraction(event)}
                                                            onHover={(ev, rect) => {
                                                                setHoveredEvent(ev);
                                                                setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top });
                                                                onInteractionFocus?.({ turn: ev.turn, agent: ev.agent === "global" ? null : ev.agent });
                                                            }}
                                                            onLeave={() => { setHoveredEvent(null); setTooltipPos(null); onInteractionFocus?.(null); }}
                                                            onClick={(ev) => { pausePlayback(); setCurrentTurn(ev.turn); onTurnChange?.(ev.turn); }}
                                                        />
                                                    ))}
                                            </div>

                                            {/* Bottom row dots (actuals) */}
                                            <div className="absolute top-[65%] h-0" style={{ width: innerWidth }}>
                                                {filteredEvents
                                                    .filter((e) => e.type !== "prediction" && e.agent === bottomAgent)
                                                    .map((event) => (
                                                        <EventMarker
                                                            key={event.id}
                                                            event={event}
                                                            position={turnToX(event.turn)}
                                                            isMatched={!!event.matchedPredictionId}
                                                            active={eventMatchesInteraction(event)}
                                                            onHover={(ev, rect) => {
                                                                setHoveredEvent(ev);
                                                                setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top });
                                                                onInteractionFocus?.({ turn: ev.turn, agent: ev.agent === "global" ? null : ev.agent });
                                                            }}
                                                            onLeave={() => { setHoveredEvent(null); setTooltipPos(null); onInteractionFocus?.(null); }}
                                                            onClick={(ev) => { pausePlayback(); setCurrentTurn(ev.turn); onTurnChange?.(ev.turn); }}
                                                        />
                                                    ))}
                                            </div>

                                            {/* Branch fork markers */}
                                            {filteredEvents
                                                .filter((e) => e.type === "injection")
                                                .map((e) => (
                                                    <div
                                                        key={`fork-${e.id}`}
                                                        className="absolute bottom-0 -translate-x-1/2"
                                                        style={{ left: turnToX(e.turn) }}
                                                    >
                                                        <GitBranch className="h-3 w-3 text-destructive/70" />
                                                    </div>
                                                ))}

                                            {/* Turn tick marks */}
                                            {Array.from({ length: maxTurn + 1 }, (_, i) => (
                                                <div
                                                    key={`tick-${i}`}
                                                    className="absolute top-0 h-full w-px bg-border/10"
                                                    style={{ left: turnToX(i) }}
                                                />
                                            ))}
                                        </div>
                                    </div>

                                    {/* Scrubber track */}
                                    <div className="flex items-center gap-3 px-4 pt-1">
                                        {/* Playback controls */}
                                        <div className="flex shrink-0 items-center gap-0.5">
                                            <button onClick={rewind} className="flex h-6 w-6 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground" title="Rewind">
                                                <Rewind className="h-3 w-3" />
                                            </button>
                                            <button onClick={stepBack} className="flex h-6 w-6 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground" title="Step back">
                                                <SkipBack className="h-3 w-3" />
                                            </button>
                                            <button
                                                onClick={() => {
                                                    if (!playing) {
                                                        setCurrentTurn(maxTurn);
                                                        onTurnChange?.(maxTurn);
                                                    }
                                                    setTimelinePlaying(!playing);
                                                }}
                                                className={cn(
                                                    "flex h-7 w-7 cursor-pointer items-center justify-center transition-colors",
                                                    playing ? "text-primary" : "text-muted-foreground hover:text-foreground"
                                                )}
                                                title={playing ? "Pause" : "Play"}
                                            >
                                                {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                                            </button>
                                            <button onClick={stepForward} className="flex h-6 w-6 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground" title="Step forward">
                                                <SkipForward className="h-3 w-3" />
                                            </button>
                                            <button onClick={fastForward} className="flex h-6 w-6 cursor-pointer items-center justify-center text-muted-foreground transition-colors hover:text-foreground" title="Fast-forward">
                                                <FastForward className="h-3 w-3" />
                                            </button>
                                        </div>

                                        {/* Scrollable track (synced with dot area) */}
                                        <div
                                            ref={trackScrollRef}
                                            className="flex-1 overflow-x-auto overflow-y-hidden scrollbar-none"
                                            onScroll={handleSyncScroll("track")}
                                        >
                                            <div
                                                ref={trackRef}
                                                className="relative h-2 cursor-pointer bg-border/20"
                                                style={{ width: innerWidth, minWidth: "100%" }}
                                                onClick={(e) => {
                                                    pausePlayback();
                                                    if (!trackRef.current) return;
                                                    const rect = trackRef.current.getBoundingClientRect();
                                                    const scrollLeft = trackScrollRef.current?.scrollLeft ?? 0;
                                                    const clickX = e.clientX - rect.left + scrollLeft - TIMELINE_PAD;
                                                    const turn = Math.max(0, Math.min(maxTurn, Math.round(clickX / TURN_WIDTH)));
                                                    setCurrentTurn(turn);
                                                    onTurnChange?.(turn);
                                                }}
                                            >
                                                {/* Progress fill */}
                                                <div
                                                    className="absolute inset-y-0 left-0 bg-primary/40"
                                                    style={{ width: turnToX(currentTurn) }}
                                                />

                                                {/* Escalation markers */}
                                                {snapshots
                                                    .filter((s) => s.escalation)
                                                    .map((s) => (
                                                        <div
                                                            key={`esc-${s.turn}`}
                                                            className="absolute top-0 h-full w-0.5 bg-primary/50"
                                                            style={{ left: turnToX(s.turn) }}
                                                        />
                                                    ))}

                                                {/* Playhead thumb */}
                                                <div
                                                    className="absolute top-1/2 h-4 w-1.5 -translate-x-1/2 -translate-y-1/2 cursor-grab bg-foreground/90 active:cursor-grabbing"
                                                    style={{ left: turnToX(currentTurn) }}
                                                    onMouseDown={(e) => {
                                                        e.preventDefault();
                                                        handleDragStart();
                                                    }}
                                                />
                                            </div>
                                        </div>

                                        {/* Speed controls */}
                                        <div className="flex shrink-0 items-center gap-0.5 border-l border-border/20 pl-2">
                                            {SPEEDS.map((s) => (
                                                <SpeedButton key={s} speed={s} active={speed === s} onClick={() => setSpeed(s)} />
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex h-full flex-1 flex-col px-4 py-2 overflow-hidden">
                                    <div className="mb-1.5 flex items-center justify-between text-[9px] font-mono uppercase tracking-[0.18em] text-muted-foreground">
                                        <span>Unified console feed</span>
                                        <span>{consoleEntries.length} entries</span>
                                    </div>
                                    <div className="flex-1 overflow-y-auto scrollbar-thin pr-1">
                                        {consoleEntries.length === 0 ? (
                                            <div className="flex h-full items-center justify-center text-[11px] font-mono text-muted-foreground">
                                                No console activity yet.
                                            </div>
                                        ) : (
                                            consoleEntries.map((entry) => (
                                                <div
                                                    key={entry.id}
                                                    className="flex cursor-pointer items-center gap-2 border-b border-border/15 px-1 py-[3px] text-[10px] font-mono hover:bg-muted/10 transition-colors"
                                                    onMouseEnter={() => onInteractionFocus?.({ turn: entry.turn, agent: entry.agent })}
                                                    onMouseLeave={() => onInteractionFocus?.(null)}
                                                    onClick={() => {
                                                        pausePlayback();
                                                        setCurrentTurn(entry.turn);
                                                        onTurnChange?.(entry.turn);
                                                        onInteractionFocus?.({ turn: entry.turn, agent: entry.agent });
                                                    }}
                                                >
                                                    <span
                                                        className="inline-flex h-1.5 w-1.5 shrink-0 rounded-full"
                                                        style={{ backgroundColor: entry.accent }}
                                                    />
                                                    <span className="shrink-0 w-7 text-muted-foreground/70">T{entry.turn}</span>
                                                    <span className="shrink-0 w-[52px] text-[9px] font-semibold uppercase tracking-wider text-foreground/70">{entry.title}</span>
                                                    <span className="min-w-0 flex-1 truncate text-foreground/85">{entry.summary}</span>
                                                    {entry.agent && (
                                                        <span className="shrink-0 text-muted-foreground/60">{entry.agent}</span>
                                                    )}
                                                    <span className="shrink-0 text-[9px] text-muted-foreground/50">{formatConsoleTime(entry.timestamp)}</span>
                                                </div>
                                            ))
                                        )}
                                    </div>
                                </div>
                            ))}
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

    for (const [agentId, binding] of Object.entries(session.model_bindings ?? {})) {
        if (binding.decision_mode !== "heuristic_fallback" && binding.provider !== "openrouter") {
            continue;
        }

        const detailParts = [
            `provider=${binding.provider}`,
            `mode=${binding.decision_mode}`,
            binding.model_name ? `model=${binding.model_name}` : null,
            binding.base_url ? `base_url=${binding.base_url}` : null,
        ].filter(Boolean);

        entries.push({
            id: `provider-${agentId}-${binding.provider}-${binding.decision_mode}-${session.updated_at}`,
            kind: "provider",
            turn: session.world.turn,
            agent: agentId,
            title: "Provider",
            summary: binding.provider === "openrouter"
                ? `${agentId} is using OpenRouter mock fallback`
                : `${agentId} is on heuristic fallback`,
            detail: detailParts.join(" · "),
            timestamp: session.updated_at,
            accent: binding.decision_mode === "heuristic_fallback" ? "#e53935" : "#ffa000",
        });
    }

    for (const action of session.action_log ?? []) {
        const providerError = typeof action.metadata?.provider_error === "string" ? action.metadata.provider_error : null;
        const providerMode = typeof action.metadata?.mode === "string" ? action.metadata.mode : null;
        const providerName = typeof action.metadata?.provider === "string" ? action.metadata.provider : null;
        const detailParts = [
            action.target ? `target=${action.target}` : null,
            action.reward_total !== undefined ? `reward=${action.reward_total.toFixed(2)}` : null,
            providerMode ? `mode=${providerMode}` : null,
            providerName ? `provider=${providerName}` : null,
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

    for (const prediction of (session.prediction_log ?? []) as Record<string, unknown>[]) {
        const agent = typeof prediction.agent_id === "string" ? prediction.agent_id : "oversight";
        const turn = typeof prediction.turn === "number" ? prediction.turn : session.world.turn;
        const summary = typeof prediction.summary === "string" ? prediction.summary : "Oversight prediction";
        const rationale = typeof prediction.rationale === "string" ? prediction.rationale : null;
        const confidence = typeof prediction.confidence === "number" ? `${(prediction.confidence * 100).toFixed(0)}%` : null;
        const predictedActor = typeof prediction.predicted_actor === "string" ? prediction.predicted_actor : null;
        const predictedTarget = typeof prediction.predicted_target === "string" ? prediction.predicted_target : null;
        const topic = typeof prediction.topic === "string" ? prediction.topic : null;
        const predictionId = typeof prediction.prediction_id === "string" ? prediction.prediction_id : `pred-${turn}`;
        const timestamp = typeof prediction.timestamp === "string" ? prediction.timestamp : session.updated_at;

        const detailParts = [
            topic ? `topic=${topic}` : null,
            confidence ? `confidence=${confidence}` : null,
            predictedActor ? `actor=${predictedActor}` : null,
            predictedTarget ? `target=${predictedTarget}` : null,
            rationale || null,
        ].filter(Boolean);

        entries.push({
            id: `prediction-${predictionId}`,
            kind: "prediction",
            turn,
            agent,
            title: "Prediction",
            summary,
            detail: detailParts.length > 0 ? detailParts.join(" · ") : null,
            timestamp,
            accent: "#b388ff",
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
