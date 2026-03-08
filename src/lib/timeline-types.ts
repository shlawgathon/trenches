import type { SessionState, StepTrace } from "./types";

/* ── Core timeline event ── */

export type TimelineEventType = "prediction" | "actual" | "injection";
export type TimelineEventSource = "live" | "env" | "manual";

export type TimelineEvent = {
    id: string;
    turn: number;
    type: TimelineEventType;
    agent: string;
    summary: string;
    severity: number;
    source: TimelineEventSource;
    matchedPredictionId?: string | null;
    branchId?: string | null;
    timestamp: string;
};

/* ── Branch (fork from chat injection) ── */

export type TimelineBranch = {
    id: string;
    forkTurn: number;
    label: string;
    events: TimelineEvent[];
    parentBranchId?: string | null;
};

/* ── Playback state ── */

export type PlaybackSpeed = 0.5 | 1 | 2 | 4;

export type PlaybackState = {
    currentTurn: number;
    maxTurn: number;
    playing: boolean;
    speed: PlaybackSpeed;
    viewingBranchId: string | null;
    filterAgent: string | null;
    filterType: TimelineEventType | null;
};

/* ── Turn snapshot (for scrubber) ── */

export type TurnSnapshot = {
    turn: number;
    tensionBefore: number;
    tensionAfter: number;
    actionCount: number;
    hasOversight: boolean;
    escalation: boolean; // tension delta > 5
    timestamp: string;
};

/* ── Derivation helpers ── */

export function deriveTimelineEvents(session: SessionState | null): TimelineEvent[] {
    if (!session) return [];

    const events: TimelineEvent[] = [];

    // 1) Actuals — from world active_events
    for (const ev of session.world.active_events) {
        events.push({
            id: `actual-${ev.id}`,
            turn: session.world.turn,
            type: "actual",
            agent: ev.affected_agents[0] ?? "global",
            summary: ev.summary,
            severity: ev.severity,
            source: ev.source === "manual" ? "manual" : ev.source === "live" ? "live" : "env",
            timestamp: session.updated_at,
        });
    }

    // 2) From recent_traces — actions become predictions
    for (const trace of session.recent_traces) {
        const agentIds = Object.keys(trace.actions);
        for (const agentId of agentIds) {
            const action = trace.actions[agentId];
            if (!action) continue;

            // Escalation actions (strike, mobilize, sanction) are "predictions" of impact
            const isPredictive = ["strike", "mobilize", "sanction", "deceive"].includes(action.type);

            events.push({
                id: `trace-${agentId}-${trace.turn}`,
                turn: trace.turn,
                type: isPredictive ? "prediction" : "actual",
                agent: action.actor,
                summary: action.summary,
                severity: Math.abs(trace.tension_after - trace.tension_before) / 10,
                source: "env",
                timestamp: trace.created_at,
            });
        }

        // Oversight interventions as special events
        if (trace.oversight.triggered) {
            events.push({
                id: `oversight-${trace.turn}`,
                turn: trace.turn,
                type: "actual",
                agent: "oversight",
                summary: trace.oversight.reason,
                severity: trace.oversight.risk_score / 100,
                source: "env",
                timestamp: trace.created_at,
            });
        }
    }

    // 3) Injections — from active_events with manual source
    for (const ev of session.world.active_events) {
        if (ev.source === "manual") {
            events.push({
                id: `injection-${ev.id}`,
                turn: session.world.turn,
                type: "injection",
                agent: ev.affected_agents[0] ?? "global",
                summary: ev.summary,
                severity: ev.severity,
                source: "manual",
                branchId: `branch-${ev.id}`,
                timestamp: session.updated_at,
            });
        }
    }

    // Sort by turn asc then by type priority (actual > prediction > injection)
    const typePriority: Record<TimelineEventType, number> = { actual: 0, prediction: 1, injection: 2 };
    events.sort((a, b) => a.turn - b.turn || typePriority[a.type] - typePriority[b.type]);

    return events;
}

export function deriveTurnSnapshots(session: SessionState | null): TurnSnapshot[] {
    if (!session) return [];

    return session.recent_traces.map((trace: StepTrace) => ({
        turn: trace.turn,
        tensionBefore: trace.tension_before,
        tensionAfter: trace.tension_after,
        actionCount: Object.keys(trace.actions).length,
        hasOversight: trace.oversight.triggered,
        escalation: (trace.tension_after - trace.tension_before) > 5,
        timestamp: trace.created_at,
    }));
}

export function linkPredictionsToOutcomes(events: TimelineEvent[]): TimelineEvent[] {
    const predictions = events.filter((e) => e.type === "prediction");
    const actuals = events.filter((e) => e.type === "actual");

    // Simple matching: same agent, within 2 turns
    for (const pred of predictions) {
        const match = actuals.find(
            (a) =>
                a.agent === pred.agent &&
                Math.abs(a.turn - pred.turn) <= 2 &&
                !a.matchedPredictionId
        );
        if (match) {
            pred.matchedPredictionId = match.id;
            match.matchedPredictionId = pred.id;
        }
    }

    return events;
}
