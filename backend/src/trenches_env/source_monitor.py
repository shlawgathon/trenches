from __future__ import annotations

from typing import Iterable

from trenches_env.agents import AGENT_IDS, AGENT_PROFILES
from trenches_env.models import (
    AgentObservation,
    AgentSourceMonitor,
    SessionState,
    SourceMonitorIssue,
    SourceMonitorReport,
    SourceMonitorSummary,
)
from trenches_env.source_bundles import AGENT_LIVE_SOURCE_BUNDLES, AGENT_TRAINING_SOURCE_BUNDLES
from trenches_env.source_catalog import get_sources_for_agent
from trenches_env.source_ingestion import SourceHarvester


def build_source_monitor_report(
    session: SessionState,
    *,
    harvester: SourceHarvester,
) -> SourceMonitorReport:
    agent_reports = [
        _build_agent_source_monitor(
            agent_id=agent_id,
            observation=session.observations.get(agent_id, AgentObservation()),
            live_enabled=session.live.enabled,
            harvester=harvester,
        )
        for agent_id in AGENT_IDS
    ]

    summary = SourceMonitorSummary(
        healthy_agents=sum(1 for report in agent_reports if report.status == "healthy"),
        degraded_agents=sum(1 for report in agent_reports if report.status == "degraded"),
        blocked_agents=sum(1 for report in agent_reports if report.status == "blocked"),
        active_source_count=sum(report.active_source_count for report in agent_reports),
        ok_packet_count=sum(report.ok_packet_count for report in agent_reports),
        delivered_source_brief_count=sum(
            report.delivered_training_brief_count + report.delivered_live_brief_count for report in agent_reports
        ),
    )

    return SourceMonitorReport(
        session_id=session.session_id,
        live_enabled=session.live.enabled,
        summary=summary,
        agents=agent_reports,
    )


def _build_agent_source_monitor(
    *,
    agent_id: str,
    observation: AgentObservation,
    live_enabled: bool,
    harvester: SourceHarvester,
) -> AgentSourceMonitor:
    training_sources = get_sources_for_agent(agent_id, "training_core")
    live_sources = get_sources_for_agent(agent_id, "live_demo")
    active_sources = training_sources + (live_sources if live_enabled else [])
    packets = observation.training_source_packets + (observation.live_source_packets if live_enabled else [])
    packet_by_id = {packet.source_id: packet for packet in packets}

    training_bundle = AGENT_TRAINING_SOURCE_BUNDLES.get(agent_id, [])
    live_bundle = AGENT_LIVE_SOURCE_BUNDLES.get(agent_id, [])
    training_names = {source.name for source in training_sources}
    live_names = {source.name for source in live_sources}

    missing_training_sources = _sorted_unique(set(training_bundle) - training_names)
    missing_live_sources = _sorted_unique(set(live_bundle) - live_names)
    unbundled_training_sources = _sorted_unique(training_names - set(training_bundle))
    unbundled_live_sources = _sorted_unique(live_names - set(live_bundle))
    missing_packet_sources = _sorted_unique(
        source.name for source in active_sources if source.id not in packet_by_id
    )
    sources_without_probe_targets = _sorted_unique(
        source.name
        for source in training_sources + live_sources
        if not harvester.probe_resolver.resolve_candidates(source)
    )

    ok_packet_count = 0
    pending_sources: list[str] = []
    error_sources: list[str] = []
    for source in active_sources:
        packet = packet_by_id.get(source.id)
        if packet is None:
            continue
        if packet.status == "ok":
            ok_packet_count += 1
        elif packet.status == "pending":
            pending_sources.append(source.name)
        else:
            error_sources.append(source.name)

    delivered_training_brief_count = sum(
        1 for brief in observation.private_brief if brief.category == "training_source"
    )
    delivered_live_brief_count = sum(
        1 for brief in observation.private_brief if brief.category == "live_source"
    )
    delivered_source_names = _sorted_unique(
        brief.source
        for brief in observation.private_brief
        if brief.category in {"training_source", "live_source"}
    )

    available_training_packet_count = sum(
        1 for packet in observation.training_source_packets if packet.status == "ok" and packet.summary
    )
    available_live_packet_count = sum(
        1 for packet in observation.live_source_packets if packet.status == "ok" and packet.summary
    )

    issues: list[SourceMonitorIssue] = []
    _append_alignment_issue(
        issues,
        missing=missing_training_sources,
        extra=unbundled_training_sources,
        label="training",
    )
    _append_alignment_issue(
        issues,
        missing=missing_live_sources,
        extra=unbundled_live_sources,
        label="live",
    )
    if sources_without_probe_targets:
        issues.append(
            SourceMonitorIssue(
                severity="error",
                message=f"No probe target configured for {', '.join(sources_without_probe_targets[:4])}.",
            )
        )
    if missing_packet_sources:
        issues.append(
            SourceMonitorIssue(
                severity="error",
                message=f"Observation is missing packets for {', '.join(missing_packet_sources[:4])}.",
            )
        )
    if error_sources:
        issues.append(
            SourceMonitorIssue(
                severity="warning",
                message=f"{len(error_sources)} active sources are returning errors.",
            )
        )
    if pending_sources:
        issues.append(
            SourceMonitorIssue(
                severity="warning",
                message=f"{len(pending_sources)} active sources are still pending collection.",
            )
        )
    if available_training_packet_count > 0 and delivered_training_brief_count == 0:
        issues.append(
            SourceMonitorIssue(
                severity="error",
                message="Training-source packets are available but none reached the model brief.",
            )
        )
    if live_enabled and available_live_packet_count > 0 and delivered_live_brief_count == 0:
        issues.append(
            SourceMonitorIssue(
                severity="error",
                message="Live-source packets are available but none reached the model brief.",
            )
        )

    status = "healthy"
    if any(issue.severity == "error" for issue in issues):
        status = "blocked"
    elif issues:
        status = "degraded"

    return AgentSourceMonitor(
        agent_id=agent_id,
        display_name=AGENT_PROFILES[agent_id].display_name,
        status=status,
        configured_training_sources=len(training_sources),
        configured_live_sources=len(live_sources),
        active_source_count=len(active_sources),
        ok_packet_count=ok_packet_count,
        pending_packet_count=len(pending_sources),
        error_packet_count=len(error_sources),
        available_training_packet_count=available_training_packet_count,
        available_live_packet_count=available_live_packet_count,
        delivered_training_brief_count=delivered_training_brief_count,
        delivered_live_brief_count=delivered_live_brief_count,
        missing_training_sources=missing_training_sources,
        missing_live_sources=missing_live_sources,
        unbundled_training_sources=unbundled_training_sources,
        unbundled_live_sources=unbundled_live_sources,
        missing_packet_sources=missing_packet_sources,
        sources_without_probe_targets=sources_without_probe_targets,
        error_sources=_sorted_unique(error_sources),
        pending_sources=_sorted_unique(pending_sources),
        delivered_source_names=delivered_source_names,
        issues=issues,
    )


def _append_alignment_issue(
    issues: list[SourceMonitorIssue],
    *,
    missing: list[str],
    extra: list[str],
    label: str,
) -> None:
    if missing:
        issues.append(
            SourceMonitorIssue(
                severity="error",
                message=f"{label.title()} bundle references unknown sources: {', '.join(missing[:4])}.",
            )
        )
    if extra:
        issues.append(
            SourceMonitorIssue(
                severity="error",
                message=f"{label.title()} catalog has unbundled sources: {', '.join(extra[:4])}.",
            )
        )


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})
