from __future__ import annotations

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import LiveControlRequest
from trenches_env.source_ingestion import SourceHarvester


class FakeFetcher:
    def fetch(self, url: str) -> tuple[str, str]:
        html = f"""
        <html>
          <head>
            <title>Snapshot for {url}</title>
            <meta name="description" content="Latest collected snapshot from {url}" />
          </head>
          <body>
            <h1>Snapshot for {url}</h1>
          </body>
        </html>
        """
        return html, "text/html; charset=utf-8"


def build_warm_env() -> FogOfWarDiplomacyEnv:
    harvester = SourceHarvester(fetcher=FakeFetcher(), auto_start=False)
    return FogOfWarDiplomacyEnv(source_harvester=harvester).enable_source_warm_start()


def test_source_monitor_reports_training_delivery_per_agent() -> None:
    env = build_warm_env()
    session = env.create_session(seed=7)
    session = env.refresh_session_sources(session, force=True)
    report = env.source_monitor(session)

    assert report.live_enabled is False
    assert report.summary.blocked_agents == 0

    for agent in report.agents:
        assert agent.configured_training_sources >= 1
        assert agent.available_training_packet_count >= 1
        assert agent.delivered_training_brief_count >= 1
        assert agent.status in {"healthy", "degraded"}
        assert all(
            issue.message != "Training-source packets are available but none reached the model brief."
            for issue in agent.issues
        )


def test_source_monitor_reports_live_delivery_per_agent() -> None:
    env = build_warm_env()
    session = env.create_session(seed=7)
    session = env.configure_live_session(
        session,
        LiveControlRequest(enabled=True, auto_step=False, poll_interval_ms=15_000),
    )
    session = env.refresh_session_sources(session, force=True)
    report = env.source_monitor(session)

    assert report.live_enabled is True
    assert report.summary.blocked_agents == 0

    for agent in report.agents:
        assert agent.configured_live_sources >= 1
        assert agent.available_live_packet_count >= 1
        assert agent.delivered_live_brief_count >= 1
        assert agent.status in {"healthy", "degraded"}
        assert all(
            issue.message != "Live-source packets are available but none reached the model brief."
            for issue in agent.issues
        )
