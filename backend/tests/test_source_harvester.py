from __future__ import annotations

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import LiveControlRequest
from trenches_env.source_bundles import AGENT_LIVE_SOURCE_BUNDLES, AGENT_TRAINING_SOURCE_BUNDLES
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


def test_every_source_has_probe_target() -> None:
    harvester = SourceHarvester(fetcher=FakeFetcher(), auto_start=False)
    assert harvester.all_sources_have_probe_targets() is True


def test_training_source_packets_are_wired_into_observations() -> None:
    harvester = SourceHarvester(fetcher=FakeFetcher(), auto_start=False)
    env = FogOfWarDiplomacyEnv(source_harvester=harvester)
    session = env.create_session(seed=7)
    session = env.refresh_session_sources(session, force=True)

    for agent_id, observation in session.observations.items():
        assert len(observation.training_source_packets) == len(AGENT_TRAINING_SOURCE_BUNDLES[agent_id])
        assert len(observation.source_packets) == len(AGENT_TRAINING_SOURCE_BUNDLES[agent_id])
        assert all(packet.status == "ok" for packet in observation.training_source_packets)
        assert any(brief.category == "training_source" for brief in observation.private_brief)


def test_live_source_packets_are_wired_when_live_mode_is_enabled() -> None:
    harvester = SourceHarvester(fetcher=FakeFetcher(), auto_start=False)
    env = FogOfWarDiplomacyEnv(source_harvester=harvester)
    session = env.create_session(seed=7)
    session = env.configure_live_session(
        session,
        LiveControlRequest(enabled=True, auto_step=False, poll_interval_ms=15_000),
    )
    session = env.refresh_session_sources(session, force=True)

    for agent_id, observation in session.observations.items():
        assert len(observation.live_source_packets) == len(AGENT_LIVE_SOURCE_BUNDLES[agent_id])
        assert len(observation.source_packets) == (
            len(AGENT_TRAINING_SOURCE_BUNDLES[agent_id]) + len(AGENT_LIVE_SOURCE_BUNDLES[agent_id])
        )
