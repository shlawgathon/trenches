from __future__ import annotations

import os

import pytest

from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import LiveControlRequest
from trenches_env.source_ingestion import HttpSourceFetcher, SourceHarvester


pytestmark = pytest.mark.skipif(
    os.getenv("TRENCHES_RUN_LIVE_SOURCE_TESTS") != "1",
    reason="set TRENCHES_RUN_LIVE_SOURCE_TESTS=1 to run live network probes",
)


def test_live_source_monitor_delivers_packets_and_briefs_per_agent() -> None:
    harvester = SourceHarvester(fetcher=HttpSourceFetcher(timeout_seconds=10.0), auto_start=False)
    env = FogOfWarDiplomacyEnv(source_harvester=harvester).enable_source_warm_start()
    try:
        session = env.create_session(seed=7)
        session = env.configure_live_session(
            session,
            LiveControlRequest(enabled=True, auto_step=False, poll_interval_ms=15_000),
        )
        report = env.source_monitor(session)

        failures: list[str] = []
        for agent in report.agents:
            if agent.available_training_packet_count == 0:
                failures.append(f"{agent.agent_id}: no healthy training packet")
            if agent.delivered_training_brief_count == 0:
                failures.append(f"{agent.agent_id}: no training brief visible to the model")
            if agent.configured_live_sources > 0 and agent.available_live_packet_count == 0:
                failures.append(f"{agent.agent_id}: no healthy live packet")
            if agent.configured_live_sources > 0 and agent.delivered_live_brief_count == 0:
                failures.append(f"{agent.agent_id}: no live brief visible to the model")

        assert not failures, "\n".join(failures)
    finally:
        env.shutdown()
