from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from trenches_env.source_catalog import get_all_sources
from trenches_env.source_ingestion import HttpSourceFetcher, SourceHarvester


pytestmark = pytest.mark.skipif(
    os.getenv("TRENCHES_RUN_LIVE_SOURCE_TESTS") != "1",
    reason="set TRENCHES_RUN_LIVE_SOURCE_TESTS=1 to run live network probes",
)


def test_all_sources_have_a_live_accessible_probe() -> None:
    harvester = SourceHarvester(fetcher=HttpSourceFetcher(timeout_seconds=10.0), auto_start=False)
    sources = get_all_sources()

    def probe(source_id: str) -> tuple[str, str, str | None]:
        source = next(source for source in sources if source.id == source_id)
        packet = harvester.probe_source(source)
        return source.name, packet.status, packet.error

    failures: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=12) as executor:
            for source_name, status, error in executor.map(probe, [source.id for source in sources]):
                if status != "ok":
                    failures.append(f"{source_name}: {error or 'probe failed'}")
    finally:
        harvester.stop()

    assert not failures, "\n".join(failures)
