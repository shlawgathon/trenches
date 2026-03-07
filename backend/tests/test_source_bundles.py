from trenches_env.source_bundles import AGENT_LIVE_SOURCE_BUNDLES, AGENT_TRAINING_SOURCE_BUNDLES


def test_each_agent_has_distinct_training_and_live_sources() -> None:
    for agent_id, training_sources in AGENT_TRAINING_SOURCE_BUNDLES.items():
        live_sources = AGENT_LIVE_SOURCE_BUNDLES.get(agent_id, [])

        assert len(training_sources) >= 6
        assert len(live_sources) >= 1
        assert len(training_sources) == len(set(training_sources))
        assert len(live_sources) == len(set(live_sources))
        assert set(training_sources).isdisjoint(live_sources)
