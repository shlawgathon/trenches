from trenches_env.source_bundles import AGENT_LIVE_SOURCE_BUNDLES, AGENT_TRAINING_SOURCE_BUNDLES


def test_each_agent_has_at_least_twenty_unique_sources() -> None:
    for agent_id, training_sources in AGENT_TRAINING_SOURCE_BUNDLES.items():
        live_sources = AGENT_LIVE_SOURCE_BUNDLES.get(agent_id, [])
        combined_sources = training_sources + live_sources

        assert len(combined_sources) >= 20
        assert len(combined_sources) == len(set(combined_sources))
