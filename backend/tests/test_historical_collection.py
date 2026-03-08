from datetime import UTC, datetime

from trenches_env.historical_collection import (
    CollectedHistoricalArticle,
    article_to_historical_event,
    build_replay_definition,
    build_source_profiles_for_agent,
    resolve_window,
)
from trenches_env.historical_replay import HistoricalReplayDefinition


def test_build_source_profiles_uses_manifest_domains() -> None:
    profiles = build_source_profiles_for_agent("us")

    assert profiles
    assert any("reuters.com" in profile.domains for profile in profiles)
    assert all(profile.query_terms for profile in profiles)


def test_resolve_window_clamps_future_end_to_current_day() -> None:
    window = resolve_window("2026", now=datetime(2026, 3, 7, 12, 0, tzinfo=UTC))

    assert window.start_date.isoformat() == "2026-01-01"
    assert window.end_date.isoformat() == "2026-03-08"


def test_article_to_historical_event_matches_replay_schema() -> None:
    article = CollectedHistoricalArticle(
        article_id="abc123",
        agent_id="us",
        source_id="us-reuters-us",
        source_name="Reuters US",
        title="Commercial shipping risk rises near Hormuz after new tanker threat warning.",
        url="https://www.reuters.com/world/middle-east/example",
        domain="reuters.com",
        timestamp=datetime(2025, 1, 12, 9, 0, tzinfo=UTC),
        query='(domainis:reuters.com) AND ("Hormuz" OR "shipping")',
        window_id="2025",
        tags=["shipping", "wire"],
    )

    event = article_to_historical_event(article, training_agent="us")

    assert event.topic == "shipping"
    assert event.source_type == "gdelt_historical_collection"
    assert event.public_summary == article.title
    assert event.impact.actor_metric_deltas["us"]


def test_build_replay_definition_outputs_valid_replay_file_shape() -> None:
    window = resolve_window("2025", now=datetime(2026, 3, 7, 12, 0, tzinfo=UTC))
    articles = [
        CollectedHistoricalArticle(
            article_id="a1",
            agent_id="israel",
            source_id="israel-reuters",
            source_name="Reuters",
            title="Drone and rocket probes test northern air-defense coverage.",
            url="https://www.reuters.com/world/middle-east/a1",
            domain="reuters.com",
            timestamp=datetime(2025, 1, 15, 6, 30, tzinfo=UTC),
            query="(domainis:reuters.com) AND (Israel OR Hezbollah)",
            window_id="2025",
            tags=["border"],
        ),
        CollectedHistoricalArticle(
            article_id="a2",
            agent_id="israel",
            source_id="israel-jpost",
            source_name="Jerusalem Post",
            title="Indirect ceasefire talks produce a temporary lull on the northern front.",
            url="https://www.jpost.com/middle-east/a2",
            domain="jpost.com",
            timestamp=datetime(2025, 9, 22, 11, 0, tzinfo=UTC),
            query="(domainis:jpost.com) AND (Israel OR Hezbollah)",
            window_id="2025",
            tags=["diplomacy"],
        ),
    ]

    replay = build_replay_definition(training_agent="israel", window=window, articles=articles, max_events=16)
    validated = HistoricalReplayDefinition.model_validate_json(replay.model_dump_json())

    assert validated.replay_id == "israel_historical_2025"
    assert len(validated.events) == 2
    assert validated.events[0].timestamp < validated.events[1].timestamp
