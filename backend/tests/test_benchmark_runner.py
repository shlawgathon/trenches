from __future__ import annotations

from trenches_env.benchmark_runner import ScenarioBenchmarkRunner
from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import BenchmarkRunRequest
from trenches_env.source_ingestion import SourceHarvester


def build_offline_env() -> FogOfWarDiplomacyEnv:
    return FogOfWarDiplomacyEnv(source_harvester=SourceHarvester(auto_start=False))


def test_named_scenario_applies_distinct_world_and_episode_metadata() -> None:
    env = build_offline_env()
    session = env.create_session(seed=11, scenario_id="shipping_crisis")

    assert session.episode.scenario_id == "shipping_crisis"
    assert session.episode.scenario_name == "Shipping Crisis"
    assert "shipping" in session.episode.scenario_tags
    assert session.world.tension_level >= 64.0
    assert session.world.oil_pressure >= 78.0
    assert session.world.actor_state["gulf"]["shipping_continuity"] < 78.0
    assert any("tankers" in event.summary.lower() or "shipping" in event.summary.lower() for event in session.world.active_events)


def test_scenario_creation_is_deterministic_for_fixed_seed() -> None:
    env = build_offline_env()
    first = env.create_session(seed=7, scenario_id="border_flareup")
    second = env.create_session(seed=7, scenario_id="border_flareup")

    assert first.world.model_dump() == second.world.model_dump()
    assert first.observations["israel"].perceived_tension == second.observations["israel"].perceived_tension


def test_benchmark_runner_returns_scorecards_for_each_agent() -> None:
    runner = ScenarioBenchmarkRunner(env_factory=build_offline_env)
    result = runner.run(
        BenchmarkRunRequest(
            scenario_ids=["shipping_crisis", "coalition_fracture"],
            seed=13,
            steps_per_scenario=4,
        )
    )

    assert result.scenario_count == 2
    assert set(result.aggregate_mean_total_rewards) == {"us", "israel", "iran", "hezbollah", "gulf", "oversight"}
    assert result.results[0].scorecards["gulf"].final_state["shipping_continuity"] < 78.0
    assert result.results[1].scorecards["israel"].final_state["us_resupply_confidence"] < 75.0
    assert result.results[1].scorecards["us"].dominant_action is not None


def test_benchmark_runner_is_deterministic_for_fixed_seed() -> None:
    runner = ScenarioBenchmarkRunner(env_factory=build_offline_env)
    request = BenchmarkRunRequest(
        scenario_ids=["corridor_interdiction"],
        seed=5,
        steps_per_scenario=4,
    )

    first = runner.run(request)
    second = runner.run(request)

    assert first.model_dump() == second.model_dump()
