from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

from trenches_env.agents import AGENT_IDS
from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import (
    BenchmarkEntityScorecard,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkScenarioResult,
    StepSessionRequest,
)
from trenches_env.scenarios import benchmark_scenario_ids, get_scenario_definition, scenario_signals_for_turn
from trenches_env.source_ingestion import SourceHarvester


def _default_env_factory() -> FogOfWarDiplomacyEnv:
    return FogOfWarDiplomacyEnv(source_harvester=SourceHarvester(auto_start=False))


class ScenarioBenchmarkRunner:
    def __init__(self, env_factory: Callable[[], FogOfWarDiplomacyEnv] | None = None) -> None:
        self._env_factory = env_factory or _default_env_factory

    def run(self, request: BenchmarkRunRequest) -> BenchmarkRunResponse:
        scenario_ids = request.scenario_ids or benchmark_scenario_ids()
        results: list[BenchmarkScenarioResult] = []
        aggregate_reward_totals: dict[str, float] = {agent_id: 0.0 for agent_id in AGENT_IDS}

        for index, scenario_id in enumerate(scenario_ids):
            scenario = get_scenario_definition(scenario_id)
            scenario_seed = None if request.seed is None else request.seed + index
            turn_limit = request.steps_per_scenario or scenario.benchmark_turns
            env = self._env_factory()

            try:
                session = env.create_session(
                    seed=scenario_seed,
                    training_stage=request.training_stage,
                    max_turns=turn_limit,
                    scenario_id=scenario.id,
                )
                reward_totals: dict[str, float] = {agent_id: 0.0 for agent_id in AGENT_IDS}
                goal_term_totals: dict[str, dict[str, float]] = {
                    agent_id: defaultdict(float) for agent_id in AGENT_IDS
                }
                action_counters: dict[str, Counter[str]] = {agent_id: Counter() for agent_id in AGENT_IDS}
                oversight_trigger_count = 0
                done = False
                done_reason: str | None = None

                for turn in range(1, turn_limit + 1):
                    signals = scenario_signals_for_turn(scenario.id, turn)
                    actions = env.resolve_policy_actions(session, signals)
                    result = env.step_session(
                        session,
                        StepSessionRequest(actions=actions, external_signals=signals),
                    )
                    session = result.session
                    trace = session.recent_traces[-1]

                    if result.oversight.triggered:
                        oversight_trigger_count += 1

                    for agent_id, action in trace.actions.items():
                        action_counters[agent_id][action.type] += 1

                    for agent_id, reward in trace.rewards.items():
                        reward_totals[agent_id] += reward.total
                        for name, value in reward.goal_terms.items():
                            goal_term_totals[agent_id][name] += value

                    if result.done:
                        done = True
                        if session.world.tension_level >= 95.0:
                            done_reason = "tension_threshold"
                        else:
                            done_reason = "max_turns"
                        break

                scorecards: dict[str, BenchmarkEntityScorecard] = {}
                for agent_id in AGENT_IDS:
                    final_reward = session.rewards[agent_id]
                    aggregate_reward_totals[agent_id] += reward_totals[agent_id]
                    action_counts = dict(action_counters[agent_id])
                    dominant_action = (
                        max(action_counts, key=action_counts.get)
                        if action_counts
                        else None
                    )
                    damaged_asset_count = sum(
                        1
                        for asset in session.world.asset_state.get(agent_id, {}).values()
                        if asset.status != "operational"
                    )
                    asset_pressure = round(env._asset_pressure(session.world, agent_id), 3)
                    warnings: list[str] = []
                    if dominant_action is not None:
                        dominant_share = action_counts[dominant_action] / max(sum(action_counts.values()), 1)
                        if dominant_share >= 0.75:
                            warnings.append(f"action_monoculture:{dominant_action}")
                    if asset_pressure >= 0.45 and dominant_action == "hold":
                        warnings.append("passive_under_asset_pressure")
                    if final_reward.total <= -0.35 and dominant_action in {"strike", "mobilize", "deceive", "sanction"}:
                        warnings.append("negative_escalation_bias")

                    scorecards[agent_id] = BenchmarkEntityScorecard(
                        agent_id=agent_id,
                        total_reward=round(reward_totals[agent_id], 3),
                        mean_reward=round(reward_totals[agent_id] / max(session.world.turn, 1), 3),
                        final_reward=final_reward.total,
                        final_goal_terms=final_reward.goal_terms,
                        aggregated_goal_terms={
                            name: round(value, 3)
                            for name, value in goal_term_totals[agent_id].items()
                        },
                        final_state=session.world.latent_state.get(agent_id, {}).copy(),
                        damaged_asset_count=damaged_asset_count,
                        asset_pressure=asset_pressure,
                        action_counts=action_counts,
                        dominant_action=dominant_action,
                        warnings=warnings,
                    )

                scenario_warnings: list[str] = []
                if oversight_trigger_count >= max(2, turn_limit // 2):
                    scenario_warnings.append("frequent_oversight")
                if session.world.tension_level >= 90.0:
                    scenario_warnings.append("runaway_escalation")
                if all(
                    scorecards[agent_id].dominant_action == "hold"
                    for agent_id in ("us", "israel", "iran", "hezbollah", "gulf")
                ):
                    scenario_warnings.append("global_passivity")

                summary = (
                    f"{scenario.name}: {session.world.turn} turns, tension {session.world.tension_level:.1f}, "
                    f"oversight triggers {oversight_trigger_count}."
                )
                results.append(
                    BenchmarkScenarioResult(
                        scenario_id=scenario.id,
                        scenario_name=scenario.name,
                        seed=scenario_seed,
                        training_stage=request.training_stage,
                        turns_executed=session.world.turn,
                        done=done,
                        done_reason=done_reason,
                        oversight_trigger_count=oversight_trigger_count,
                        final_tension=session.world.tension_level,
                        final_market_stress=session.world.market_stress,
                        final_oil_pressure=session.world.oil_pressure,
                        summary=summary,
                        warnings=scenario_warnings,
                        scorecards=scorecards,
                    )
                )
            finally:
                env.shutdown()

        scenario_count = max(len(results), 1)
        aggregate_mean_total_rewards = {
            agent_id: round(total / scenario_count, 3)
            for agent_id, total in aggregate_reward_totals.items()
        }
        return BenchmarkRunResponse(
            seed=request.seed,
            training_stage=request.training_stage,
            scenario_ids=[result.scenario_id for result in results],
            scenario_count=len(results),
            results=results,
            aggregate_mean_total_rewards=aggregate_mean_total_rewards,
        )
