from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI
from pydantic import BaseModel, Field

from trenches_env.agents import AGENT_IDS
from trenches_env.env import FogOfWarDiplomacyEnv
from trenches_env.models import (
    AgentAction,
    AgentObservation,
    ExternalSignal,
    HistoricalEvent,
    HistoricalReplayState,
    OversightIntervention,
    Prediction,
    PredictionAssessment,
    RewardBreakdown,
    SessionState,
    StepSessionRequest,
)
from trenches_env.rl import DEFAULT_MAX_TURNS, DEFAULT_TRAINING_STAGE, TrainingStage
from trenches_env.session_manager import SessionManager
from trenches_env.source_ingestion import SourceHarvester

try:
    from openenv.core.env_server import create_app as create_openenv_app
    from openenv.core.env_server.interfaces import Environment as OpenEnvEnvironmentBase
    from openenv.core.env_server.types import Action as OpenEnvActionBase
    from openenv.core.env_server.types import Observation as OpenEnvObservationBase
    from openenv.core.env_server.types import State as OpenEnvStateBase

    OPENENV_CORE_AVAILABLE = True
except ImportError:
    OPENENV_CORE_AVAILABLE = False

    class OpenEnvActionBase(BaseModel):
        """Fallback shim used when openenv-core is not installed."""

    class OpenEnvObservationBase(BaseModel):
        """Fallback shim used when openenv-core is not installed."""

        reward: float = 0.0
        done: bool = False

    class OpenEnvStateBase(BaseModel):
        """Fallback shim used when openenv-core is not installed."""

        episode_id: str = ""
        step_count: int = 0

    class OpenEnvEnvironmentBase:
        """Fallback shim used when openenv-core is not installed."""

        SUPPORTS_CONCURRENT_SESSIONS = True

        @property
        def state(self) -> OpenEnvStateBase:
            raise NotImplementedError

        def reset(self, **kwargs: Any) -> OpenEnvObservationBase:
            raise NotImplementedError

        def step(self, action: OpenEnvActionBase, **kwargs: Any) -> OpenEnvObservationBase:
            raise NotImplementedError

        def close(self) -> None:
            return None

    def create_openenv_app(*args: Any, **kwargs: Any) -> FastAPI:  # type: ignore[misc]
        raise RuntimeError("openenv-core is not installed")


class TrenchesOpenEnvAction(OpenEnvActionBase):
    """Structured joint action payload for a single OpenEnv step.

    `action` is convenient for single-policy training.
    `actions` enables explicit joint rollouts or self-play.
    """

    action: AgentAction | None = None
    actions: dict[str, AgentAction] = Field(default_factory=dict)
    prediction: Prediction | None = None
    predictions: dict[str, Prediction] = Field(default_factory=dict)
    external_signals: list[ExternalSignal] = Field(default_factory=list)
    autofill_missing_with_policy: bool = True
    autofill_missing_with_hold: bool = True


class TrenchesOpenEnvObservation(OpenEnvObservationBase):
    session_id: str
    training_agent: str
    turn: int
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    agent_observation: AgentObservation
    joint_observations: dict[str, AgentObservation] = Field(default_factory=dict)
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    oversight: OversightIntervention = Field(default_factory=OversightIntervention)
    historical_replay: HistoricalReplayState = Field(default_factory=HistoricalReplayState)
    revealed_event: HistoricalEvent | None = None
    prediction_assessments: dict[str, PredictionAssessment] = Field(default_factory=dict)
    done_reason: str | None = None


class TrenchesOpenEnvState(OpenEnvStateBase):
    session_id: str = ""
    training_agent: str = "us"
    training_stage: TrainingStage = DEFAULT_TRAINING_STAGE
    max_turns: int = DEFAULT_MAX_TURNS
    live_enabled: bool = False
    reward_breakdowns: dict[str, RewardBreakdown] = Field(default_factory=dict)
    last_oversight: OversightIntervention = Field(default_factory=OversightIntervention)
    session: SessionState | None = None


class TrenchesOpenEnvEnvironment(OpenEnvEnvironmentBase):
    """Native OpenEnv environment boundary for the crisis simulator."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, env: FogOfWarDiplomacyEnv | None = None) -> None:
        self._env = env or FogOfWarDiplomacyEnv(source_harvester=SourceHarvester(auto_start=False))
        self._session: SessionState | None = None
        self._training_agent = "us"
        self._include_joint_observations = False
        self._last_oversight = OversightIntervention()

    @property
    def state(self) -> TrenchesOpenEnvState:
        if self._session is None:
            return TrenchesOpenEnvState(
                episode_id="",
                step_count=0,
                session_id="",
                training_agent=self._training_agent,
                training_stage=DEFAULT_TRAINING_STAGE,
                max_turns=DEFAULT_MAX_TURNS,
            )

        return TrenchesOpenEnvState(
            episode_id=self._session.session_id,
            step_count=self._session.world.turn,
            session_id=self._session.session_id,
            training_agent=self._training_agent,
            training_stage=self._session.episode.training_stage,
            max_turns=self._session.episode.max_turns,
            live_enabled=self._session.live.enabled,
            reward_breakdowns=self._session.rewards,
            last_oversight=self._last_oversight,
            session=self._session,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        episode_id: str | None = None,
        training_agent: str = "us",
        training_stage: TrainingStage = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
        scenario_id: str | None = None,
        replay_id: str | None = None,
        replay_start_index: int | None = None,
        include_joint_observations: bool = False,
        **_: Any,
    ) -> TrenchesOpenEnvObservation:
        self._training_agent = self._validate_training_agent(training_agent)
        self._include_joint_observations = include_joint_observations
        self._last_oversight = OversightIntervention()
        self._session = self._env.create_session(
            seed=seed,
            session_id=episode_id,
            training_agent=self._training_agent,
            training_stage=training_stage,
            max_turns=max_turns,
            scenario_id=scenario_id,
            replay_id=replay_id,
            replay_start_index=replay_start_index,
        )
        return self._build_observation(self._session)

    def step(self, action: TrenchesOpenEnvAction, **_: Any) -> TrenchesOpenEnvObservation:
        if self._session is None:
            raise RuntimeError("reset() must be called before step().")

        joint_actions = self._resolve_joint_actions(action)
        result = self._env.step_session(
            self._session,
            StepSessionRequest(
                actions=joint_actions,
                predictions=self._resolve_joint_predictions(action),
                external_signals=action.external_signals,
            ),
        )
        self._session = result.session
        self._last_oversight = result.oversight
        return self._build_observation(result.session)

    def close(self) -> None:
        self._env.shutdown()

    def _build_observation(self, session: SessionState) -> TrenchesOpenEnvObservation:
        reward_breakdown = session.rewards.get(self._training_agent, RewardBreakdown())
        done_reason: str | None = None
        if session.world.tension_level >= 95.0:
            done_reason = "tension_threshold"
        elif session.world.turn >= session.episode.max_turns:
            done_reason = "max_turns"
        elif (
            session.historical_replay.enabled
            and session.historical_replay.current_event_index >= len(session.historical_replay.ground_truth_timeline) - 1
        ):
            done_reason = "replay_complete"

        recent_trace = session.recent_traces[-1] if session.recent_traces else None

        return TrenchesOpenEnvObservation(
            session_id=session.session_id,
            training_agent=self._training_agent,
            turn=session.world.turn,
            training_stage=session.episode.training_stage,
            agent_observation=session.observations[self._training_agent],
            joint_observations=session.observations if self._include_joint_observations else {},
            reward=reward_breakdown.total,
            done=done_reason is not None,
            reward_breakdown=reward_breakdown,
            oversight=self._last_oversight,
            historical_replay=self._public_historical_replay(session.historical_replay),
            revealed_event=recent_trace.revealed_event if recent_trace is not None else session.historical_replay.last_revealed_event,
            prediction_assessments=recent_trace.prediction_assessments if recent_trace is not None else {},
            done_reason=done_reason,
        )

    def _resolve_joint_actions(self, action: TrenchesOpenEnvAction) -> dict[str, AgentAction]:
        joint_actions = dict(action.actions)
        if action.action is not None:
            joint_actions[action.action.actor] = action.action

        unknown_agents = sorted(set(joint_actions) - set(AGENT_IDS))
        if unknown_agents:
            raise ValueError(f"Unknown agents in joint action: {', '.join(unknown_agents)}")

        missing_agents = [agent_id for agent_id in AGENT_IDS if agent_id not in joint_actions]
        if missing_agents and action.autofill_missing_with_policy and self._session is not None:
            joint_actions = self._env.resolve_policy_actions(
                self._session,
                action.external_signals,
                preset_actions=joint_actions,
                agent_ids=missing_agents,
            )
        elif action.autofill_missing_with_hold:
            for agent_id in missing_agents:
                joint_actions[agent_id] = AgentAction(
                    actor=agent_id,
                    type="hold",
                    summary=f"Auto-filled hold for {agent_id} in the OpenEnv step.",
                )

        return joint_actions

    def _resolve_joint_predictions(self, action: TrenchesOpenEnvAction) -> dict[str, Prediction]:
        joint_predictions = dict(action.predictions)
        if action.prediction is not None:
            joint_predictions[action.prediction.agent_id] = action.prediction
        return joint_predictions

    @staticmethod
    def _validate_training_agent(training_agent: str) -> str:
        if training_agent not in AGENT_IDS:
            raise ValueError(f"Unknown training_agent: {training_agent}")
        return training_agent

    @staticmethod
    def _public_historical_replay(historical_replay: HistoricalReplayState) -> HistoricalReplayState:
        if not historical_replay.enabled:
            return historical_replay.model_copy(deep=True)
        return historical_replay.model_copy(
            update={
                "ground_truth_timeline": [],
            },
            deep=True,
        )


def create_openenv_fastapi_app(
    env_factory: Callable[[], TrenchesOpenEnvEnvironment] | None = None,
) -> FastAPI | None:
    if not OPENENV_CORE_AVAILABLE:
        return None

    factory = env_factory or TrenchesOpenEnvEnvironment
    return create_openenv_app(
        factory,
        TrenchesOpenEnvAction,
        TrenchesOpenEnvObservation,
        env_name="trenches",
    )


class OpenEnvAdapter:
    """Legacy tuple-based runtime kept for the existing frontend/session endpoints."""

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        self.session_manager = session_manager or SessionManager()
        self._current_session_id: str | None = None

    def reset(
        self,
        seed: int | None = None,
        training_agent: str = "us",
        training_stage: TrainingStage = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
        scenario_id: str | None = None,
        replay_id: str | None = None,
        replay_start_index: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        session = self.session_manager.create_session(
            seed=seed,
            training_agent=training_agent,
            training_stage=training_stage,
            max_turns=max_turns,
            scenario_id=scenario_id,
            replay_id=replay_id,
            replay_start_index=replay_start_index,
        )
        self._current_session_id = session.session_id
        return session.observations, self._build_info(session=session, oversight=OversightIntervention())

    def step(
        self,
        actions: dict[str, Any],
        predictions: dict[str, Prediction] | None = None,
        external_signals: list[ExternalSignal] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], bool, bool, dict[str, Any]]:
        if self._current_session_id is None:
            self.reset()

        result = self.session_manager.step_session(
            self._current_session_id,
            StepSessionRequest(
                actions=actions,
                predictions=predictions or {},
                external_signals=external_signals or [],
            ),
        )
        session = result.session
        terminated = session.world.tension_level >= 95.0
        truncated = session.world.turn >= session.episode.max_turns or (
            session.historical_replay.enabled
            and session.historical_replay.current_event_index >= len(session.historical_replay.ground_truth_timeline) - 1
        )
        return (
            session.observations,
            session.rewards,
            terminated,
            truncated,
            self._build_info(session=session, oversight=result.oversight),
        )

    def state(self) -> SessionState | None:
        if self._current_session_id is None:
            return None
        return self.session_manager.get_session(self._current_session_id)

    @staticmethod
    def _build_info(session: SessionState, oversight: OversightIntervention) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "turn": session.world.turn,
            "agent_ids": list(AGENT_IDS),
            "world": session.world.model_dump(mode="json"),
            "belief_state": {agent_id: state.model_dump(mode="json") for agent_id, state in session.belief_state.items()},
            "episode": session.episode.model_dump(mode="json"),
            "oversight": oversight.model_dump(mode="json"),
            "live": session.live.model_dump(mode="json"),
            "historical_replay": session.historical_replay.model_dump(mode="json"),
            "prediction_assessments": [assessment.model_dump(mode="json") for assessment in session.prediction_assessments[-5:]],
        }
