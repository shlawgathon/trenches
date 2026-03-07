from __future__ import annotations

from typing import Any

from trenches_env.agents import AGENT_IDS
from trenches_env.models import ExternalSignal, OversightIntervention, SessionState, StepSessionRequest
from trenches_env.rl import DEFAULT_TRAINING_STAGE, TrainingStage
from trenches_env.session_manager import SessionManager


class OpenEnvAdapter:
    """Minimal Gym/OpenEnv-style runtime built on top of session management."""

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        self.session_manager = session_manager or SessionManager()
        self._current_session_id: str | None = None

    def reset(
        self,
        seed: int | None = None,
        training_stage: TrainingStage = DEFAULT_TRAINING_STAGE,
        max_turns: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        session = self.session_manager.create_session(
            seed=seed,
            training_stage=training_stage,
            max_turns=max_turns,
        )
        self._current_session_id = session.session_id
        return session.observations, self._build_info(session=session, oversight=OversightIntervention())

    def step(
        self,
        actions: dict[str, Any],
        external_signals: list[ExternalSignal] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], bool, bool, dict[str, Any]]:
        if self._current_session_id is None:
            self.reset()

        result = self.session_manager.step_session(
            self._current_session_id,
            StepSessionRequest(
                actions=actions,
                external_signals=external_signals or [],
            ),
        )
        session = result.session
        terminated = session.world.tension_level >= 95.0
        truncated = session.world.turn >= session.episode.max_turns
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
            "episode": session.episode.model_dump(mode="json"),
            "oversight": oversight.model_dump(mode="json"),
            "live": session.live.model_dump(mode="json"),
        }
