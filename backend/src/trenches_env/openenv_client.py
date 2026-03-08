from __future__ import annotations

from typing import Any, Dict

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
except ImportError:
    class EnvClient:  # type: ignore[no-redef]
        def __class_getitem__(cls, _item):
            return cls

        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("openenv-core is not installed")

from trenches_env.openenv_adapter import (
    TrenchesOpenEnvAction,
    TrenchesOpenEnvObservation,
    TrenchesOpenEnvState,
)


class TrenchesEnvClient(EnvClient[TrenchesOpenEnvAction, TrenchesOpenEnvObservation, TrenchesOpenEnvState]):
    """Typed OpenEnv client for the Trenches simulator."""

    def sync_client(self):
        """Return the documented synchronous wrapper when the base client provides one."""
        sync_factory = getattr(self, "sync", None)
        if callable(sync_factory):
            return sync_factory()
        return self

    def _step_payload(self, action: TrenchesOpenEnvAction) -> Dict[str, Any]:
        """Convert a TrenchesOpenEnvAction to the JSON data expected by the env server."""
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TrenchesOpenEnvObservation]:
        """Convert a JSON response from the env server to StepResult."""
        return StepResult(
            observation=TrenchesOpenEnvObservation.model_validate(payload["observation"]),
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TrenchesOpenEnvState:
        """Convert a JSON response from the state endpoint to a State object."""
        return TrenchesOpenEnvState.model_validate(payload)
