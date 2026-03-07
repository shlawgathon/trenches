from __future__ import annotations

from openenv.core.env_client import EnvClient

from trenches_env.openenv_adapter import (
    TrenchesOpenEnvAction,
    TrenchesOpenEnvObservation,
    TrenchesOpenEnvState,
)


class TrenchesEnvClient(EnvClient[TrenchesOpenEnvAction, TrenchesOpenEnvObservation, TrenchesOpenEnvState]):
    """Typed OpenEnv client for the Trenches simulator."""

