from __future__ import annotations

try:
    from openenv.core.env_client import EnvClient
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
