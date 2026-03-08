from __future__ import annotations

import pytest

from trenches_env.openenv_client import TrenchesEnvClient
from trenches_env.training_cli import _validate_runtime_ports


def test_sync_client_uses_base_client_sync_wrapper_when_available() -> None:
    client = object.__new__(TrenchesEnvClient)
    sentinel = object()
    client.sync = lambda: sentinel  # type: ignore[attr-defined]

    assert client.sync_client() is sentinel


def test_validate_runtime_ports_rejects_server_mode_collision() -> None:
    with pytest.raises(RuntimeError, match="must differ"):
        _validate_runtime_ports(
            backend_port=8000,
            vllm_mode="server",
            vllm_server_port=8000,
        )


def test_validate_runtime_ports_allows_colocate_same_port() -> None:
    _validate_runtime_ports(
        backend_port=8000,
        vllm_mode="colocate",
        vllm_server_port=8000,
    )
