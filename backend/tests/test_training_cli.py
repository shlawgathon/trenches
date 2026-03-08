from __future__ import annotations

import pytest

from trenches_env.openenv_client import TrenchesEnvClient
from trenches_env.training_cli import (
    _generate_rollout_completions_vllm_server,
    _resolve_optimizer,
    _validate_runtime_ports,
)


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(self, *, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)


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


def test_resolve_optimizer_prefers_fused_on_cuda() -> None:
    assert _resolve_optimizer("auto", _FakeTorch(cuda_available=True)) == "adamw_torch_fused"


def test_resolve_optimizer_falls_back_to_adamw_torch_without_cuda() -> None:
    assert _resolve_optimizer("auto", _FakeTorch(cuda_available=False)) == "adamw_torch"


def test_resolve_optimizer_preserves_explicit_value() -> None:
    assert _resolve_optimizer("adamw_bnb_8bit", _FakeTorch(cuda_available=False)) == "adamw_bnb_8bit"


def test_vllm_server_rollout_keeps_prompt_ids_per_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[list[int]]]:
            return {
                "prompt_ids": [[11, 12], [21, 22]],
                "completion_ids": [[101], [102], [201], [202]],
                "logprobs": [[-0.1], [-0.2], [-0.3], [-0.4]],
            }

    def _fake_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeResponse()

    monkeypatch.setattr("trenches_env.training_cli.httpx.post", _fake_post)

    result = _generate_rollout_completions_vllm_server(
        prompts=["prompt-a", "prompt-b"],
        server_base_url="http://127.0.0.1:8001",
        num_generations=2,
        temperature=0.8,
        top_p=0.95,
        top_k=10,
        max_completion_length=64,
    )

    assert result["prompt_ids"] == [[11, 12], [21, 22]]
    assert result["completion_ids"] == [[101], [102], [201], [202]]
    assert result["logprobs"] == [[-0.1], [-0.2], [-0.3], [-0.4]]
