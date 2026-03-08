from __future__ import annotations

from trenches_env.agents import AGENT_IDS, AgentId


def _vllm_base_url(endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/v1"


SELF_HOSTED_BINDINGS: dict[AgentId, dict[str, str]] = {
    "us": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-us-qwen3-8b-real",
        "base_url": _vllm_base_url("https://trendsweep--trenches-us.modal.run"),
    },
    "israel": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-israel-qwen3-8b-real",
        "base_url": _vllm_base_url("https://trendsweep--trenches-israel.modal.run"),
    },
    "iran": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-iran-qwen3-8b-real",
        "base_url": _vllm_base_url("https://trendsweep--trenches-iran.modal.run"),
    },
    "hezbollah": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-hezbollah-qwen3-8b-real",
        "base_url": _vllm_base_url("https://trendsweep--trenches-hezbollah.modal.run"),
    },
    "gulf": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-gulf-qwen3-8b-real",
        "base_url": _vllm_base_url("https://trendsweep--trenches-gulf.modal.run"),
    },
    "oversight": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-oversight-qwen3-8b-real",
        "base_url": _vllm_base_url("https://trendsweep--trenches-oversight.modal.run"),
    },
}


def default_self_hosted_binding(agent_id: str) -> dict[str, str]:
    if agent_id not in AGENT_IDS:
        return {}
    return SELF_HOSTED_BINDINGS.get(agent_id, {}).copy()  # type: ignore[return-value]
