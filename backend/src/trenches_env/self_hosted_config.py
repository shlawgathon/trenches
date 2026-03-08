from __future__ import annotations

from trenches_env.agents import AGENT_IDS, AgentId


def _vllm_base_url(endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/v1"


SELF_HOSTED_BINDINGS: dict[AgentId, dict[str, str]] = {
    "us": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-us-qwen3-8b-real",
        "base_url": _vllm_base_url("https://random-elephant-ranch-beverage.trycloudflare.com"),
    },
    "israel": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-israel-qwen3-8b-real",
        "base_url": _vllm_base_url("https://cdna-dancing-discussion-claimed.trycloudflare.com"),
    },
    "iran": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-iran-qwen3-8b-real",
        "base_url": _vllm_base_url("https://months-flash-functional-overhead.trycloudflare.com"),
    },
    "hezbollah": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-hezbollah-qwen3-8b-real",
        "base_url": _vllm_base_url("https://fool-conducted-occurs-occurring.trycloudflare.com"),
    },
    "gulf": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-gulf-qwen3-8b-real",
        "base_url": _vllm_base_url("https://responsibility-cowboy-collar-does.trycloudflare.com"),
    },
    "oversight": {
        "provider": "vllm",
        "model_name": "AlazarM/trenches-oversight-qwen3-8b-real",
        "base_url": _vllm_base_url("https://upc-postcards-earnings-suppose.trycloudflare.com"),
    },
}


def default_self_hosted_binding(agent_id: str) -> dict[str, str]:
    if agent_id not in AGENT_IDS:
        return {}
    return SELF_HOSTED_BINDINGS.get(agent_id, {}).copy()  # type: ignore[return-value]
