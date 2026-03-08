from __future__ import annotations

import os

from trenches_env.agents import AGENT_IDS
from trenches_env.models import EntityModelBinding, ModelProviderName
from trenches_env.rl import AGENT_ALLOWED_ACTIONS

_OBSERVATION_TOOLS = [
    "inspect_public_brief",
    "inspect_private_brief",
    "inspect_assets",
    "inspect_sources",
    "emit_action",
]
_KNOWN_PROVIDERS: set[str] = {"none", "openai", "anthropic", "openrouter", "huggingface", "ollama", "vllm", "custom"}


def _env_value(base_name: str, agent_id: str) -> str | None:
    suffix = agent_id.upper()
    return os.getenv(f"{base_name}_{suffix}") or os.getenv(base_name)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _provider_name(value: str | None) -> ModelProviderName:
    normalized = (value or "none").strip().lower()
    if normalized not in _KNOWN_PROVIDERS:
        normalized = "custom"
    return normalized  # type: ignore[return-value]


def build_entity_model_bindings() -> dict[str, EntityModelBinding]:
    # --- Mock mode: route all entities to OpenRouter mock ---
    from trenches_env.mock.config import is_mock_enabled, build_mock_bindings
    if is_mock_enabled():
        return build_mock_bindings()
    # --- Normal mode: per-entity env-based bindings ---
    bindings: dict[str, EntityModelBinding] = {}
    for agent_id in AGENT_IDS:
        provider = _provider_name(_env_value("TRENCHES_MODEL_PROVIDER", agent_id))
        model_name = (_env_value("TRENCHES_MODEL_NAME", agent_id) or "").strip()
        base_url = (_env_value("TRENCHES_MODEL_BASE_URL", agent_id) or "").strip() or None
        api_key_env = (_env_value("TRENCHES_MODEL_API_KEY_ENV", agent_id) or "").strip() or None
        if provider == "huggingface" and api_key_env is None:
            api_key_env = "HF_TOKEN"
        configured = provider != "none" and bool(model_name)
        supports_tool_calls = _parse_bool(_env_value("TRENCHES_MODEL_SUPPORTS_TOOL_CALLS", agent_id), default=configured)
        supports_structured_output = _parse_bool(
            _env_value("TRENCHES_MODEL_SUPPORTS_STRUCTURED_OUTPUT", agent_id),
            default=configured,
        )

        notes: list[str] = []
        if not configured:
            notes.append("Provider binding is not configured; heuristic fallback remains active.")
        if configured and api_key_env is None and provider not in {"ollama", "vllm"}:
            notes.append("No API key environment variable declared for this provider binding.")
        if configured and provider == "huggingface" and api_key_env == "HF_TOKEN":
            notes.append("Hugging Face bindings default to HF_TOKEN unless overridden per entity.")
        if configured and not supports_tool_calls:
            notes.append("Provider is configured without tool-calling support; action selection must stay text-only.")

        bindings[agent_id] = EntityModelBinding(
            agent_id=agent_id,
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            api_key_env=api_key_env,
            configured=configured,
            ready_for_inference=configured,
            decision_mode="provider_ready" if configured else "heuristic_fallback",
            supports_tool_calls=supports_tool_calls,
            supports_structured_output=supports_structured_output,
            action_tools=list(AGENT_ALLOWED_ACTIONS.get(agent_id, ())),
            observation_tools=list(_OBSERVATION_TOOLS),
            notes=notes,
        )
    return bindings
