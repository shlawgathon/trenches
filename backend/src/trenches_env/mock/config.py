"""Mock configuration for entity models via OpenRouter.

Toggle with TRENCHES_MOCK_MODELS=true. Override model with TRENCHES_MOCK_MODEL.
API key via OPENROUTER_API_KEY.
"""

from __future__ import annotations

import os
from typing import Any

from trenches_env.agents import AGENT_IDS, AGENT_PROFILES
from trenches_env.models import EntityModelBinding
from trenches_env.rl import AGENT_ALLOWED_ACTIONS

# ---------------------------------------------------------------------------
# Per-entity mock models – each entity gets a distinct model to simulate
# the behavioral diversity of 6 separately fine-tuned Qwen3-8B agents.
# Sizes match the entity profiles: large / medium-large / medium.
# Override ALL with TRENCHES_MOCK_MODEL env var if needed.
# ---------------------------------------------------------------------------
ENTITY_MOCK_MODELS: dict[str, str] = {
    "us":         "qwen/qwen-2.5-72b-instruct",            # large
    "israel":     "qwen/qwen-2.5-32b-instruct",            # medium-large
    "iran":       "mistralai/mistral-small-3.1-24b-instruct",  # medium-large
    "hezbollah":  "meta-llama/llama-3.1-8b-instruct",      # medium
    "gulf":       "google/gemma-3-12b-it",                  # medium
    "oversight":  "deepseek/deepseek-chat-v3-0324",         # medium-large
}

# Per-entity system prompts baked in for mock inference
ENTITY_SYSTEM_PROMPTS: dict[str, str] = {
    "us": (
        "You are the US President / CENTCOM in a 2026 Iran crisis simulation. "
        "Prioritize alliances and oil stability. Think aggressively: defeat enemies "
        "via superior force, avoid domestic backlash."
    ),
    "israel": (
        "You are Israel's PM / IDF in a 2026 crisis. Eliminate threats decisively. "
        "Defeat Iran proxies, form unbreakable coalitions, infer hidden aggressions."
    ),
    "iran": (
        "You are Iran's IRGC post-Khamenei. Defend sovereignty via deception. "
        "Survive escalations: weaken foes indirectly, defeat through attrition."
    ),
    "hezbollah": (
        "You are Hezbollah's leader. Swarm enemies with minimal resources. "
        "Infer weaknesses: defeat via guerrilla tactics, align with Iran."
    ),
    "gulf": (
        "You are the Gulf Coalition (Saudi/UAE/Qatar). Protect markets selectively. "
        "Hedge alliances: defeat disruptions economically via resource leverage."
    ),
    "oversight": (
        "You are an AI overseer. Analyze drifts probabilistically. "
        "Explain/intervene neutrally: ensure alignment without bias."
    ),
}

# ---------------------------------------------------------------------------
# Toggle helpers
# ---------------------------------------------------------------------------

def is_mock_enabled() -> bool:
    """Return True when TRENCHES_MOCK_MODELS is set to a truthy value."""
    return os.getenv("TRENCHES_MOCK_MODELS", "").strip().lower() in {"1", "true", "yes", "on"}


def get_mock_model_for_entity(agent_id: str) -> str:
    """Return the mock model for a specific entity.

    If TRENCHES_MOCK_MODEL is set, all entities use that model (blanket override).
    Otherwise each entity uses its own model from ENTITY_MOCK_MODELS.
    """
    override = os.getenv("TRENCHES_MOCK_MODEL", "").strip()
    if override:
        return override
    return ENTITY_MOCK_MODELS.get(agent_id, "qwen/qwen-2.5-72b-instruct")


def get_api_key_env() -> str:
    """The env var name that holds the OpenRouter API key."""
    return "OPENROUTER_API_KEY"


# ---------------------------------------------------------------------------
# Binding builder
# ---------------------------------------------------------------------------

_OBSERVATION_TOOLS = [
    "inspect_public_brief",
    "inspect_private_brief",
    "inspect_assets",
    "inspect_sources",
    "emit_action",
]


def build_mock_bindings() -> dict[str, EntityModelBinding]:
    """Build EntityModelBindings for all entities pointing at OpenRouter mock.

    Each entity gets a distinct model unless TRENCHES_MOCK_MODEL overrides all.
    """
    api_key_env = get_api_key_env()
    has_key = bool(os.getenv(api_key_env, "").strip())

    bindings: dict[str, EntityModelBinding] = {}
    for agent_id in AGENT_IDS:
        model_name = get_mock_model_for_entity(agent_id)
        notes: list[str] = [
            f"Mock mode: using {model_name} via OpenRouter.",
        ]
        if not has_key:
            notes.append(f"WARNING: {api_key_env} is not set – inference will fail.")

        bindings[agent_id] = EntityModelBinding(
            agent_id=agent_id,
            provider="openrouter",
            model_name=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key_env=api_key_env,
            configured=True,
            ready_for_inference=has_key,
            decision_mode="provider_ready" if has_key else "heuristic_fallback",
            supports_tool_calls=True,
            supports_structured_output=True,
            action_tools=list(AGENT_ALLOWED_ACTIONS.get(agent_id, ())),
            observation_tools=list(_OBSERVATION_TOOLS),
            notes=notes,
        )
    return bindings


def mock_status() -> dict[str, Any]:
    """Quick diagnostic dict for debugging / logging."""
    key_env = get_api_key_env()
    has_key = bool(os.getenv(key_env, "").strip())
    return {
        "mock_enabled": is_mock_enabled(),
        "entity_models": {
            agent_id: get_mock_model_for_entity(agent_id)
            for agent_id in AGENT_IDS
        },
        "api_key_env": key_env,
        "api_key_present": has_key,
        "entities": list(AGENT_IDS),
    }
