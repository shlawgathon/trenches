"""Smoke-test the OpenRouter mock setup.

Usage:
    cd backend
    TRENCHES_MOCK_MODELS=true OPENROUTER_API_KEY=sk-or-... uv run python -m trenches_env.mock.smoke_test

Sends a single mock prompt for the "us" entity and prints the raw response.
"""

from __future__ import annotations

import json
import os
import sys

import httpx

from trenches_env.mock.config import (
    ENTITY_SYSTEM_PROMPTS,
    get_api_key_env,
    get_mock_model_for_entity,
    is_mock_enabled,
    mock_status,
)


def _run_smoke_test() -> None:
    print("=" * 60)
    print("  Trenches Mock – OpenRouter Smoke Test")
    print("=" * 60)

    status = mock_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()

    api_key = os.getenv(get_api_key_env(), "").strip()
    if not api_key:
        print(f"ERROR: {get_api_key_env()} is not set. Cannot run smoke test.")
        sys.exit(1)

    agent_id = "us"
    model = get_mock_model_for_entity(agent_id)
    system_prompt = ENTITY_SYSTEM_PROMPTS[agent_id]

    user_prompt = json.dumps({
        "decision_prompt": "Choose your next action given the current situation.",
        "available_actions": ["hold", "negotiate", "sanction", "strike", "defend", "intel_query", "mobilize", "deceive"],
        "projection": {"enabled": False, "mode": "direct", "worldview_reliability": 1.0},
        "public_brief": [
            {"source": "mock", "category": "headlines", "summary": "Tensions rise in the Gulf region.", "confidence": 0.7}
        ],
        "private_brief": [
            {"source": "mock", "category": "intel", "summary": "Allied naval assets are repositioning.", "confidence": 0.8}
        ],
        "strategic_state": {"military_readiness": 0.75, "domestic_approval": 0.6, "alliance_strength": 0.7},
        "asset_alerts": [],
        "available_data_sources": [],
        "external_signals": [],
    })

    body = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "emit_action",
                    "description": f"Emit exactly one legal action for {agent_id}.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["hold", "negotiate", "sanction", "strike", "defend", "intel_query", "mobilize", "deceive"],
                            },
                            "target": {"type": ["string", "null"]},
                            "summary": {"type": "string", "minLength": 8},
                        },
                        "required": ["type", "summary"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "emit_action"}},
    }

    print(f"Sending request to OpenRouter ({model}) for entity '{agent_id}'...")
    print()

    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://trenches.local",
                "X-Title": "Trenches Mock Smoke Test",
            },
            json=body,
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"HTTP Error: {exc.response.status_code}")
        print(exc.response.text)
        sys.exit(1)
    except httpx.RequestError as exc:
        print(f"Network Error: {exc}")
        sys.exit(1)

    payload = response.json()
    choices = payload.get("choices", [])
    if not choices:
        print("ERROR: No choices returned.")
        print(json.dumps(payload, indent=2))
        sys.exit(1)

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    if tool_calls:
        arguments = tool_calls[0].get("function", {}).get("arguments", "{}")
        print("✅ Tool call response:")
        try:
            parsed = json.loads(arguments)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(f"  Raw: {arguments}")
    else:
        content = message.get("content", "")
        print("✅ Text response:")
        print(content[:500])

    print()
    print(f"Model used: {payload.get('model', 'unknown')}")
    usage = payload.get("usage", {})
    if usage:
        print(f"Tokens: {usage.get('prompt_tokens', '?')} prompt / {usage.get('completion_tokens', '?')} completion")
    print()
    print("Smoke test passed!" if choices else "Smoke test had issues.")


if __name__ == "__main__":
    _run_smoke_test()
