from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from trenches_env.models import AgentAction, AgentObservation, EntityModelBinding, ExternalSignal
from trenches_env.rl import AGENT_ALLOWED_ACTIONS

_OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openrouter", "ollama", "vllm", "custom"}
_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "vllm": "http://127.0.0.1:8000/v1",
}


class ProviderDecisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderDecisionRequest:
    agent_id: str
    binding: EntityModelBinding
    observation: AgentObservation
    external_signals: list[ExternalSignal]


class ProviderDecisionRuntime:
    def __init__(self, client: httpx.Client | None = None, timeout_seconds: float = 20.0) -> None:
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def decide_action(self, request: ProviderDecisionRequest) -> AgentAction:
        binding = request.binding
        if not binding.ready_for_inference:
            raise ProviderDecisionError("binding is not ready for inference")

        provider = binding.provider
        if provider in _OPENAI_COMPATIBLE_PROVIDERS:
            payload = self._request_openai_compatible(request)
        elif provider == "anthropic":
            payload = self._request_anthropic(request)
        else:
            raise ProviderDecisionError(f"unsupported provider: {provider}")

        return self._payload_to_action(request.agent_id, binding, payload)

    def _request_openai_compatible(self, request: ProviderDecisionRequest) -> dict[str, Any]:
        binding = request.binding
        base_url = (binding.base_url or _DEFAULT_BASE_URLS.get(binding.provider) or "").rstrip("/")
        if not base_url:
            raise ProviderDecisionError("missing base_url for provider binding")

        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = self._resolve_api_key(binding)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if binding.provider == "openrouter":
            headers.setdefault("HTTP-Referer", "https://trenches.local")
            headers.setdefault("X-Title", "Trenches")

        body: dict[str, Any] = {
            "model": binding.model_name,
            "temperature": 0.1,
            "messages": self._messages(request),
        }
        if binding.supports_tool_calls:
            body["tools"] = [self._openai_emit_action_tool(request.agent_id)]
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": "emit_action"},
            }

        response = self._client.post(url, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise ProviderDecisionError("provider returned no choices")
        message = choices[0].get("message") or {}

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            arguments = tool_calls[0].get("function", {}).get("arguments", "{}")
            return self._parse_json_payload(arguments)

        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise ProviderDecisionError("provider returned empty message content")
        return self._parse_json_payload(content)

    def _request_anthropic(self, request: ProviderDecisionRequest) -> dict[str, Any]:
        binding = request.binding
        base_url = (binding.base_url or "https://api.anthropic.com/v1").rstrip("/")
        api_key = self._resolve_api_key(binding)
        if not api_key:
            raise ProviderDecisionError("anthropic provider requires an API key")

        body: dict[str, Any] = {
            "model": binding.model_name,
            "max_tokens": 350,
            "temperature": 0.1,
            "system": self._system_prompt(request.agent_id),
            "messages": [{"role": "user", "content": self._user_prompt(request)}],
        }
        if binding.supports_tool_calls:
            body["tools"] = [self._anthropic_emit_action_tool(request.agent_id)]

        response = self._client.post(
            f"{base_url}/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("content") or []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return block.get("input") or {}

        text_blocks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if not text_blocks:
            raise ProviderDecisionError("anthropic provider returned no usable content")
        return self._parse_json_payload("\n".join(text_blocks))

    @staticmethod
    def _resolve_api_key(binding: EntityModelBinding) -> str | None:
        if not binding.api_key_env:
            return None
        value = os.getenv(binding.api_key_env)
        return value.strip() if value else None

    @staticmethod
    def _messages(request: ProviderDecisionRequest) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": ProviderDecisionRuntime._system_prompt(request.agent_id)},
            {"role": "user", "content": ProviderDecisionRuntime._user_prompt(request)},
        ]

    @staticmethod
    def _system_prompt(agent_id: str) -> str:
        return (
            f"You are the decision runtime for {agent_id} in a geopolitical simulation. "
            "Choose exactly one legal action. Do not invent actions, targets, or tools that are not provided. "
            "If using text output, return strict JSON with keys type, target, and summary."
        )

    @staticmethod
    def _user_prompt(request: ProviderDecisionRequest) -> str:
        observation = request.observation
        payload = {
            "decision_prompt": observation.decision_prompt,
            "available_actions": observation.available_actions,
            "projection": observation.projection.model_dump(mode="json"),
            "public_brief": [brief.model_dump(mode="json") for brief in observation.public_brief[:4]],
            "private_brief": [brief.model_dump(mode="json") for brief in observation.private_brief[:6]],
            "strategic_state": observation.strategic_state,
            "asset_alerts": observation.asset_alerts[:6],
            "available_data_sources": [source.model_dump(mode="json") for source in observation.available_data_sources[:8]],
            "external_signals": [signal.model_dump(mode="json") for signal in request.external_signals[:6]],
        }
        return json.dumps(payload, ensure_ascii=True)

    @staticmethod
    def _openai_emit_action_tool(agent_id: str) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "emit_action",
                "description": f"Emit exactly one legal action for {agent_id}.",
                "parameters": ProviderDecisionRuntime._action_schema(agent_id),
            },
        }

    @staticmethod
    def _anthropic_emit_action_tool(agent_id: str) -> dict[str, Any]:
        return {
            "name": "emit_action",
            "description": f"Emit exactly one legal action for {agent_id}.",
            "input_schema": ProviderDecisionRuntime._action_schema(agent_id),
        }

    @staticmethod
    def _action_schema(agent_id: str) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": list(AGENT_ALLOWED_ACTIONS.get(agent_id, ())),
                },
                "target": {
                    "type": ["string", "null"],
                },
                "summary": {
                    "type": "string",
                    "minLength": 8,
                },
            },
            "required": ["type", "summary"],
            "additionalProperties": False,
        }

    @staticmethod
    def _parse_json_payload(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.startswith("```")]
            text = "\n".join(lines).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderDecisionError(f"provider returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ProviderDecisionError("provider payload must be a JSON object")
        return payload

    @staticmethod
    def _payload_to_action(agent_id: str, binding: EntityModelBinding, payload: dict[str, Any]) -> AgentAction:
        action_type = payload.get("type")
        summary = str(payload.get("summary", "")).strip()
        target = payload.get("target")

        if not isinstance(action_type, str) or action_type not in AGENT_ALLOWED_ACTIONS.get(agent_id, ()):
            raise ProviderDecisionError(f"provider selected illegal action for {agent_id}: {action_type}")
        if not summary:
            raise ProviderDecisionError("provider did not return an action summary")
        if target is not None and not isinstance(target, str):
            raise ProviderDecisionError("provider target must be a string or null")

        return AgentAction(
            actor=agent_id,
            type=action_type,
            target=target,
            summary=summary,
            metadata={
                "mode": "provider_inference",
                "provider": binding.provider,
                "model": binding.model_name,
            },
        )
