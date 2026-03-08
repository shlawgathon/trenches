from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import httpx

from trenches_env.models import (
    AgentAction,
    AgentObservation,
    EntityModelBinding,
    ExternalSignal,
    ProviderAgentDiagnostics,
)
from trenches_env.rl import AGENT_ALLOWED_ACTIONS

_OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openrouter", "huggingface", "ollama", "vllm", "custom"}
_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "huggingface": "https://router.huggingface.co/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "vllm": "http://127.0.0.1:8000/v1",
}
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_HF_ROUTING_POLICIES = {"fastest", "cheapest", "preferred"}


class ProviderDecisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderDecisionRequest:
    agent_id: str
    binding: EntityModelBinding
    observation: AgentObservation
    external_signals: list[ExternalSignal]


@dataclass
class _ProviderRuntimeStats:
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    consecutive_failures: int = 0
    total_latency_ms: float = 0.0
    last_latency_ms: float | None = None
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    last_error: str | None = None


class ProviderDecisionRuntime:
    def __init__(
        self,
        client: httpx.Client | None = None,
        timeout_seconds: float = 20.0,
        max_attempts: int = 2,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None
        self._max_attempts = max(1, max_attempts)
        self._stats: dict[str, _ProviderRuntimeStats] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def decide_action(self, request: ProviderDecisionRequest) -> AgentAction:
        binding = request.binding
        if not binding.ready_for_inference:
            raise ProviderDecisionError("binding is not ready for inference")

        stats = self._stats.setdefault(request.agent_id, _ProviderRuntimeStats())
        stats.request_count += 1
        last_error: str | None = None

        for attempt in range(1, self._max_attempts + 1):
            started = perf_counter()
            try:
                payload = self._request_payload(request)
                action = self._payload_to_action(request.agent_id, binding, payload)
                latency_ms = round((perf_counter() - started) * 1000.0, 2)
                self._record_success(stats, latency_ms)
                action.metadata.setdefault("provider_attempts", attempt)
                return action
            except ProviderDecisionError as exc:
                latency_ms = round((perf_counter() - started) * 1000.0, 2)
                last_error = str(exc)
                if attempt >= self._max_attempts or not self._is_retryable_error(exc):
                    self._record_error(stats, latency_ms, last_error)
                    raise
            except httpx.RequestError as exc:
                latency_ms = round((perf_counter() - started) * 1000.0, 2)
                last_error = f"provider network error: {exc}"
                if attempt >= self._max_attempts:
                    self._record_error(stats, latency_ms, last_error)
                    raise ProviderDecisionError(last_error) from exc

        self._record_error(stats, 0.0, last_error or "provider decision failed")
        raise ProviderDecisionError(last_error or "provider decision failed")

    def diagnostics(self, bindings: dict[str, EntityModelBinding]) -> list[ProviderAgentDiagnostics]:
        diagnostics: list[ProviderAgentDiagnostics] = []
        for agent_id, binding in bindings.items():
            stats = self._stats.get(agent_id, _ProviderRuntimeStats())
            diagnostics.append(
                ProviderAgentDiagnostics(
                    agent_id=agent_id,
                    provider=binding.provider,
                    model_name=binding.model_name,
                    configured=binding.configured,
                    ready_for_inference=binding.ready_for_inference,
                    decision_mode=binding.decision_mode,
                    status=self._status_for(binding, stats),
                    request_count=stats.request_count,
                    success_count=stats.success_count,
                    error_count=stats.error_count,
                    consecutive_failures=stats.consecutive_failures,
                    last_latency_ms=stats.last_latency_ms,
                    avg_latency_ms=round(stats.total_latency_ms / stats.success_count, 2) if stats.success_count else None,
                    last_success_at=stats.last_success_at,
                    last_error_at=stats.last_error_at,
                    last_error=stats.last_error,
                )
            )
        return diagnostics

    def _request_payload(self, request: ProviderDecisionRequest) -> dict[str, Any]:
        provider = request.binding.provider
        if provider in _OPENAI_COMPATIBLE_PROVIDERS:
            return self._request_openai_compatible(request)
        if provider == "anthropic":
            return self._request_anthropic(request)
        raise ProviderDecisionError(f"unsupported provider: {provider}")

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
            "model": self._resolved_model_name(binding),
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
        self._raise_for_status(response)
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

    @staticmethod
    def _resolved_model_name(binding: EntityModelBinding) -> str:
        if binding.provider != "huggingface":
            return binding.model_name
        policy = (os.getenv("TRENCHES_HF_ROUTING_POLICY") or "fastest").strip().lower()
        if ":" in binding.model_name or policy not in _HF_ROUTING_POLICIES:
            return binding.model_name
        return f"{binding.model_name}:{policy}"

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
        self._raise_for_status(response)
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
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            message = f"provider returned HTTP {exc.response.status_code}"
            raise ProviderDecisionError(message) from exc

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

    @staticmethod
    def _is_retryable_error(error: ProviderDecisionError) -> bool:
        message = str(error).lower()
        if "timeout" in message or "timed out" in message:
            return True
        if "connection" in message or "network" in message:
            return True
        if "http " in message:
            for status_code in _RETRYABLE_STATUS_CODES:
                if f"http {status_code}" in message:
                    return True
        return False

    @staticmethod
    def _record_success(stats: _ProviderRuntimeStats, latency_ms: float) -> None:
        stats.success_count += 1
        stats.consecutive_failures = 0
        stats.last_latency_ms = latency_ms
        stats.total_latency_ms += latency_ms
        stats.last_success_at = datetime.now(timezone.utc)
        stats.last_error = None

    @staticmethod
    def _record_error(stats: _ProviderRuntimeStats, latency_ms: float, error: str) -> None:
        stats.error_count += 1
        stats.consecutive_failures += 1
        stats.last_latency_ms = latency_ms if latency_ms > 0.0 else stats.last_latency_ms
        stats.last_error = error
        stats.last_error_at = datetime.now(timezone.utc)

    @staticmethod
    def _status_for(binding: EntityModelBinding, stats: _ProviderRuntimeStats) -> str:
        if not binding.ready_for_inference:
            return "fallback_only"
        if stats.request_count == 0:
            return "idle"
        if stats.consecutive_failures > 0:
            return "degraded"
        return "healthy"
