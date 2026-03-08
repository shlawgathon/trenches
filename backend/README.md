# Trenches Backend

This directory contains the Python backend for the Trenches simulator.

It now exposes two layers:

- the existing session-oriented FastAPI API used by the React dashboard
- a native OpenEnv-compatible environment mounted under `/openenv` when `openenv-core` is installed

The backend does not serve frontend assets and is intended to stay frontend-stack agnostic. Any web client
(Next.js, Vite, Bun, mobile, or a thin dashboard proxy) should be able to consume the same HTTP contract.

CORS is configurable so frontend migrations do not require backend code changes:

- `TRENCHES_CORS_ALLOW_ORIGINS=https://app.example.com,https://ops.example.com`
- `TRENCHES_CORS_ALLOW_ORIGIN_REGEX=https://.*\\.example\\.com`
- `TRENCHES_CORS_ALLOW_CREDENTIALS=true|false`

If no CORS env vars are set, the backend allows local development origins on `localhost` / `127.0.0.1` for any port.

Entity-model provider bindings are also configurable per agent. The repo now ships with bundled defaults for the
six self-hosted vLLM agents listed in `frontend.md`, and explicit env overrides still take precedence. If you set
`TRENCHES_MOCK_MODELS=true`, the backend now forces heuristic fallback bindings instead of routing to a hosted mock provider.

Supported env patterns:

- `TRENCHES_MODEL_PROVIDER=huggingface|ollama|vllm|custom`
- `TRENCHES_MODEL_NAME=<provider model id>`
- `TRENCHES_MODEL_BASE_URL=<custom base url>`
- `TRENCHES_MODEL_API_KEY_ENV=<name of env var holding the secret>`
- `TRENCHES_MODEL_SUPPORTS_TOOL_CALLS=true|false`
- `TRENCHES_MODEL_SUPPORTS_STRUCTURED_OUTPUT=true|false`

Bundled defaults if you do not set overrides:

- `us` -> `AlazarM/trenches-us-qwen3-8b-real`
- `israel` -> `AlazarM/trenches-israel-qwen3-8b-real`
- `iran` -> `AlazarM/trenches-iran-qwen3-8b-real`
- `hezbollah` -> `AlazarM/trenches-hezbollah-qwen3-8b-real`
- `gulf` -> `AlazarM/trenches-gulf-qwen3-8b-real`
- `oversight` -> `AlazarM/trenches-oversight-qwen3-8b-real`

Those defaults target the Cloudflare tunnel endpoints from `frontend.md` and append `/v1` for the OpenAI-compatible
vLLM API surface. No API key is attached for those bindings.

Per-entity overrides use the uppercase agent suffix, for example:

- `TRENCHES_MODEL_PROVIDER_US=vllm`
- `TRENCHES_MODEL_NAME_US=AlazarM/trenches-us-qwen3-8b-real`
- `TRENCHES_MODEL_BASE_URL_US=https://random-elephant-ranch-beverage.trycloudflare.com/v1`

Relevant OpenEnv pieces in this package:

- `trenches_env.openenv_adapter.TrenchesOpenEnvEnvironment`
- `trenches_env.openenv_adapter.TrenchesOpenEnvAction`
- `trenches_env.openenv_adapter.TrenchesOpenEnvObservation`
- `trenches_env.openenv_adapter.TrenchesOpenEnvState`
- `trenches_env.openenv_client.TrenchesEnvClient`

Historical replay training pieces:

- `trenches_env.models.Prediction`
- `trenches_env.models.HistoricalEvent`
- `trenches_env.models.HistoricalReplayState`
- `trenches_env.training_cli`

The backend now supports replay-aware forecast training:

- `reset(..., replay_id=...)` starts from a visible historical context event
- `step(...)` accepts separate `action` and `prediction`
- the next ground-truth event is revealed on the same OpenEnv step
- reward blends the entity action reward with forecast scoring terms

Bundled bootstrap replay (⚠️ **all replays are synthetic seed data** — replace with curated truth sets for production):

- `us_synthetic_seed_2025_2026`

CLI training entrypoint:

```bash
trenches-train \
  --training-agent us \
  --replay-id us_synthetic_seed_2025_2026 \
  --generation-backend transformers
```

The CLI supports two rollout backends:

- `transformers` for portable local smoke runs
- `vllm` for the documented colocated OpenEnv + TRL path on a GPU box

Planned responsibilities:

- Hold in-memory crisis sessions.
- Expose `create`, `reset`, `step`, and `state` HTTP endpoints.
- Model the fog-of-war world state and per-agent observations.
- Provide a native OpenEnv boundary with scalar rewards for one active training agent while retaining full per-agent state internally.
- Provide extension points for World Monitor ingestion and RL training hooks.
