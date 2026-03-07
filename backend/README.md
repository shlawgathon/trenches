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

Relevant OpenEnv pieces in this package:

- `trenches_env.openenv_adapter.TrenchesOpenEnvEnvironment`
- `trenches_env.openenv_adapter.TrenchesOpenEnvAction`
- `trenches_env.openenv_adapter.TrenchesOpenEnvObservation`
- `trenches_env.openenv_adapter.TrenchesOpenEnvState`
- `trenches_env.openenv_client.TrenchesEnvClient`

Planned responsibilities:

- Hold in-memory crisis sessions.
- Expose `create`, `reset`, `step`, and `state` HTTP endpoints.
- Model the fog-of-war world state and per-agent observations.
- Provide a native OpenEnv boundary with scalar rewards for one active training agent while retaining full per-agent state internally.
- Provide extension points for World Monitor ingestion and RL training hooks.
