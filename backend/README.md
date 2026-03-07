# Trenches Backend

This directory contains the Python backend for the Trenches simulator.

It now exposes two layers:

- the existing session-oriented FastAPI API used by the React dashboard
- a native OpenEnv-compatible environment mounted under `/openenv` when `openenv-core` is installed

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
