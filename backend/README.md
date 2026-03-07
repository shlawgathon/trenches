# Trenches Backend

This directory contains the initial Python scaffolding for the OpenEnv-style
session backend. It is intentionally lightweight and leaves the real OpenEnv
package integration as a follow-up once the dependency/version is pinned.

Planned responsibilities:

- Hold in-memory crisis sessions.
- Expose `create`, `reset`, `step`, and `state` HTTP endpoints.
- Model the fog-of-war world state and per-agent observations.
- Provide extension points for World Monitor ingestion and RL training hooks.
