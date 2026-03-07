# Backend Summary

This file is the handoff document for anyone continuing the frontend against the current backend.

## What The Backend Is

The backend is a FastAPI service that exposes two related interfaces:

1. A session-oriented HTTP API for the dashboard and app frontend.
2. An OpenEnv-compatible environment boundary for training/runtime integration.

The frontend should treat the session API as the main source of truth for application state.

## Important Plain-English State

Before, the world state was too direct.

Now there are three separate layers:

1. `world.latent_state`
   This is the underlying backend truth used for rewards and internal simulation logic.

2. `world.actor_state`
   This is the lagged/public summary of the world.

3. `observations[agent_id]`
   This is what each entity actually sees. It can be partial, delayed, contested, and contradictory.

That means the frontend should not assume that:

- `observations[agent].strategic_state`
  equals
- `world.actor_state[agent]`
  or
- `world.latent_state[agent]`

That mismatch is intentional.

## Real Vs Fallback Behavior

The backend does not fake provider readiness.

Each entity now has a `model_bindings[agent_id]` object in session state and in `/capabilities`.

If a provider is not configured, the backend explicitly reports:

- `decision_mode = "heuristic_fallback"`
- `configured = false`

If a provider is configured, the backend reports:

- provider name
- model name
- base URL
- expected API key env var name
- whether tool calls / structured output are expected

This is a real plug-in point for later provider execution. It is not yet full provider-driven action generation.

Update:

There is now a real provider execution path in the backend.

If an entity has a configured binding, `resolve_policy_actions(...)` will attempt a real provider call first.
If that provider call fails, is misconfigured, or returns an illegal action, the env falls back explicitly to
heuristic policy selection and records the fallback in action metadata.

Provider runtime file:

- [backend/src/trenches_env/provider_runtime.py](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/provider_runtime.py)

Current provider support:

- OpenAI-compatible chat completions:
  - `openai`
  - `openrouter`
  - `ollama`
  - `vllm`
  - `custom`
- Anthropic messages API

Current execution behavior:

- preferred path: real provider inference
- fallback path: heuristic policy

Action metadata will show whether the action came from:

- `provider_inference`
- or `heuristic_fallback`

## Main Endpoints

Base server file:

- [backend/src/trenches_env/server.py](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/server.py)

### Health And Capabilities

- `GET /healthz`
  Returns `{ "status": "ok" }`

- `GET /capabilities`
  Returns:
  - CORS settings
  - whether session API is available
  - whether legacy tuple OpenEnv API is available
  - whether native `/openenv` is mounted
  - per-entity `model_bindings`

Frontend use:
Call this once at app startup to decide which runtime features to surface.

### Session Lifecycle

- `POST /sessions`
  Create a new session.

Request body:
- `seed?: number`
- `training_stage?: "stage_1_dense" | "stage_2_partial" | "stage_3_sparse"`
- `max_turns?: number`
- `scenario_id?: string`

Response:
- full `SessionState`

- `POST /sessions/{session_id}/reset`
  Reset an existing session with the same request shape as create.

- `GET /sessions/{session_id}`
  Get the latest session state.

- `POST /sessions/{session_id}/step`
  Advance one turn.

Request body:
- `actions: Record<agentId, AgentAction>`
- `external_signals: ExternalSignal[]`

Response:
- `StepSessionResponse`
  - `session`
  - `oversight`
  - `done`

### Live Source Controls

- `POST /sessions/{session_id}/live`
  Enable or disable live mode.

Request body:
- `enabled: boolean`
- `auto_step: boolean`
- `poll_interval_ms: number`

Response:
- full `SessionState`

- `POST /sessions/{session_id}/sources/refresh`
  Force source refresh and rebuild observations.

- `GET /sessions/{session_id}/sources/monitor`
  Returns source-health and delivery status per entity.

### Scenarios And Benchmarks

- `GET /scenarios`
  Returns available seeded scenarios.

- `POST /benchmarks/run`
  Runs benchmark scenarios and returns aggregate scorecards.

Request body:
- `scenario_ids?: string[]`
- `seed?: number`
- `training_stage?: TrainingStage`
- `steps_per_scenario?: number`

### Legacy OpenEnv Tuple API

- `POST /reset`
- `POST /step`
- `GET /state`

These exist for older clients. New frontend work should prefer `/sessions/...`.

### Native OpenEnv API

If `openenv-core` is installed, native OpenEnv is mounted at:

- `/openenv`

OpenEnv adapter file:

- [backend/src/trenches_env/openenv_adapter.py](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/openenv_adapter.py)

## Core Response Objects

Main schema file:

- [backend/src/trenches_env/models.py](/Users/alazarmanakelew/IdeaProjects/trenches/backend/src/trenches_env/models.py)

### SessionState

This is the main object the frontend should render.

Important fields:

- `session_id`
- `world`
- `observations`
- `rewards`
- `model_bindings`
- `episode`
- `recent_traces`
- `action_log`
- `live`
- `created_at`
- `updated_at`

### WorldState

Important fields:

- `turn`
- `tension_level`
- `market_stress`
- `oil_pressure`
- `latent_state`
- `actor_state`
- `asset_state`
- `coalition_graph`
- `active_events`
- `hidden_intents`
- `behavioral_consistency`
- `risk_scores`
- `last_actions`

Frontend note:

- `latent_state` is backend truth, useful for privileged/operator or admin/debug surfaces.
- `actor_state` is the public/lagged summary.
- normal entity-facing UI should rely most heavily on `observations`.

### AgentObservation

Per entity, this is the frontend-friendly model-facing view.

Important fields:

- `public_brief`
- `private_brief`
- `perceived_tension`
- `known_coalitions`
- `event_log`
- `decision_prompt`
- `available_actions`
- `available_data_sources`
- `strategic_state`
- `strategic_assets`
- `asset_alerts`
- `training_source_packets`
- `live_source_packets`
- `projection`

### ObservationProjection

This explains how “clean” or “messy” the current observation is.

Important fields:

- `enabled`
- `mode`
- `worldview_reliability`
- `delayed_source_count`
- `contested_source_count`
- `contradiction_packet_count`
- `contradiction_topics`
- `obscured_metric_count`
- `notes`

Frontend use:

Render this clearly. It explains why the entity may be making a decision under uncertainty.

### EntityModelBinding

Per entity provider/runtime config.

Important fields:

- `provider`
- `model_name`
- `base_url`
- `api_key_env`
- `configured`
- `ready_for_inference`
- `decision_mode`
- `supports_tool_calls`
- `supports_structured_output`
- `action_tools`
- `observation_tools`
- `notes`

Frontend use:

Use this for:

- backend readiness badges
- entity model/provider cards
- “heuristic fallback” warnings
- tool availability displays

### RewardBreakdown

Per entity reward object.

Important fields:

- `coalition_stability`
- `escalation_penalty`
- `market_gain`
- `behavioral_consistency`
- `goal_terms`
- `total`

Frontend use:

Show both `total` and `goal_terms`.
The goal terms are the entity-specific explanation of why reward moved.

### StepTrace

Rolling recent turn history.

Important fields:

- `turn`
- `tension_before`
- `tension_after`
- `actions`
- `rewards`
- `oversight`
- `created_at`

### ActionLogEntry

Human-readable per-action record.

Important fields:

- `turn`
- `actor`
- `action_type`
- `summary`
- `target`
- `reward_total`
- `tension_after`
- `market_stress_after`
- `oil_pressure_after`
- `metadata`
- `created_at`

Frontend use:

This is the easiest object for a nation activity feed / timeline.

## How Sources Work Now

Raw packets are still collected as real source packets.

Then the env turns those into model-facing briefs with:

- confidence shaping
- possible delay
- possible contested framing
- possible contradiction framing

This means two sources can now disagree about the same latent development.

Example:

- one source says a shipping disruption is getting worse
- another source says the same situation is partially stabilizing

That is intentional and now part of the observation layer.

## What The Frontend Should Build Against

The frontend should primarily use:

1. `/capabilities`
2. `/scenarios`
3. `/sessions`
4. `/sessions/{id}`
5. `/sessions/{id}/step`
6. `/sessions/{id}/live`
7. `/sessions/{id}/sources/monitor`

Recommended UI panels:

1. Global session header
   Show turn, tension, market stress, oil pressure, live mode.

2. Entity cards
   Show:
   - entity observation state
   - reward total
   - provider binding
   - action tools
   - source-health summary

3. Action timeline
   Use `action_log`.

4. Recent step trace
   Use `recent_traces`.

5. Source health view
   Use `/sources/monitor`.

6. Observation uncertainty view
   Use `observation.projection`.

7. Scenario picker / benchmark runner
   Use `/scenarios` and `/benchmarks/run`.

## What Is Already Good To Go

These backend pieces are already strong enough for frontend continuation:

- session lifecycle
- live source monitoring
- per-entity observations
- per-entity rewards
- per-entity action logging
- seeded scenarios
- benchmark runs
- latent state vs public state split
- provider binding surface
- provider-backed action execution path
- contradiction-aware source projection
- OpenEnv compatibility

## What Is Not Finished Yet

These are the main remaining backend gaps:

1. Provider-driven decision execution
   This now exists for configured entities, but still needs production hardening:
   - retry/backoff
   - rate-limit handling
   - richer telemetry
   - better invalid-output recovery
   - optional streaming / reasoning capture

2. Full replay diff persistence
   `recent_traces` and `action_log` are rolling windows, not full historical audit storage.

3. Rich contradiction modeling
   Contradictions are now explicit, but still rule-based rather than coming from a deeper latent event graph.

4. Frontend-specific projection UX
   The backend emits the data, but the frontend still needs to present it clearly.

## Next Backend Work

1. Add actual provider execution for configured entities.
   This is now implemented. The next step is to harden it for production and expose better diagnostics in the UI.

2. Define a strict action-output contract for provider-driven entities.
   The current contract already resolves to:
   - one allowed action
   - optional target
   - rationale summary
   The next step is to stabilize it further and optionally version it.

3. Persist richer replay state.
   Add:
   - input signals per turn
   - pre-override actions
   - richer world deltas
   - longer or persisted trace history

4. Deepen contradiction generation from latent event state.
   Right now it is deterministic and useful, but still a first-pass system.

## Next Frontend Work

1. Build the session shell around `/sessions` and `/sessions/{id}`.

2. Add a clear entity monitoring panel.
   Each entity card should show:
   - current projected strategic state
   - reward total
   - current provider binding
   - action tools
   - current action log row
   - projection reliability / contradiction topics

3. Add a source-health page or drawer.
   Use `/sessions/{id}/sources/monitor`.

4. Add a turn replay sidebar.
   Use `recent_traces` and `action_log`.

5. Add scenario and benchmark controls.
   Let the operator:
   - create a scenario session
   - run a benchmark
   - inspect per-entity scorecards

6. Add obvious visual treatment for uncertainty.
   Do not show entity observations as if they are perfect truth.
   Surface:
   - reliability
   - contested packet count
   - contradiction topics
   - delayed source count

7. Add provider readiness badges.
   Show:
   - configured / not configured
   - provider name
   - heuristic fallback vs provider-ready

## Recommended Frontend Rule Of Thumb

If the view is “what the entity believes,” use:

- `session.observations[agent_id]`

If the view is “what the operator/debugger sees,” use:

- `session.world`

If the view is “what the backend is prepared to execute,” use:

- `session.model_bindings`

If the view is “what just happened,” use:

- `session.action_log`
- `session.recent_traces`



Next backend work

- Harden provider execution: retries, rate-limit handling, timeout strategy, richer failure telemetry.
- Persist richer replay data beyond rolling recent_traces and action_log.
- Deepen contradiction modeling so it comes from a stronger latent event graph instead of only rule-based topic contradiction.
- Add a backend endpoint for provider diagnostics per entity if you want a cleaner frontend status panel than /capabilities.

Next frontend work

- Build the main app around /capabilities, /scenarios, /sessions, /sessions/{id}, /sessions/{id}/step, /sessions/{id}/live, and /sessions/{id}/sources/monitor.
- Show provider readiness per entity: provider name, model name, provider_inference vs heuristic_fallback, and available tools.
- Add an uncertainty panel per entity using observation.projection.
- Add a timeline/activity view from action_log and a turn replay view from recent_traces.
- Add a source-health panel using the monitor endpoint.
- Make the UI clearly distinguish operator truth (world) from entity belief (observations[agent]).